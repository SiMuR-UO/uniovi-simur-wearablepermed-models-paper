import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping

ACTIVITIES = sorted(['FASE REPOSO CON K5', 'TAPIZ RODANTE',
                     'INCREMENTAL CICLOERGOMETRO', 'YOGA', 'SENTADO VIENDO LA TV',
                     'SENTADO LEYENDO', 'SENTADO USANDO PC', 'DE PIE USANDO PC',
                     'DE PIE DOBLANDO TOALLAS', 'DE PIE MOVIENDO LIBROS',
                     'DE PIE BARRIENDO', 'CAMINAR USUAL SPEED',
                     'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR CON LA COMPRA',
                     'CAMINAR ZIGZAG', 'TROTAR', 'SUBIR Y BAJAR ESCALERAS'])

ACTIVITIES_TO_BE_REMOVED=['TAPIZ RODANTE', 'YOGA']

SUPERCLASES_CAPTURED24 = sorted(['WALKING', 'HOUSEHOLD-CHORES',
                                 'STANDING', 'SLEEP', 'BICYCLING',
                                 'SITTING', 'MIXED-ACTIVITY', 'SPORTS'])

MAPPING_CAPTURED24 = {
    # WALKING
    'CAMINAR CON MÓVIL O LIBRO': 'WALKING',
    'CAMINAR CON LA COMPRA': 'WALKING',
    'CAMINAR ZIGZAG': 'WALKING',
    'CAMINAR USUAL SPEED': 'WALKING',    

    # HOUSEHOLD-CHORES
    'DE PIE BARRIENDO': 'HOUSEHOLD-CHORES',
    'DE PIE MOVIENDO LIBROS': 'HOUSEHOLD-CHORES',
    'DE PIE DOBLANDO TOALLAS': 'HOUSEHOLD-CHORES',
    
    # STANDING
    'DE PIE USANDO PC': 'STANDING',

    # SLEEP
    'FASE REPOSO CON K5': 'SLEEP',

    # BICYCLING
    'INCREMENTAL CICLOERGOMETRO': 'BICYCLING',

    # SITTING
    'SENTADO LEYENDO': 'SITTING',
    'SENTADO USANDO PC': 'SITTING',
    'SENTADO VIENDO LA TV': 'SITTING',

    # MIXED-ACTIVITY
    'SUBIR Y BAJAR ESCALERAS': 'MIXED-ACTIVITY',

    # SPORTS
    'TROTAR': 'SPORTS',
}

SUPERCLASES_CPA_METS = ['SEDENTARY', 'LIGHT-INTENSITY',
                        'MODERATE-INTENSITY', 'VIGOROUS-INTENSITY']

MAPPING_CPA_METS = {
    # SEDENTARY
    'FASE REPOSO CON K5': 'SEDENTARY',
    'SENTADO LEYENDO': 'SEDENTARY',
    'SENTADO USANDO PC': 'SEDENTARY',
    'SENTADO VIENDO LA TV': 'SEDENTARY',    

    # LIGHT-INTENSITY
    'DE PIE DOBLANDO TOALLAS': 'LIGHT-INTENSITY',
    'DE PIE USANDO PC': 'LIGHT-INTENSITY',
    'CAMINAR CON MÓVIL O LIBRO': 'LIGHT-INTENSITY',
    'CAMINAR ZIGZAG': 'LIGHT-INTENSITY',
    
    # MODERATE-INTENSITY
    'DE PIE BARRIENDO': 'MODERATE-INTENSITY',
    'DE PIE MOVIENDO LIBROS': 'MODERATE-INTENSITY',
    'CAMINAR CON LA COMPRA': 'MODERATE-INTENSITY',
    'CAMINAR USUAL SPEED': 'MODERATE-INTENSITY',
    'SUBIR Y BAJAR ESCALERAS': 'MODERATE-INTENSITY',

    # VIGOROUS-INTENSITY
    'INCREMENTAL CICLOERGOMETRO': 'VIGOROUS-INTENSITY',  
    'TROTAR': 'VIGOROUS-INTENSITY' 
}

WINDOW_DATA = "arr_0"
WINDOW_LABELS = "arr_1"
WINDOW_METADATA = "arr_2"

CSV_FILE_NAME = "metrics_pi_m_c_loocv_all.csv"

metrics = []

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Mixture of Experts with autoencoders experts")

    parser.add_argument(
        "-stack-all",
        "--stack-all",
        required=True,        
        dest="stack_all",       
        help=f"Participant stack data."        
    )
    parser.add_argument(
        "-superclases",
        "--superclases",
        dest="superclases",        
        help=f"Use Superclases: WearablePerMed, Captured24, CPA-METS"
    )
    parser.add_argument(
        "-loops",
        "--loops",
        dest="loops",        
        type=int,        
        default=30,        
        help="Number of loops."
    )     
    parser.add_argument(
        "-optimize-trials",
        "--optimize-trials",               
        dest="optimize_trials",
        type=int,
        default=1,               
        help=f"Optimize hyperparameters num trials."        
    )
    parser.add_argument(
        '-generate-plots',
        '--generate-plots',
        dest='generate_plots',
        action='store_true',
        default=False,
        help="Generate Plots"
    )                   
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        action="store_const",
        const=logging.INFO,
        help="set log level to verbose."
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        action="store_const",
        const=logging.DEBUG,
        help="set log level to very verbose."        
    )

    return parser.parse_args(args)

def get_save_path(superclases):
    if superclases == 'CPA-METS':
        return '4_classes'
    elif superclases == 'Captured24':
        return '8_classes'
    else:
        return '15_classes'
    
def pretreatment(y_data):
    # Get indices of elements to remove
    indices_to_remove = [i for i, lbl in enumerate(y_data) if lbl in ACTIVITIES_TO_BE_REMOVED]

    return indices_to_remove

def superclases_captured24(y_data):
    return np.array([MAPPING_CAPTURED24.get(label, "UNKNOWN") for label in y_data])

def superclases_cpa_mets(y_data):
    return np.array([MAPPING_CPA_METS.get(label, "UNKNOWN") for label in y_data])

def participant_loocv_iterator(X_data, y_data, m_data, val_size=0.2):
    # Transporm string labels to numbers
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)
    class_names = le.classes_

    """
    Generator that performs Leave-One-Group-Out for testing, 
    and uses GroupShuffleSplit to create a validation set from the remainder.
    """
    logo = LeaveOneGroupOut()

    # Outer Loop: Leave one participant out for TEST
    for train_val_idx, test_idx in logo.split(X_data, y_data, groups=m_data):
        
        X_train_val, X_test = X_data[train_val_idx], X_data[test_idx]
        y_train_val, y_test = y_data[train_val_idx], y_data[test_idx]
        m_train_val, m_test = m_data[train_val_idx], m_data[test_idx]

        # Inner Split: Get VALIDATION from the remaining participants
        # We use GroupShuffleSplit to ensure the val participant isn't in train
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size)
        
        try:
            train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=m_train_val))
        except StopIteration:
            # Handle cases where there aren't enough groups left to split
            continue

        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        m_train, m_val = m_train_val[train_idx], m_train_val[val_idx]

        # Log the current fold status
        print(f"--- Fold for Participant(s) {np.unique(m_test)} ---")
        print(f"Train Groups: {len(np.unique(m_train))} | Val Groups: {len(np.unique(m_val))}")

        # Split features into M(91) and PI(91)
        X_train_M, X_train_PI, X_train_C = X_train[:, :91], X_train[:, 91:182], X_train[:, 182:273]
        X_val_M, X_val_PI, X_val_C       = X_val[:, :91],   X_val[:, 91:182],   X_val[:, 182:273]
        X_test_M, X_test_PI, X_test_C    = X_test[:, :91],  X_test[:, 91:182],  X_test[:, 182:273]

        # Yield the data so the loop can process one fold at a time
        yield (
            X_train_PI, X_val_PI, X_test_PI,
            X_train_M,  X_val_M,  X_test_M,
            X_train_C,  X_val_C,  X_test_C,
            y_train, y_val, y_test,
            m_train, m_val, m_test,
            class_names
        )

def build_autoencoder(input_dim, latent_dim, dropout=0.0, l2_reg=0.0):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,), name="input")    
    encoded = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="enc_dense_128")(input_layer)
    encoded = layers.Dropout(dropout)(encoded)
    encoded = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="enc_dense_64")(encoded)

    latent = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(l2_reg), name="latent")(encoded)

    # Decoder
    decoded = layers.Dense(64, activation="relu", name="dec_dense_64")(latent)
    decoded = layers.Dense(128, activation="relu", name="dec_dense_128")(decoded)
    output_layer = layers.Dense(input_dim, activation="linear", name="output")(decoded)

    autoencoder = models.Model(input_layer, output_layer)
    encoder = models.Model(input_layer, latent)

    return autoencoder, encoder

def objective(trial, X_train, X_validation):
    latent_dim = trial.suggest_int("latent_dim", 4, 64)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True) # L2 should ALWAYS be log-scaled
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    
    autoencoder, encoder = build_autoencoder(X_train.shape[1], latent_dim, dropout, l2_reg)

    autoencoder.compile(
        optimizer=Adam(lr),
        loss="mse"
    )

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_validation, X_validation),
        epochs=50,
        batch_size=64,
        callbacks=[EarlyStopping(patience=5)],
        verbose=0
    )

    return min(history.history["val_loss"])

def extract_latent_stats(encoder, X, batch_size=256):
    """
    Extract deterministic latent space from a classic Autoencoder
    """
    z = encoder.predict(X, batch_size=batch_size)

    return z

def build_gate_router(expert_PI, expert_M, expert_C, X_PI, X_M, X_C, y):
    # 1. Get predicted probabilities for all three experts
    p_PI_train = expert_PI.predict_proba(X_PI)
    p_M_train = expert_M.predict_proba(X_M)
    p_C_train = expert_C.predict_proba(X_C)

    # 2. Determine per-sample correctness (1 if correct, 0 if incorrect)
    correct_PI = (p_PI_train.argmax(axis=1) == y).astype(int)
    correct_M = (p_M_train.argmax(axis=1) == y).astype(int)
    correct_C = (p_C_train.argmax(axis=1) == y).astype(int)

    # 3. Extract the confidence (maximum probability) for each exper
    conf_PI = p_PI_train.max(axis=1)
    conf_M = p_M_train.max(axis=1)
    conf_C = p_C_train.max(axis=1)

# 4. Stack columns for matrix operations: Shape (N, 3)
    # Column indices: 0 = C, 1 = M, 2 = PI (matching your target label assignment)
    correct_matrix = np.column_stack((correct_C, correct_M, correct_PI))
    conf_matrix    = np.column_stack((conf_C, conf_M, conf_PI))

    # 5. Initialize the routing targets array
    gate_y = np.zeros_like(y)
    num_samples = len(y)

    # 6. Route samples based on correctness and confidence
    for i in range(num_samples):
        correct_row = correct_matrix[i]
        conf_row = conf_matrix[i]
        
        max_correctness = correct_row.max()
        
        # Find which experts achieved the maximum correctness for this sample
        # (e.g., if any are correct, find the correct ones; if all are wrong, find all of them)
        candidates = np.where(correct_row == max_correctness)[0]
        
        if len(candidates) == 1:
            # One clear winner based on correctness
            gate_y[i] = candidates[0]
        else:
            # Tie-breaker: Among the best-performing experts, pick the one with the highest confidence
            best_candidate_idx = candidates[np.argmax(conf_row[candidates])]
            gate_y[i] = best_candidate_idx

    return gate_y

def mixture_of_experts_soft_predict_proba(clf_PI, clf_M, clf_C, gate, Z_test_PI, Z_test_M, Z_test_C):
    # 1. Expert probabilities prediction
    p_test_PI = clf_PI.predict_proba(Z_test_PI)
    p_test_M  = clf_M.predict_proba(Z_test_M)
    p_test_C  = clf_C.predict_proba(Z_test_C)

    # 2. Gate probabilities prediction (N, 3)
    # Ensure features are stacked in the same order used to train the gate
    gate_features = np.hstack([p_test_C, p_test_M, p_test_PI])
    w = gate.predict_proba(gate_features)

    # 3. Extract expert weights (N, 1) based on the mapping: 0=C, 1=M, 2=PI
    w_C  = w[:, 0].reshape(-1, 1)
    w_M  = w[:, 1].reshape(-1, 1)
    w_PI = w[:, 2].reshape(-1, 1)

    # 4. Weighted mixture
    return (w_C * p_test_C) + (w_M * p_test_M) + (w_PI * p_test_PI)

def mixture_of_experts_hard_predict_proba(clf_PI, clf_M, clf_C, gate, Z_test_PI, Z_test_M, Z_test_C):
    # 1. Expert probabilities prediction
    p_test_PI = clf_PI.predict_proba(Z_test_PI)
    p_test_M  = clf_M.predict_proba(Z_test_M)
    p_test_C  = clf_C.predict_proba(Z_test_C)

    # 2. Gate probabilities prediction (N, 3)
    gate_features = np.hstack([p_test_C, p_test_M, p_test_PI])
    w = gate.predict_proba(gate_features)

    # 3. Choose the top-1 expert per sample (returns 0, 1, or 2 for each row)
    chosen_expert_indices = w.argmax(axis=1)

    # 4. Stack all expert predictions into a 3D array: Shape (N, 3, Num_Classes)
    # Axis 1 matches our indices: 0 = C, 1 = M, 2 = PI
    stacked_experts = np.stack([p_test_C, p_test_M, p_test_PI], axis=1)

    # 5. Advanced Indexing: Pick the chosen expert's slice for each sample row
    # row_indices creates a sequence from 0 to N-1
    row_indices = np.arange(len(chosen_expert_indices))
    p_final = stacked_experts[row_indices, chosen_expert_indices]

    return p_final

def plot_reconstruction_error(model, X, file_name, title="Reconstruction Error"):
    X_hat = model.predict(X)
    mse = np.mean((X - X_hat) ** 2, axis=1)

    plt.figure(figsize=(8, 4))
    plt.hist(mse, bins=50, alpha=0.7, label="Reconstruction Error")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"images/{file_name}", dpi=300, bbox_inches="tight")

def plot_reconstruction(model, X, file_name, n_samples=5, title="Autoencoder Reconstruction"):
    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sel = X[idx]
    X_hat = model.predict(X_sel)

    plt.figure(figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        # Original
        plt.subplot(n_samples, 2, 2*i + 1)
        plt.plot(X_sel[i], label="Original", linewidth=2)
        plt.title("Original")
        plt.legend()
        plt.grid(True)

        # Reconstruction
        plt.subplot(n_samples, 2, 2*i + 2)
        plt.plot(X_hat[i], label="Reconstruction", linestyle="--")
        plt.title("Reconstruction")
        plt.legend()
        plt.grid(True)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"images/{file_name}", dpi=300, bbox_inches="tight")

def compare_reconstruction_errors(model_PI, model_M, model_C, X_PI, X_M, X_C, file_name):
    err_PI = np.mean((X_PI - model_PI.predict(X_PI)) ** 2, axis=1)
    err_M  = np.mean((X_M  - model_M.predict(X_M)) ** 2, axis=1)
    err_C  = np.mean((X_C  - model_C.predict(X_C)) ** 2, axis=1)

    plt.figure(figsize=(8, 4))
    plt.hist(err_PI, bins=50, alpha=0.7, label="PI Autoencoder")
    plt.hist(err_M,  bins=50, alpha=0.7, label="M Autoencoder")
    plt.hist(err_C,  bins=50, alpha=0.7, label="C Autoencoder")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"images/{file_name}", dpi=300, bbox_inches="tight")

def compute_tsne(Z, n_components=2, perplexity=30, random_state=42):
    Z_scaled = StandardScaler().fit_transform(Z)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state
    )

    Z_tsne = tsne.fit_transform(Z_scaled)

    return Z_tsne

def plot_tsne_autoencoder(Z_tsne, y_train, class_names, title, file_name):
    fig, ax = plt.subplots(figsize=(7, 6))

    unique_classes = np.unique(y_train)
    cmap = plt.get_cmap("tab10")

    for i, cls in enumerate(unique_classes):
        idx = y_train == cls
        ax.scatter(
            Z_tsne[idx, 0],
            Z_tsne[idx, 1],
            s=15,
            alpha=0.7,
            color=cmap(i),
            label=class_names[cls]
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)

    # Preserve geometry
    ax.set_aspect("equal", adjustable="box")

    # Legend outside without squeezing plot
    ax.legend(
        title="Activity",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True
    )

    ax.grid(True)

    # Save safely (legend included)
    plt.savefig(f"images/{file_name}", dpi=300, bbox_inches="tight")

start_app = time.perf_counter()

args = parse_args(sys.argv[1:])

print("🟢 load stack PI+M")
stack_data_all = np.load(args.stack_all)

X_data_all = stack_data_all[WINDOW_DATA]
y_data_all = stack_data_all[WINDOW_LABELS]
m_data_all = stack_data_all[WINDOW_METADATA]

print("🟢 Remove some activities")
ACTIVITIES = [x for x in ACTIVITIES if x not in ACTIVITIES_TO_BE_REMOVED]

indices_to_remove = pretreatment(y_data_all)

X_data = np.delete(X_data_all, indices_to_remove, axis=0)
y_data = np.delete(y_data_all, indices_to_remove, axis=0)
m_data = np.delete(m_data_all, indices_to_remove, axis=0)

print("🟢 Regroup labels vector")
if (args.superclases == "Captured24"):
    ACTIVITIES = SUPERCLASES_CAPTURED24
    (y_data) = superclases_captured24(y_data)    
elif (args.superclases == "CPA-METS"):
    ACTIVITIES = SUPERCLASES_CPA_METS
    (y_data) = superclases_cpa_mets(y_data)

print("🟢 Normalize PI and M Datasets") 
sc = StandardScaler()

X_data = sc.fit_transform(X_data)

print("Calculate PI+M LOOCV(Leave-One-Out)")
data_iterator = participant_loocv_iterator(X_data, y_data, m_data)

for loop, (X_train_PI, X_validation_PI, X_test_PI, 
           X_train_M, X_validation_M, X_test_M,
           X_train_C, X_validation_C, X_test_C,
           y_train,
           y_validation,
           y_test, m_train,
           m_validation,
           m_test,
           class_names) in enumerate(data_iterator, start=1):        
    start_loop = time.perf_counter()

    metric = {}

    print("🔵 Loop: " + str(loop))

    print("🟢 Optimize Autoencoder hyperparameters PI")
    study_PI = optuna.create_study(direction="minimize")
    study_PI.optimize(lambda trial: objective(trial, X_train_PI, X_validation_PI), n_trials=args.optimize_trials)

    best_params_PI = study_PI.best_trial.params
    print(best_params_PI)

    print("🟢 Optimize Autoencoder hyperparameters M")
    study_M = optuna.create_study(direction="minimize")
    study_M.optimize(lambda trial: objective(trial, X_train_M, X_validation_M), n_trials=args.optimize_trials)

    best_params_M = study_M.best_trial.params
    print(best_params_M)

    print("🟢 Optimize Autoencoder hyperparameters C")
    study_C = optuna.create_study(direction="minimize")
    study_C.optimize(lambda trial: objective(trial, X_train_C, X_validation_C), n_trials=args.optimize_trials)

    best_params_C = study_C.best_trial.params
    print(best_params_C)

    print("🟢 Build Autoencoder with best parameters PI")
    clear_session()

    autoencoder_PI, encoder_PI = build_autoencoder(input_dim=X_train_PI.shape[1], latent_dim=best_params_PI["latent_dim"], dropout=best_params_PI["dropout"])

    print("🟢 Freeze Encoder M")
    encoder_PI.trainable = False

    print("🟢 Compile Autoencoder PI")
    autoencoder_PI.compile(optimizer=Adam(learning_rate=best_params_PI["lr"]), loss="mse")

    autoencoder_PI.summary()

    print("🟢 Build Autoencoder with best parameters M")
    clear_session()

    autoencoder_M, encoder_M = build_autoencoder(input_dim=X_train_M.shape[1], latent_dim=best_params_M["latent_dim"], dropout=best_params_M["dropout"])
    
    print("🟢 Freeze Encoder M")
    encoder_M.trainable = False

    print("🟢 Compile Autoencoder M")
    autoencoder_M.compile(optimizer=Adam(learning_rate=best_params_M["lr"]), loss="mse")

    autoencoder_M.summary()

    print("🟢 Build Autoencoder with best parameters C")
    clear_session()

    autoencoder_C, encoder_C = build_autoencoder(input_dim=X_train_C.shape[1], latent_dim=best_params_C["latent_dim"], dropout=best_params_C["dropout"])
    
    print("🟢 Freeze Encoder C")
    encoder_C.trainable = False

    print("🟢 Compile Autoencoder C")
    autoencoder_C.compile(optimizer=Adam(learning_rate=best_params_C["lr"]), loss="mse")

    autoencoder_C.summary()

    print("🟢 Compute reconstruction MSE for PI")
    X_train_pred_PI = autoencoder_PI.predict(X_train_PI)
    X_test_pred_PI  = autoencoder_PI.predict(X_test_PI)

    mse_train_PI = np.mean(np.square(X_train_PI - X_train_pred_PI), axis=1)
    mse_test_PI  = np.mean(np.square(X_test_PI - X_test_pred_PI),  axis=1)

    print("Mean reconstruction MSE for train PI:", np.mean(mse_train_PI))
    print("Mean reconstruction MSE for test PI:", np.mean(mse_test_PI))

    print("🟢 Compute reconstruction MSE for M")
    X_train_pred_M = autoencoder_M.predict(X_train_M)
    X_test_pred_M  = autoencoder_M.predict(X_test_M)

    mse_train_M = np.mean(np.square(X_train_M - X_train_pred_M), axis=1)
    mse_test_M  = np.mean(np.square(X_test_M - X_test_pred_M),  axis=1)

    print("Mean reconstruction MSE for train M:", np.mean(mse_train_M))
    print("Mean reconstruction MSE for test M:", np.mean(mse_test_M))

    print("🟢 Compute reconstruction MSE for C")
    X_train_pred_C = autoencoder_C.predict(X_train_C)
    X_test_pred_C  = autoencoder_C.predict(X_test_C)

    mse_train_C = np.mean(np.square(X_train_C - X_train_pred_C), axis=1)
    mse_test_C  = np.mean(np.square(X_test_C - X_test_pred_C),  axis=1)

    print("Mean reconstruction MSE for train C:", np.mean(mse_train_C))
    print("Mean reconstruction MSE for test C:", np.mean(mse_test_C))

    print("🟢 Build classifier PI")
    # I will use the latent space to train the classifier
    Z_train_PI = encoder_PI.predict(X_train_PI)

    clf_PI = LogisticRegression(max_iter=1000)
    clf_PI.fit(Z_train_PI, y_train)

    print("🟢 Test classifier PI")
    Z_test_PI = encoder_PI.predict(X_test_PI)

    y_test_pred_PI = clf_PI.predict(Z_test_PI)
    acc_score_test_PI = accuracy_score(y_test, y_test_pred_PI)
    f1_score_test_PI = f1_score(y_test, y_test_pred_PI, average='macro')

    print("🟢 Build classifier M")
    Z_train_M = encoder_M.predict(X_train_M)

    clf_M = LogisticRegression(max_iter=1000)
    clf_M.fit(Z_train_M, y_train)

    print("🟢 Test classifier M")
    Z_test_M = encoder_M.predict(X_test_M)

    y_test_pred_M = clf_M.predict(Z_test_M)
    acc_score_test_M = accuracy_score(y_test, y_test_pred_M)
    f1_score_test_M = f1_score(y_test, y_test_pred_M, average='macro')

    print("🟢 Build classifier C")
    Z_train_C = encoder_C.predict(X_train_C)

    clf_C = LogisticRegression(max_iter=1000)
    clf_C.fit(Z_train_C, y_train)

    print("🟢 Test classifier C")
    Z_test_C = encoder_C.predict(X_test_C)

    y_test_pred_C = clf_C.predict(Z_test_C)
    acc_score_test_C = accuracy_score(y_test, y_test_pred_C)
    f1_score_test_C = f1_score(y_test, y_test_pred_C, average='macro')

    print("🟢 Build gate validation datasets")
    X_train_pred_PI = clf_PI.predict_proba(Z_train_PI)
    X_train_pred_M = clf_M.predict_proba(Z_train_M)
    X_train_pred_C = clf_C.predict_proba(Z_train_C)

    X_gate_val = np.hstack([X_train_pred_PI, X_train_pred_M, X_train_pred_C])
    y_gate_val = build_gate_router(clf_PI, clf_M, clf_C, Z_train_PI, Z_train_M, Z_train_C, y_train)

    print("🟢 Training gate")
    gate = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000
        ))
    ])

    gate.fit(X_gate_val, y_gate_val)

    print("🟢 Test gate")
    X_test_pred_PI = clf_PI.predict_proba(Z_test_PI)
    X_test_pred_M = clf_M.predict_proba(Z_test_M)
    X_test_pred_C = clf_C.predict_proba(Z_test_C)

    X_gate_test = np.hstack([X_test_pred_PI, X_test_pred_M, X_test_pred_C])
    y_gate_test = build_gate_router(clf_PI, clf_M, clf_C, Z_test_PI, Z_test_M, Z_test_C, y_test)

    gate_pred = gate.predict(X_gate_test)

    gate_acc = (gate_pred == y_gate_test).mean()

    print("Gate accuracy:", gate_acc)

    print("🟢 Soft Test MoE")
    p_final_soft = mixture_of_experts_soft_predict_proba(clf_PI, clf_M, clf_C, gate, Z_test_PI, Z_test_M, Z_test_C)
    y_pred_soft = p_final_soft.argmax(axis=1)

    moe_acc_soft = accuracy_score(y_test, y_pred_soft)
    moe_f1_weight_soft = f1_score(y_test, y_pred_soft, average="weighted")

    print(f"Soft MoE Accuracy: {moe_acc_soft:.4f}, Soft MoE F1-score: {moe_f1_weight_soft:.4f}")

    print("🟢 Hard Test MoE")
    p_final_hard = mixture_of_experts_hard_predict_proba(clf_PI, clf_M, clf_C, gate, Z_test_PI, Z_test_M, Z_test_C)
    y_pred_hard = p_final_hard.argmax(axis=1)

    moe_acc_hard = accuracy_score(y_test, y_pred_hard)
    moe_f1_weight_hard = f1_score(y_test, y_pred_hard, average="weighted")

    print(f"Hard MoE Accuracy: {moe_acc_hard:.4f}, Hard MoE F1-score: {moe_f1_weight_hard:.4f}")

    print("🟢 Add metrics")
    metric["loop"] = loop

    metric["expert_model_test_accuracy_PI"] = acc_score_test_PI
    metric["expert_model_test_f1_score_PI"] = f1_score_test_PI
    metric["expert_model_test_accuracy_M"] = acc_score_test_M
    metric["expert_model_test_f1_score_M"] = f1_score_test_M
    metric["expert_model_test_accuracy_C"] = acc_score_test_C
    metric["expert_model_test_f1_score_C"] = f1_score_test_C    
    metric["gate_model_test_accuracy"] = gate_acc
    metric["moe_model_test_soft_accuracy"] = moe_acc_soft
    metric["moe_model_test_soft_f1_score"] = moe_f1_weight_soft
    metric["moe_model_test_hard_accuracy"] = moe_acc_soft
    metric["moe_model_test_hard_f1_score"] = moe_f1_weight_soft

    metrics.append(metric)

    if args.generate_plots == True:
        print("🟢 Reconstruction plots")
        plot_reconstruction_error(autoencoder_PI, X_test_PI, "mse_AE_PI.png", "Reconstruction Error PI")
        plot_reconstruction_error(autoencoder_M, X_test_M, "mse_AE_M.png", "Reconstruction Error M")
        plot_reconstruction_error(autoencoder_C, X_test_C, "mse_AE_C.png", "Reconstruction Error C")

        plot_reconstruction(autoencoder_PI, X_test_PI, "reconstruction_AE_PI.png", n_samples=5, title="Samples AE Reconstruction PI")
        plot_reconstruction(autoencoder_M, X_test_M, "reconstruction_AE_M.png", n_samples=5, title="Samples AE Reconstruction M")
        plot_reconstruction(autoencoder_C, X_test_C, "reconstruction_AE_C.png", n_samples=5, title="Samples AE Reconstruction C")

        compare_reconstruction_errors(autoencoder_PI, autoencoder_M, autoencoder_C, X_test_PI, X_test_M, X_test_C, "compare_reconstruction_AE_PI_M_C.png")

        print("🟢 Latent Space t-SNE plots")
        z_PI = extract_latent_stats(encoder_PI, X_train_PI)
        z_M  = extract_latent_stats(encoder_M, X_train_M)
        z_C  = extract_latent_stats(encoder_C, X_train_C)

        Z_PI_tsne = compute_tsne(z_PI)
        Z_M_tsne  = compute_tsne(z_M)
        Z_C_tsne  = compute_tsne(z_C)

        plot_tsne_autoencoder(Z_PI_tsne, y_train, class_names, title="AE Latent Space (PI)", file_name="tsne_AE_latent_PI.png")
        plot_tsne_autoencoder(Z_M_tsne, y_train, class_names, title="AE Latent Space (M)", file_name="tsne_AE_latent_M.png")
        plot_tsne_autoencoder(Z_C_tsne, y_train, class_names, title="AE Latent Space (C)", file_name="tsne_AE_latent_C.png")

    elapsed_loop = time.perf_counter() - start_loop
    print(f"Loop time: {elapsed_loop:.2f} seconds")        

print("🟢 Calculate metrics mean and standard deviations")
df_metrics = pd.DataFrame(metrics)

# Compute mean and std (numeric columns only)
mean_row = df_metrics.mean(numeric_only=True)
std_row = df_metrics.std(numeric_only=True)

# Add a label for the index column
mean_row["loop"] = "mean"
std_row["loop"] = "std"

# Append to dataframe
df_metrics = pd.concat(
    [df_metrics, pd.DataFrame([mean_row, std_row])],
    ignore_index=True
)

print("🟢 Save metrics")
df_metrics.to_csv(str(Path.cwd()) + "/paper/4_moe_ae/" + get_save_path(args.superclases) + "/" + CSV_FILE_NAME, index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")