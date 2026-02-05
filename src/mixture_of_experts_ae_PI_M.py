import sys
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
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
        help=f"Use Superclases: Captured24, CPA-METS"
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
        '-plot-tsne',
        '--plot-tsne',
        dest='plot_tsne',
        action='store_true',
        default=False,
        help="Plot laten t-SNE"
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
    
def pretreatment(y_data):
    # Get indices of elements to remove
    indices_to_remove = [i for i, lbl in enumerate(y_data) if lbl in ACTIVITIES_TO_BE_REMOVED]

    return indices_to_remove

def superclases_captured24(y_data):
    return np.array([MAPPING_CAPTURED24.get(label, "UNKNOWN") for label in y_data])

def superclases_cpa_mets(y_data):
    return np.array([MAPPING_CPA_METS.get(label, "UNKNOWN") for label in y_data])

def participant_group_split(X_data, y_data, m_data, val_size=0.2, test_size=0.1):
    # Transporm string labels to numbers
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)
    class_names = le.classes_

    # Unique participants
    unique_groups = np.unique(m_data)

    n_total = len(unique_groups)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    test_groups = unique_groups[:n_test]
    val_groups = unique_groups[n_test:n_test + n_val]
    train_groups = unique_groups[n_test + n_val:]

    # Index selection
    train_idx = np.where(np.isin(m_data, train_groups))[0]
    val_idx   = np.where(np.isin(m_data, val_groups))[0]
    test_idx  = np.where(np.isin(m_data, test_groups))[0]

    # Split datasets
    X_train, X_val, X_test = X_data[train_idx], X_data[val_idx], X_data[test_idx]
    y_train, y_val, y_test = y_data[train_idx], y_data[val_idx], y_data[test_idx]
    m_train, m_val, m_test = m_data[train_idx], m_data[val_idx], m_data[test_idx]

    print(f"Participants → Train: {len(np.unique(m_train))}")
    print(f"Participants → Val:   {len(np.unique(m_val))}")
    print(f"Participants → Test:  {len(np.unique(m_test))}")

    # Split training/validation/test for M(91), PI(91)
    X_train_M, X_train_PI = X_train[:, :91], X_train[:, 91:]
    X_val_M,   X_val_PI   = X_val[:, :91],   X_val[:, 91:]
    X_test_M,  X_test_PI  = X_test[:, :91],  X_test[:, 91:]

    return (
        X_train_PI, X_val_PI, X_test_PI,
        X_train_M,  X_val_M,  X_test_M,
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

def build_gate_router(expert_PI, expert_M, X_PI, X_M, y):
    p_PI_val = expert_PI.predict_proba(X_PI)
    p_M_val = expert_M.predict_proba(X_M)

    # Per-sample correctness
    correct_PI = (p_PI_val.argmax(axis=1) == y).astype(int)
    correct_M = (p_M_val.argmax(axis=1) == y).astype(int)

    conf_PI = p_PI_val.max(axis=1)
    conf_M = p_M_val.max(axis=1)

    gate_y = np.zeros_like(y)

    mask_PI = (correct_PI == 1) & (correct_M == 0)
    gate_y[mask_PI] = 1

    mask_M = (correct_M == 1) & (correct_PI == 0)
    gate_y[mask_M] = 0

    mask_tie = (correct_PI == correct_M)
    gate_y[mask_tie] = (conf_PI[mask_tie] > conf_M[mask_tie]).astype(int)

    return gate_y

def mixture_of_experts_soft_predict_proba(expert_PI, expert_M, gate, X_test_PI, X_test_M):
    # Expert probabilities prediction (N,8)
    p_test_PI = expert_PI.predict_proba(X_test_PI)
    p_test_M = expert_M.predict_proba(X_test_M)

    # Gate probabilities prediction (N,2)
    w = gate.predict_proba(np.hstack([X_test_PI, X_test_M]))

    # Extract expert weights (N, 1)
    w_PI = w[:, 1].reshape(-1, 1)
    w_M = w[:, 0].reshape(-1, 1)

    # Weighted mixture
    return w_PI * p_test_PI + w_M * p_test_M

def mixture_of_experts_hard_predict_proba(expert_PI, expert_M, gate, X_test_PI, X_test_M):
    # Expert probabilities prediction (N, 8)
    p_test_PI = expert_PI.predict_proba(X_test_PI)
    p_test_M = expert_M.predict_proba(X_test_M)

    # Gate probabilities prediction (N, 2)
    w = gate.predict_proba(np.hstack([X_test_PI, X_test_M]))

    # Choose expert per sample (top-1)
    choose_PI = (w[:, 1] >= w[:, 0])  # True → expert PI, False → expert M

    # Allocate output
    p_final = np.zeros_like(p_test_PI)

    # Fill per-sample
    p_final[choose_PI] = p_test_PI[choose_PI]
    p_final[~choose_PI] = p_test_M[~choose_PI]

    return p_final

def plot_reconstruction_error(ae, X, file_name, title="Reconstruction Error"):
    X_hat = ae.predict(X)
    mse = np.mean((X - X_hat) ** 2, axis=1)

    plt.figure(figsize=(8, 4))
    plt.hist(mse, bins=50, alpha=0.7, label="Reconstruction Error")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"images/{file_name}", dpi=300, bbox_inches="tight")

def plot_ae_reconstruction(ae, X, file_name, n_samples=5, title="Autoencoder Reconstruction"):
    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sel = X[idx]
    X_hat = ae.predict(X_sel)

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

def compare_reconstruction_errors(ae_PI, ae_M, X_PI, X_M, file_name):
    err_PI = np.mean((X_PI - ae_PI.predict(X_PI)) ** 2, axis=1)
    err_M  = np.mean((X_M  - ae_M.predict(X_M)) ** 2, axis=1)

    plt.figure(figsize=(8, 4))
    plt.hist(err_PI, bins=50, alpha=0.7, label="PI Autoencoder")
    plt.hist(err_M,  bins=50, alpha=0.7, label="M Autoencoder")
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title("Reconstruction Error Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"images/{file_name}", dpi=300, bbox_inches="tight")

def get_latent(encoder, X, batch_size=256):
    """
    Extract deterministic latent space from a classic Autoencoder
    """
    Z = encoder.predict(X, batch_size=batch_size)
    return Z

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

print("🟢 Split stack (Training/Validation/Test)")
(X_train_PI, X_validation_PI, X_test_PI,
 X_train_M,  X_validation_M,  X_test_M,
 y_train, y_validation, y_test,
 m_train, m_validation, m_test,
 class_names) = participant_group_split(X_data, y_data, m_data)

loops = []

expert_model_test_accuracies_PI = []
expert_model_test_f1_scores_PI = []
expert_model_test_accuracies_M = []
expert_model_test_f1_scores_M = []
gate_model_test_accuracies = []
moe_model_test_soft_accuracies = []
moe_model_test_soft_f1_scores = []
moe_model_test_hard_accuracies = []
moe_model_test_hard_f1_scores = []

for loop in range(args.loops):
    print("🔵 Loop: " + str(loop))
    loops.append(loop)

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

    print("🟢 Build Autoencoder with best parameters PI")
    clear_session()

    autoencoder_PI, encoder_PI = build_autoencoder(input_dim=X_train_PI.shape[1], latent_dim=best_params_PI["latent_dim"], dropout=best_params_PI["dropout"])

    print("🟢 Compile Autoencoder PI")
    autoencoder_PI.compile(optimizer=Adam(learning_rate=best_params_PI["lr"]), loss="mse")

    autoencoder_PI.summary()

    print("🟢 Build Autoencoder with best parameters M")
    clear_session()

    autoencoder_M, encoder_M = build_autoencoder(input_dim=X_train_M.shape[1], latent_dim=best_params_M["latent_dim"], dropout=best_params_M["dropout"])

    print("🟢 Compile Autoencoder M")
    autoencoder_M.compile(optimizer=Adam(learning_rate=best_params_M["lr"]), loss="mse")

    autoencoder_M.summary()

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

    if args.plot_tsne == True:
        print("🟢 Latent Space t-SNE plots")
        z_PI = get_latent(encoder_PI, X_train_PI)
        z_M  = get_latent(encoder_M, X_train_M)

        Z_PI_tsne = compute_tsne(z_PI)
        Z_M_tsne  = compute_tsne(z_M)

        plot_tsne_autoencoder(Z_PI_tsne, y_train, class_names, title="AE Latent Space (PI)", file_name="tsne_AE_latent_PI.png")
        plot_tsne_autoencoder(Z_M_tsne, y_train, class_names, title="AE Latent Space (M)", file_name="tsne_AE_latent_M.png")

    print("🟢 Build classifier PI")
    Z_validation_PI = encoder_PI.predict(X_validation_PI)

    clf_PI = LogisticRegression(max_iter=1000)
    clf_PI.fit(Z_validation_PI, y_validation)

    print("🟢 Validate classifier PI")
    Z_test_PI = encoder_PI.predict(X_test_PI)

    y_test_pred_PI = clf_PI.predict(Z_test_PI)
    acc_score_test_PI = accuracy_score(y_test, y_test_pred_PI)
    f1_score_test_PI = f1_score(y_test, y_test_pred_PI, average='macro')

    expert_model_test_accuracies_PI.append(acc_score_test_PI)
    expert_model_test_f1_scores_PI.append(f1_score_test_PI)

    print("🟢 Build classifier M")
    Z_validation_M = encoder_M.predict(X_validation_M)

    clf_M = LogisticRegression(max_iter=1000)
    clf_M.fit(Z_validation_M, y_validation)

    print("🟢 Validate classifier M")
    Z_test_M = encoder_M.predict(X_test_M)

    y_test_pred_M = clf_M.predict(Z_test_M)
    acc_score_test_M = accuracy_score(y_test, y_test_pred_M)
    f1_score_test_M = f1_score(y_test, y_test_pred_M, average='macro')

    expert_model_test_accuracies_M.append(acc_score_test_M)
    expert_model_test_f1_scores_M.append(f1_score_test_M)

    print("🟢 Build gate validation datasets")
    X_gate_val = np.hstack([Z_validation_PI, Z_validation_M])
    y_gate_val = build_gate_router(clf_PI, clf_M, Z_validation_PI, Z_validation_M, y_validation)

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

    print("🟢 Validate gate")
    Z_test_PI = encoder_PI.predict(X_test_PI)
    Z_test_M = encoder_M.predict(X_test_M)

    X_gate_test = np.hstack([Z_test_PI, Z_test_M])
    y_gate_test = build_gate_router(clf_PI, clf_M, Z_test_PI, Z_test_M, y_test)

    gate_pred = gate.predict(X_gate_test)

    gate_acc = (gate_pred == y_gate_test).mean()

    gate_model_test_accuracies.append(gate_acc)
    print("Gate accuracy:", gate_acc)

    print("🟢 Soft Validate MoE")
    p_final_soft = mixture_of_experts_soft_predict_proba(clf_PI, clf_M, gate, Z_test_PI, Z_test_M)

    y_pred_soft = p_final_soft.argmax(axis=1)

    moe_acc_soft = accuracy_score(y_test, y_pred_soft)
    moe_f1_weight_soft = f1_score(y_test, y_pred_soft, average="weighted")

    moe_model_test_soft_accuracies.append(moe_acc_soft)
    moe_model_test_soft_f1_scores.append(moe_f1_weight_soft)

    print(f"Soft MoE Accuracy: {moe_acc_soft:.4f}, Soft MoE F1-score: {moe_f1_weight_soft:.4f}")

    print("🟢 Hard Validate MoE")
    p_final_hard = mixture_of_experts_hard_predict_proba(clf_PI, clf_M, gate, Z_test_PI, Z_test_M)

    y_pred_hard = p_final_hard.argmax(axis=1)

    moe_acc_hard = accuracy_score(y_test, y_pred_hard)
    moe_f1_weight_hard = f1_score(y_test, y_pred_hard, average="weighted")

    moe_model_test_hard_accuracies.append(moe_acc_soft)
    moe_model_test_hard_f1_scores.append(moe_f1_weight_soft)

    print(f"Hard MoE Accuracy: {moe_acc_hard:.4f}, Hard MoE F1-score: {moe_f1_weight_hard:.4f}")

    print("🟢 Reconstruction plots")
    plot_reconstruction_error(autoencoder_PI, X_test_PI, "mse_AE_PI.png", "Reconstruction Error PI")
    plot_reconstruction_error(autoencoder_M, X_test_M, "mse_AE_M.png", "Reconstruction Error M")

    plot_ae_reconstruction(autoencoder_PI, X_test_PI, "reconstruction_AE_PI.png", n_samples=5, title="Samples AE Reconstruction PI")
    plot_ae_reconstruction(autoencoder_M, X_test_M, "reconstruction_AE_M.png", n_samples=5, title="Samples AE Reconstruction M")

    compare_reconstruction_errors(autoencoder_PI, autoencoder_M, X_test_PI, X_test_M, "compare_reconstruction_AE_PI_M.png")

expert_model_test_accuracies_PI = []
expert_model_test_f1_scores_PI = []
expert_model_test_accuracies_M = []
expert_model_test_f1_scores_M = []
gate_model_test_accuracies = []
moe_model_test_soft_accuracies = []
moe_model_test_soft_f1_scores = []
moe_model_test_hard_accuracies = []
moe_model_test_hard_f1_scores = []

print("🟢 Save metrics")
df_metrics = pd.DataFrame({   
    'loop': loops,
    'expert_model_test_accuracy_PI': expert_model_test_accuracies_PI,
    'expert_model_test_f1_score_PI': expert_model_test_f1_scores_PI,
    'expert_model_test_accuracy_M': expert_model_test_accuracies_M,
    'expert_model_test_f1_score_M': expert_model_test_f1_scores_M,
    'gate_model_test_accuracy': gate_model_test_accuracies,
    'moe_model_test_soft_accuracy': moe_model_test_soft_accuracies,
    'moe_model_test_soft_f1_score': moe_model_test_soft_f1_scores,
    'moe_model_test_hard_accuracy': moe_model_test_hard_accuracies,
    'moe_model_test_hard_f1_score': moe_model_test_hard_f1_scores,
})

df_metrics.to_csv(str(Path.cwd()) + "/results/moe_ae_metrics.csv", index=False)     