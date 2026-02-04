import sys
import argparse
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
        "-optimize-trials",
        "--optimize-trials",               
        dest="optimize_trials",
        type=int,
        default=1,               
        help=f"Optimize hyperparameters num trials."        
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

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)

    return -np.sum(p * np.log(p), axis=1)

def max_prob(p):
    return np.max(p, axis=1)

def margin(p):
    # diferencia entre la probabilidad mayor y la segunda mayor
    part = np.partition(-p, 1, axis=1)
    p1 = -part[:, 0]
    p2 = -part[:, 1]

    return p1 - p2

def build_gate_features(p_PI, p_M):
    H_PI = entropy(p_PI)
    H_M = entropy(p_M)

    # normalización de entropía
    K = p_PI.shape[1]
    H_PI /= np.log(K)
    H_M /= np.log(K)

    pmax_PI = max_prob(p_PI)
    pmax_M = max_prob(p_M)

    m_PI = margin(p_PI)
    m_M = margin(p_M)

    X_gate = np.column_stack([
        H_PI, pmax_PI, m_PI,
        H_M, pmax_M, m_M,
        H_PI - H_M,
        pmax_PI - pmax_M,
        m_PI - m_M
    ])

    return X_gate

def build_gate_router(p_PI, p_M, y_true, clf_PI, clf_M):
    y_hat_PI = clf_PI.classes_[np.argmax(p_PI, axis=1)]
    y_hat_M = clf_M.classes_[np.argmax(p_M, axis=1)]

    correct_PI = (y_hat_PI == y_true)
    correct_M  = (y_hat_M  == y_true)

    y_gate = np.full(len(y_true), -1)

    y_gate[(correct_PI) & (~correct_M)] = 0
    y_gate[(~correct_PI) & (correct_M)] = 1

    both = correct_PI & correct_M
    y_gate[both] = (
        max_prob(p_M[both]) > max_prob(p_PI[both])
    ).astype(int)

    mask = y_gate != -1

    return y_gate[mask], mask

def gated_prediction(p_PI, p_M, gate, X_gate, clf_PI, clf_M):
    gate_choice = gate.predict(X_gate)
    
    # convert argmax to real classes
    y_hat_PI = clf_PI.classes_[np.argmax(p_PI, axis=1)]
    y_hat_M  = clf_M.classes_[np.argmax(p_M, axis=1)]

    return np.where(gate_choice == 0, y_hat_PI, y_hat_M)

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

# get datasets from stack
X_data_all = stack_data_all[WINDOW_DATA]
y_data_all = stack_data_all[WINDOW_LABELS]
m_data_all = stack_data_all[WINDOW_METADATA]

# remove some activities
print("🟢 Remove some activities from stack")
ACTIVITIES = [x for x in ACTIVITIES if x not in ACTIVITIES_TO_BE_REMOVED]

indices_to_remove = pretreatment(y_data_all)

X_data = np.delete(X_data_all, indices_to_remove, axis=0)
y_data = np.delete(y_data_all, indices_to_remove, axis=0)
m_data = np.delete(m_data_all, indices_to_remove, axis=0)

# Superclasses from PI and M
print("🟢 Regroup in superclasses from PI and M Datasets")
ACTIVITIES = SUPERCLASES_CAPTURED24
(y_data) = superclases_captured24(y_data)

print("🟢 Normalize PI and M Datasets") 
sc = StandardScaler()

X_data = sc.fit_transform(X_data)

print("🟢 Split stack (Training/Validation/Test)")
(X_train_PI, X_validation_PI, X_test_PI,
 X_train_M,  X_validation_M,  X_test_M,
 y_train, y_validation, y_test,
 m_train, m_validation, m_test,
 class_names) = participant_group_split(X_data, y_data, m_data)

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

print("🟢 Latent Space t-SNE plots")
z_PI = get_latent(encoder_PI, X_train_PI)
z_M  = get_latent(encoder_M, X_train_M)

Z_PI_tsne = compute_tsne(z_PI)
Z_M_tsne  = compute_tsne(z_M)

plot_tsne_autoencoder(Z_PI_tsne, y_train, class_names, title="AE Latent Space (PI)", file_name="tsne_AE_latent_PI.png")
plot_tsne_autoencoder(Z_M_tsne, y_train, class_names, title="AE Latent Space (M)", file_name="tsne_AE_latent_M.png")

print("🟢 Concatenate label") 
y_data = np.concatenate((y_train, y_validation, y_test), axis=0)

print("🟢 Extract Latent Features for PI") 
X_data_PI = np.concatenate((X_train_PI, X_validation_PI, X_test_PI), axis=0)
Z_data_PI = encoder_PI.predict(X_data_PI)

print("🟢 Extract Latent Features for M")
X_data_M = np.concatenate((X_train_M, X_validation_M, X_test_M), axis=0)
Z_data_M = encoder_M.predict(X_data_M)

print("🟢 Get probability distribution from a Logistic Regression for PI")
clf_PI = LogisticRegression(max_iter=1000)
clf_PI.fit(Z_data_PI, y_data)

p_data_PI = clf_PI.predict_proba(Z_data_PI)

print("🟢 Get probability distribution from a Logistic Regression for M")
clf_M = LogisticRegression(max_iter=1000)
clf_M.fit(Z_data_M, y_data)

p_data_M = clf_M.predict_proba(Z_data_M)

print("🟢 Build Gate features: Entropy, maxP, margin p and differences from PI and M probability distribution")
X_gate = build_gate_features(p_data_PI, p_data_M)

print("🟢 Create Gate Router")
y_gate, mask = build_gate_router(p_data_PI, p_data_M, y_data, clf_PI, clf_M)

X_gate = X_gate[mask]

print("🟢 Split Gate Dataset")
X_train, X_test, y_train, y_test = train_test_split(X_gate, y_gate, test_size=0.2, random_state=42)

print("🟢 Model Gate Pipeline")
gate = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000
    ))
])

gate.fit(X_train, y_train)

print("🟢 Gate evaluation")
y_pred = gate.predict(X_test)
print("Gate accuracy:", accuracy_score(y_test, y_pred))

print("🟢 Moe evaluation")
y_gated = gated_prediction(p_data_PI[mask], p_data_M[mask], gate, X_gate, clf_PI, clf_M)
print("MoE accuracy:", accuracy_score(y_data[mask], y_gated))

print("🟢 Debug and Interpretation")
clf = gate.named_steps["clf"]
for name, coef in zip(
    ["H1", "pmax1", "m1", "H2", "pmax2", "m2",
     "ΔH", "Δpmax", "Δm"],
    clf.coef_[0]
):
    print(f"{name:8s}: {coef:+.3f}")

print("🟢 Baseline without gate")
best_single = max(
    accuracy_score(y_data, clf_PI.classes_[np.argmax(p_data_PI, axis=1)]),
    accuracy_score(y_data, clf_M.classes_[np.argmax(p_data_M, axis=1)])
)

print("Best single expert:", best_single)

print("🟢 Reconstruction plots")
plot_reconstruction_error(autoencoder_PI, X_test_PI, "mse_AE_PI.png", "Reconstruction Error PI")
plot_reconstruction_error(autoencoder_M, X_test_M, "mse_AE_M.png", "Reconstruction Error M")

plot_ae_reconstruction(autoencoder_PI, X_test_PI, "reconstruction_AE_PI.png", n_samples=5, title="Samples AE Reconstruction PI")
plot_ae_reconstruction(autoencoder_M, X_test_M, "reconstruction_AE_M.png", n_samples=5, title="Samples AE Reconstruction M")

compare_reconstruction_errors(autoencoder_PI, autoencoder_M, X_test_PI, X_test_M, "compare_reconstruction_AE_PI_M.png")