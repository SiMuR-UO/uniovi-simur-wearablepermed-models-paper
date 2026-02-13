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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import tensorflow as tf
from keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session, shape

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

metrics = []

class VAE(Model):
    def __init__(self, encoder, decoder, beta):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs)

        return self.decoder(z)

    def compute_loss(self, x):
        x = tf.cast(x, tf.float32)

        z_mean, z_log_var, z = self.encoder(x)
        x_hat = self.decoder(z)

        recon = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - x_hat), axis=1)
        )

        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )

        return recon + self.beta * kl, recon, kl

    def train_step(self, x):
        with tf.GradientTape() as tape:
            total, recon, kl = self.compute_loss(x)

        grads = tape.gradient(total, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": total, "recon": recon, "kl": kl}

    def test_step(self, x):
        total, recon, kl = self.compute_loss(x)

        return {"loss": total, "recon": recon, "kl": kl}

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Mixture of Experts with variational autoencoders experts")

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

def participant_group_split(X_data, y_data, m_data, val_size=0.2, test_size=0.2):
    # Transporm string labels to numbers
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)
    class_names = le.classes_

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size)

    train_validation_idx, test_idx = next(gss_test.split(X_data, y_data, groups=m_data))

    X_train_validation, X_test = X_data[train_validation_idx], X_data[test_idx]
    y_train_validation, y_test = y_data[train_validation_idx], y_data[test_idx]
    m_train_validation, m_test = m_data[train_validation_idx], m_data[test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=(val_size / (1 - test_size)))

    train_idx, val_idx = next(gss_val.split(X_train_validation, y_train_validation, groups=m_train_validation))

    X_train, X_val = X_train_validation[train_idx], X_train_validation[val_idx]
    y_train, y_val = y_train_validation[train_idx], y_train_validation[val_idx]
    m_train, m_val = m_train_validation[train_idx], m_train_validation[val_idx]

    print(f"Unique participants in train: {np.unique(m_train)}")
    print(f"Unique participants in validation:  {np.unique(m_val)}")
    print(f"Unique participants in test:  {np.unique(m_test)}")

    # Split training/validation/test for M(91), PI(91)
    X_train_M, X_train_PI = X_train[:, :91], X_train[:, 91:]
    X_val_M, X_val_PI = X_val[:, :91], X_val[:, 91:]
    X_test_M, X_test_PI = X_test[:, :91], X_test[:, 91:]
    
    return (
        X_train_PI, X_val_PI, X_test_PI,
        X_train_M,  X_val_M,  X_test_M,
        y_train, y_val, y_test,
        m_train, m_val, m_test,
        class_names
    )

def build_encoder(input_dim, latent_dim, hidden_dim, name):
    inputs = Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    def sampling(args):
        mu, logvar = args
        eps = tf.random.normal(shape=shape(mu))

        return mu + tf.exp(0.5 * logvar) * eps

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name=f"{name}_z")([z_mean, z_log_var])

    return Model(inputs, [z_mean, z_log_var, z], name=name)

def build_decoder(output_dim, latent_dim, hidden_dim, name):
    inputs = Input(shape=(latent_dim,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    outputs = layers.Dense(output_dim)(x)

    return Model(inputs, outputs, name=name)

def objective(trial, X_train, X_val, input_dim):
    latent_dim = trial.suggest_int("latent_dim", 8, 32)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    beta = trial.suggest_float("beta", 0.1, 2.0)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    encoder = build_encoder(input_dim, latent_dim, hidden_dim, "encoder")
    decoder = build_decoder(input_dim, latent_dim, hidden_dim, "decoder")

    vae = VAE(encoder, decoder, beta)
    vae.compile(optimizer=Adam(lr))

    vae.fit(
        X_train,
        epochs=30,
        batch_size=batch_size,
        validation_data=(X_val, None),
        verbose=0
    )

    # 🔹 Manual validation loss
    val_loss = 0.0
    n_batches = 0

    for x in tf.data.Dataset.from_tensor_slices(X_val).batch(batch_size):
        total, _, _ = vae.compute_loss(x)
        val_loss += float(total.numpy())
        n_batches += 1

    val_loss /= n_batches

    return float(val_loss)    

def extract_latent_stats(encoder, X, batch_size=256):
    """
    Extract gaussian latent space from a classic Autoencoder
    """
    z_mean, z_log_var, z = encoder.predict(X, batch_size=batch_size)
    
    return z_mean, z_log_var, z

def reconstruction_error(model, X, batch_size=256):
    X_hat = model.predict(X, batch_size=batch_size)

    return np.mean((X - X_hat) ** 2, axis=1)

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

def mixture_of_experts_soft_predict_proba(clf_PI, clf_M, gate, Z_test_PI, Z_test_M):
    # Expert probabilities prediction (N,8)
    p_test_PI = clf_PI.predict_proba(Z_test_PI)
    p_test_M = clf_M.predict_proba(Z_test_M)

    # Gate probabilities prediction (N,2)
    w = gate.predict_proba(np.hstack([p_test_PI, p_test_M]))

    # Extract expert weights (N, 1)
    w_PI = w[:, 1].reshape(-1, 1)
    w_M = w[:, 0].reshape(-1, 1)

    # Weighted mixture
    return w_PI * p_test_PI + w_M * p_test_M

def mixture_of_experts_hard_predict_proba(clf_PI, clf_M, gate, Z_test_PI, Z_test_M):
    # Expert probabilities prediction (N, 8)
    p_test_PI = clf_PI.predict_proba(Z_test_PI)
    p_test_M = clf_M.predict_proba(Z_test_M)

    # Gate probabilities prediction (N, 2)
    w = gate.predict_proba(np.hstack([p_test_PI, p_test_M]))

    # Choose expert per sample (top-1)
    choose_PI = (w[:, 1] >= w[:, 0])  # True → expert PI, False → expert M

    # Allocate output
    p_final = np.zeros_like(p_test_PI)

    # Fill per-sample
    p_final[choose_PI] = p_test_PI[choose_PI]
    p_final[~choose_PI] = p_test_M[~choose_PI]

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

def plot_vae_reconstruction(model, X, file_name, n_samples=5, title="VAE Reconstruction"):
    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sel = X[idx]

    X_hat = model.predict(X_sel)

    plt.figure(figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        # Original
        plt.subplot(n_samples, 2, 2*i + 1)
        plt.plot(X_sel[i], label="Original", linewidth=2)
        plt.title("Original")
        plt.grid(True)

        # Reconstruction
        plt.subplot(n_samples, 2, 2*i + 2)
        plt.plot(X_hat[i], label="Reconstruction", linestyle="--")
        plt.title("Reconstruction")
        plt.grid(True)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("images/"+ file_name, dpi=300, bbox_inches="tight")

def compare_reconstruction_errors(model_PI, model_M, X_PI, X_M, file_name):
    err_PI = np.mean((X_PI - model_PI.predict(X_PI))**2, axis=1)
    err_M  = np.mean((X_M  - model_M.predict(X_M))**2, axis=1)

    plt.figure(figsize=(8, 4))
    plt.hist(err_PI, bins=50, alpha=0.7, label="PI")
    plt.hist(err_M,  bins=50, alpha=0.7, label="M")
    plt.xlabel("Reconstruction MSE")
    plt.legend()
    plt.title("Reconstruction Error Comparison")
    plt.grid(True)
    plt.savefig("images/"+ file_name, dpi=300, bbox_inches="tight")

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
    plt.figure(figsize=(7, 6))

    unique_classes = np.unique(y_train)
    cmap = plt.get_cmap("tab10")

    for i, cls in enumerate(unique_classes):
        idx = y_train == cls
        plt.scatter(
            Z_tsne[idx, 0],
            Z_tsne[idx, 1],
            s=15,
            alpha=0.7,
            color=cmap(i),
            label=class_names[cls]
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)

    # Preserve geometry
    plt.gca().set_aspect("equal", adjustable="box")

    # Legend OUTSIDE without shrinking axes
    plt.legend(
        title="Activity",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True
    )

    plt.grid(True)

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

print("🟢 Split Dataset (Training/Validation/Test) for PI and M")
(X_train_PI, X_validation_PI, X_test_PI,
 X_train_M,  X_validation_M,  X_test_M,
 y_train, y_validation, y_test,
 m_train, m_validation, m_test,
 class_names) = participant_group_split(X_data, y_data, m_data)

for loop in range(args.loops):
    start_loop = time.perf_counter()

    metric = {}

    print("🔵 Loop: " + str(loop))

    print("🟢 Optimize Autoencoder hyperparameters PI")
    study_PI = optuna.create_study(direction="minimize")
    study_PI.optimize(lambda trial: objective(trial, X_train_PI, X_validation_PI, input_dim=91), n_trials=args.optimize_trials)

    best_params_PI = study_PI.best_trial.params
    print(best_params_PI)

    print("🟢 Optimize Autoencoder hyperparameters M")
    study_M = optuna.create_study(direction="minimize")
    study_M.optimize(lambda trial: objective(trial, X_train_M, X_validation_M, input_dim=91), n_trials=args.optimize_trials)

    best_params_M = study_M.best_trial.params
    print(best_params_M)

    print("🟢 Build VAE with best parameters PI")
    clear_session()

    encoder_PI = build_encoder(91, best_params_PI["latent_dim"], best_params_PI["hidden_dim"], "encoder_PI")
    decoder_PI= build_decoder(91, best_params_PI["latent_dim"], best_params_PI["hidden_dim"], "decoder_PI")

    vae_PI = VAE(encoder_PI, decoder_PI, best_params_PI["beta"])

    print("🟢 Compile VAE PI")
    vae_PI.compile(optimizer=Adam(best_params_PI["lr"]))

    vae_PI.summary() 

    print("🟢 Train VAE PI")
    vae_PI.fit(X_train_PI, epochs=80, batch_size=best_params_PI["batch_size"])

    encoder_PI.trainable = False

    print("🟢 Build optimus VAE M")
    clear_session()

    encoder_M = build_encoder(91, best_params_M["latent_dim"], best_params_M["hidden_dim"], "encoder_M")
    decoder_M = build_decoder(91, best_params_M["latent_dim"], best_params_M["hidden_dim"], "decoder_M")

    vae_M = VAE(encoder_M, decoder_M, best_params_M["beta"])

    print("🟢 Compile VAE M")
    vae_M.compile(optimizer=Adam(best_params_M["lr"]))

    vae_M.summary()

    print("🟢 Train VAE M")
    vae_M.fit(X_train_M, epochs=80, batch_size=best_params_M["batch_size"])

    encoder_M.trainable = False

    print("🟢 Build classifier PI")
    # I will use the latent space to train the classifier and the projection z = mu + epsilon * std
    _, _, Z_train_PI  = extract_latent_stats(encoder_PI, X_train_PI)

    clf_PI = LogisticRegression(max_iter=1000)
    clf_PI.fit(Z_train_PI, y_train)

    print("🟢 Test classifier PI")
    _, _, Z_test_PI  = extract_latent_stats(encoder_PI, X_test_PI)

    y_test_pred_PI = clf_PI.predict(Z_test_PI)
    acc_score_test_PI = accuracy_score(y_test, y_test_pred_PI)
    f1_score_test_PI = f1_score(y_test, y_test_pred_PI, average='macro')

    print("🟢 Build classifier M")
    _, _, Z_train_M  = extract_latent_stats(encoder_M, X_train_M)

    clf_M = LogisticRegression(max_iter=1000)
    clf_M.fit(Z_train_M, y_train)

    print("🟢 Test classifier M")
    _, _, Z_test_M  = extract_latent_stats(encoder_M, X_test_M)

    y_test_pred_M = clf_M.predict(Z_test_M)
    acc_score_test_M = accuracy_score(y_test, y_test_pred_M)
    f1_score_test_M = f1_score(y_test, y_test_pred_M, average='macro')

    print("🟢 Build gate validation datasets")
    X_train_pred_PI = clf_PI.predict_proba(Z_train_PI)
    X_train_pred_M = clf_M.predict_proba(Z_train_M)

    #X_gate_val = np.hstack([Z_train_PI, Z_train_M])
    X_gate_val = np.hstack([X_train_pred_PI, X_train_pred_M])
    y_gate_val = build_gate_router(clf_PI, clf_M, Z_train_PI, Z_train_M, y_train)

    print("🟢 Train Logistic Regression gate") 
    gate = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    gate.fit(X_gate_val, y_gate_val)

    print("🟢 Test gate")
    X_test_pred_PI = clf_PI.predict_proba(Z_test_PI)
    X_test_pred_M = clf_M.predict_proba(Z_test_M)

    #X_gate_test = np.hstack([Z_test_PI, Z_test_M])
    X_gate_test = np.hstack([X_test_pred_PI, X_test_pred_M])
    y_gate_test = build_gate_router(clf_PI, clf_M, Z_test_PI, Z_test_M, y_test)

    gate_pred = gate.predict(X_gate_test)

    gate_acc = (gate_pred == y_gate_test).mean()

    print("Gate accuracy:", gate_acc)

    print("🟢 Soft Test MoE")
    p_final_soft = mixture_of_experts_soft_predict_proba(clf_PI, clf_M, gate, Z_test_PI, Z_test_M)

    y_pred_soft = p_final_soft.argmax(axis=1)

    moe_acc_soft = accuracy_score(y_test, y_pred_soft)
    moe_f1_weight_soft = f1_score(y_test, y_pred_soft, average="weighted")

    print(f"Soft MoE Accuracy: {moe_acc_soft:.4f}, Soft MoE F1-score: {moe_f1_weight_soft:.4f}")

    print("🟢 Hard Test MoE")
    p_final_hard = mixture_of_experts_hard_predict_proba(clf_PI, clf_M, gate, Z_test_PI, Z_test_M)

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
    metric["gate_model_test_accuracy"] = gate_acc
    metric["moe_model_test_soft_accuracy"] = moe_acc_soft
    metric["moe_model_test_soft_f1_score"] = moe_f1_weight_soft
    metric["moe_model_test_hard_accuracy"] = moe_acc_soft
    metric["moe_model_test_hard_f1_score"] = moe_f1_weight_soft

    metrics.append(metric)

    if args.generate_plots == True:
        print("🟢 Reconstruction plots")
        plot_reconstruction_error(vae_PI, X_test_PI, "mse_VAE_PI.png", "Reconstruction Error PI")
        plot_reconstruction_error(vae_M, X_test_M, "mse_VAE_M.png", "Reconstruction Error M")

        plot_reconstruction(vae_PI, X_test_PI, "reconstruction_VAE_PI.png", n_samples=5, title="Samples VAE Reconstruction PI")
        plot_reconstruction(vae_M, X_test_M, "reconstruction_VAE_M.png", n_samples=5, title="Samples VAE Reconstruction M")

        compare_reconstruction_errors(autoencoder_PI, autoencoder_M, X_test_PI, X_test_M, "compare_reconstruction_AE_PI_M.png")  

        print("🟢 Latent Space t-SNE plots")
        z_mu_PI, z_lv_PI = extract_latent_stats(encoder_PI, X_train_PI)
        z_mu_M, z_lv_M  = extract_latent_stats(encoder_M, X_train_M)

        Z_PI_tsne = compute_tsne(z_mu_PI)
        Z_M_tsne  = compute_tsne(z_mu_M)

        plot_tsne_autoencoder(Z_PI_tsne, y_train, class_names, title="VAE Latent Space (PI)", file_name="tsne_VAE_latent_PI.png")
        plot_tsne_autoencoder(Z_M_tsne, y_train, class_names, title="VAE Latent Space (M)", file_name="tsne_VAE_latent_M.png")

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
df_metrics.to_csv(str(Path.cwd()) + "/paper/4_moe_vae/" + get_save_path(args.superclases) + "/metrics.csv", index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")