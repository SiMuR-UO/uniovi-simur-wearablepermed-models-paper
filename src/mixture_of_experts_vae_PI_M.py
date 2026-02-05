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

def build_classifier(num_classes, latent_dim, hidden_dim, name):
    inputs = Input(shape=(latent_dim,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

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
    z_mean, z_log_var, _ = encoder.predict(X, batch_size=batch_size)
    
    return z_mean, z_log_var

def reconstruction_error(vae, X, batch_size=256):
    X_hat = vae.predict(X, batch_size=batch_size)

    return np.mean((X - X_hat) ** 2, axis=1)

def gate_decision(encoder_PI, encoder_M, gate, X_PI, X_M):
    z_mu_PI, z_lv_PI, _ = encoder_PI.predict(X_PI)
    z_mu_M,  z_lv_M,  _ = encoder_M.predict(X_M)

    X_gate = np.concatenate(
        [z_mu_PI, z_lv_PI, z_mu_M, z_lv_M],
        axis=1
    )

    return gate.predict(X_gate)

def plot_reconstruction_error(vae, X, file_name, title="Reconstruction Error"):
    X_hat = vae.predict(X)
    mse = np.mean((X - X_hat) ** 2, axis=1)

    plt.figure(figsize=(8, 4))    
    plt.hist(mse, bins=50, alpha=0.7)
    plt.xlabel("Reconstruction MSE")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)

    plt.savefig("images/" + file_name, dpi=300, bbox_inches="tight")

def plot_vae_reconstruction(vae, X, file_name, n_samples=5, title="VAE Reconstruction"):
    idx = np.random.choice(len(X), n_samples, replace=False)
    X_sel = X[idx]

    X_hat = vae.predict(X_sel)

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

def compare_reconstruction_errors(vae_PI, vae_M, X_PI, X_M, file_name):
    err_PI = np.mean((X_PI - vae_PI.predict(X_PI))**2, axis=1)
    err_M  = np.mean((X_M  - vae_M.predict(X_M))**2, axis=1)

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

def mixture_of_experts_soft_predict_proba(p_PI, p_M, gate_proba):
    w_PI = gate_proba[:, 1][:, None]
    w_M = gate_proba[:, 0][:, None]

    p_moe = w_PI * p_PI + w_M * p_M
    y_pred = p_moe.argmax(axis=1)

    return y_pred, p_moe

def mixture_of_experts_hard_predict_proba(p_PI, p_M, gate_proba):
    choose_PI = gate_proba[:, 1] >= gate_proba[:, 0]

    p_moe = np.zeros_like(p_PI)
    p_moe[choose_PI] = p_PI[choose_PI]
    p_moe[~choose_PI] = p_M[~choose_PI]

    y_pred = p_moe.argmax(axis=1)

    return y_pred, p_moe

def moe_predict(X_test_PI, X_test_M, encoder_PI, classifier_PI, encoder_M, classifier_M, gate):
    # Encode test data
    z_mean_PI, z_logvar_PI, z_sample_PI = encoder_PI(X_test_PI, training=False)  # shape (N, latent_dim)
    z_mean_M, z_logvar_M, z_sample_M  = encoder_M(X_test_M, training=False)

    # Expert class probabilities (convert logits if needed)
    p_PI = classifier_PI(z_sample_PI, training=False).numpy()
    p_M = classifier_M(z_sample_M, training=False).numpy()

    # Gate input = concatenation of latent vectors
    #X_gate_test = np.concatenate([z_sample_PI.numpy(), z_sample_M.numpy()], axis=1)
    X_gate_test = np.concatenate([
        z_mean_PI.numpy(), z_logvar_PI.numpy(),
        z_mean_M.numpy(),  z_logvar_M.numpy(),
    ], axis=1)

    # Gate probabilities
    #gate_proba = gate.predict(X_gate_test)  # shape (N, 2)
    gate_proba = gate.predict_proba(X_gate_test)

    # Soft MoE
    y_soft, p_soft = mixture_of_experts_soft_predict_proba(p_PI, p_M, gate_proba)

    # Hard MoE
    y_hard, p_hard = mixture_of_experts_hard_predict_proba(p_PI, p_M, gate_proba)

    return y_soft, y_hard, p_soft, p_hard

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

    print("🟢 Concatenate datasets")
    z_mu_PI, z_lv_PI  = extract_latent_stats(encoder_PI, X_train_PI)
    z_mu_M, z_lv_M  = extract_latent_stats(encoder_M, X_train_M)

    X_gate = np.concatenate([z_mu_PI, z_lv_PI, z_mu_M, z_lv_M], axis=1)

    print("🟢 Generate gate labels") 
    err_PI = reconstruction_error(vae_PI, X_train_PI)
    err_M  = reconstruction_error(vae_M,  X_train_M)

    # Gate label: 1 → PI expert, 0 → M expert
    y_gate = (err_PI < err_M).astype(int)

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

    print("🟢 Gate training")
    gate.fit(X_gate, y_gate)

    print("🟢 Gate evaluation")
    y_gate_pred = gate.predict(X_gate)
    print("Gate accuracy:", accuracy_score(y_gate, y_gate_pred))

    print("🟢 Gate validation accuracy")
    z_mu_PI_val, z_lv_PI_val = extract_latent_stats(encoder_PI, X_validation_PI)
    z_mu_M_val,  z_lv_M_val  = extract_latent_stats(encoder_M,  X_validation_M)

    X_gate_val = np.concatenate([z_mu_PI_val, z_lv_PI_val, z_mu_M_val, z_lv_M_val], axis=1)

    err_PI_val = reconstruction_error(vae_PI, X_validation_PI)
    err_M_val  = reconstruction_error(vae_M,  X_validation_M)

    y_gate_val = (err_PI_val < err_M_val).astype(int)

    gate_acc = accuracy_score(y_gate_val, gate.predict(X_gate_val)

    gate_model_test_accuracies.append(gate_acc) #TODO

    print("Gate validation accuracy:", gate_acc)

    print("🟢 Build classifier PI")
    z_mean_PI, z_logvar_PI, z_sample_PI = encoder_PI(X_train_PI, training=False)

    classifier_PI = build_classifier(len(ACTIVITIES), best_params_PI["latent_dim"], best_params_PI["hidden_dim"], name="classifier_PI")
    classifier_PI.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier_PI.fit(z_sample_PI, y_train, epochs=20, batch_size=64)

    print("🟢 Validate classifier PI")
    y_test_pred_PI = classifier_PI.predict(X_test_PI)
    acc_score_test_PI = accuracy_score(y_test, y_test_pred_PI)
    f1_score_test_PI = f1_score(y_test, y_test_pred_PI, average='macro')

    expert_model_test_accuracies_PI.append(acc_score_test_PI)
    expert_model_test_f1_scores_PI.append(f1_score_test_PI)

    print("🟢 Build classifier M")
    z_mean_M, z_logvar_M, z_sample_M = encoder_M(X_train_M, training=False)

    classifier_M = build_classifier(len(ACTIVITIES), best_params_M["latent_dim"], best_params_M["hidden_dim"], name="classifier_M")
    classifier_M.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier_M.fit(z_sample_M, y_train, epochs=20, batch_size=64)

    print("🟢 Validate classifier M")
    y_test_pred_M = classifier_M.predict(X_test_M)
    acc_score_test_M = accuracy_score(y_test, y_test_pred_M)
    f1_score_test_M = f1_score(y_test, y_test_pred_M, average='macro')

    expert_model_test_accuracies_M.append(acc_score_test_M)
    expert_model_test_f1_scores_M.append(f1_score_test_M)

    print("🟢 MoE validation")
    y_pred_soft, y_pred_hard, p_soft, p_hard = moe_predict(X_test_PI, X_test_M, encoder_PI, classifier_PI, encoder_M, classifier_M, gate)

    moe_acc_soft = accuracy_score(y_test, y_pred_soft)
    moe_f1_weight_soft = f1_score(y_test, y_pred_soft, average="weighted")
    print(f"Soft MoE Accuracy: {moe_acc_soft:.4f}, Soft MoE F1-score: {moe_f1_weight_soft:.4f}")

    moe_acc_hard = accuracy_score(y_test, y_pred_hard)
    moe_f1_weight_hard = f1_score(y_test, y_pred_hard, average="weighted")
    print(f"Hard MoE Accuracy: {moe_acc_hard:.4f}, Hard MoE F1-score: {moe_f1_weight_hard:.4f}")

    moe_model_test_soft_accuracies.append(moe_acc_soft)
    moe_model_test_soft_f1_scores.append(moe_f1_weight_soft)
    moe_model_test_hard_accuracies.append(moe_acc_soft)
    moe_model_test_hard_f1_scores.append(moe_f1_weight_soft)

    print("🟢 Reconstruction plots")
    plot_reconstruction_error(vae_PI, X_test_PI, "mse_VAE_PI.png", "Reconstruction Error PI")
    plot_reconstruction_error(vae_M, X_test_M, "mse_VAE_M.png", "Reconstruction Error M")

    plot_vae_reconstruction(vae_PI, X_test_PI, "reconstruction_VAE_PI.png", n_samples=5, title="Samples VAE Reconstruction PI")
    plot_vae_reconstruction(vae_M, X_test_M, "reconstruction_VAE_M.png", n_samples=5, title="Samples VAE Reconstruction M")

    compare_reconstruction_errors(vae_PI, vae_M, X_test_PI, X_test_M, "compare_reconstruction_VAE_PI_M.png")

    if args.plot_tsne == True:
        print("🟢 Latent Space t-SNE plots")
        z_mu_PI, z_lv_PI = extract_latent_stats(encoder_PI, X_train_PI)
        z_mu_M, z_lv_M  = extract_latent_stats(encoder_M, X_train_M)

        Z_PI_tsne = compute_tsne(z_mu_PI)
        Z_M_tsne  = compute_tsne(z_mu_M)

        plot_tsne_autoencoder(Z_PI_tsne, y_train, class_names, title="VAE Latent Space (PI)", file_name="tsne_VAE_latent_PI.png")
        plot_tsne_autoencoder(Z_M_tsne, y_train, class_names, title="VAE Latent Space (M)", file_name="tsne_VAE_latent_M.png")

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

df_metrics.to_csv(str(Path.cwd()) + "/results/moe_vae_metrics.csv", index=False)         