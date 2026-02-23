import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
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

def participant_group_split(X_data, y_data, m_data, val_size=0.2, test_size=0.2):
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
        m_train, m_val, m_test
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

print("🟢 Standarize PI and M") 
sc = StandardScaler()

X_data = sc.fit_transform(X_data)

loops = []

for loop in range(args.loops):
    start_loop = time.perf_counter()
        
    metric = {}

    print("🔵 Loop: " + str(loop))
    loops.append(loop)

    print("🟢 Split stack (Training/Validation/Test)")
    (X_train_PI, X_validation_PI, X_test_PI,
    X_train_M,  X_validation_M,  X_test_M,
    y_train, y_validation, y_test,
    m_train, m_validation, m_test) = participant_group_split(X_data, y_data, m_data)

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

    autoencoder_PI.compile(optimizer=Adam(learning_rate=best_params_PI["lr"]), loss="mse")
    autoencoder_PI.summary()

    print("🟢 Build Autoencoder with best parameters M")
    clear_session()

    autoencoder_M, encoder_M = build_autoencoder(input_dim=X_train_M.shape[1], latent_dim=best_params_M["latent_dim"], dropout=best_params_M["dropout"])

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

    print("🟢 Build classifier PI")
    # I will use the latent space to train the classifier
    Z_train_PI = encoder_PI.predict(X_train_PI)

    clf_PI = LogisticRegression(max_iter=1000)
    clf_PI.fit(Z_train_PI, y_train)

    print("🟢 Test classifier PI")
    Z_test_PI = encoder_PI.predict(X_test_PI)

    y_test_pred_PI = clf_PI.predict(Z_test_PI)
    acc_score_validation_PI = accuracy_score(y_test, y_test_pred_PI)
    f1_score_validation_PI = f1_score(y_test, y_test_pred_PI, average='macro')

    print("🟢 Build classifier M")
    Z_train_M = encoder_M.predict(X_train_M)

    clf_M = LogisticRegression(max_iter=1000)
    clf_M.fit(Z_train_M, y_train)

    print("🟢 Test classifier M")
    Z_test_M = encoder_M.predict(X_test_M)

    y_test_pred_M = clf_M.predict(Z_test_M)
    acc_score_validation_M = accuracy_score(y_test, y_test_pred_M)
    f1_score_validation_M = f1_score(y_test, y_test_pred_M, average='macro')

    print("🟢 Probability predictions on train for PI and M")
    pr_train_PI = clf_PI.predict_proba(Z_train_PI)
    pr_train_M = clf_M.predict_proba(Z_train_M)

    X_meta_train_all = np.hstack([pr_train_PI, pr_train_M])

    print("🟢 Training meta model")
    meta_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000
        ))
    ])

    meta_model.fit(X_meta_train_all, y_train)

    print("🟢 Test meta model")
    Z_test_PI = encoder_PI.predict(X_test_PI)
    Z_test_M = encoder_M.predict(X_test_M)

    y_test_pred_PI = clf_PI.predict_proba(Z_test_PI)
    y_test_pred_M = clf_M.predict_proba(Z_test_M)

    p_meta_test = meta_model.predict(np.hstack([y_test_pred_PI, y_test_pred_M]))

    # get meta model metrics
    meta_model_test_accuracy = accuracy_score(y_test, p_meta_test)
    meta_model_test_f1_score = f1_score(y_test, p_meta_test, average='macro')

    # save meta model metrics
    metric["loop"] = loop

    metric["classifier_model_accuracy_PI"] = acc_score_validation_PI
    metric["classifier_model_f1_score_PI"] = f1_score_validation_PI
    metric["classifier_model_accuracy_M"] = acc_score_validation_M
    metric["classifier_model_f1_score_M"] = f1_score_validation_M
    metric["meta_model_accuracy"] = meta_model_test_accuracy
    metric["meta_model_f1_score"] = meta_model_test_f1_score

    # add metrics to collection
    metrics.append(metric)

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
df_metrics.to_csv(str(Path.cwd()) + "/paper/3_statcking_ae/" + get_save_path(args.superclases) + "/metrics.csv", index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")