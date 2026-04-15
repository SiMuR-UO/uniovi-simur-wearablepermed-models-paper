import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

N_TRIALS = 5 # You can increase n_trials for better tuning
N_SPLITS = 3
CV = 3

metrics = []

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Stacking with random forest base models")

    parser.add_argument(
        "-stack-all",
        "--stack-all",      
        dest="stack_all",        
        required=True,
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

def participant_loocv_iterator(X_data, y_data, m_data):
    # Transporm string labels to numbers
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)

    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X_data, y_data, m_data):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        m_train, m_test = m_data[train_idx], m_data[test_idx]

        # The 'test' participant is the one currently left out
        left_out_participant = np.unique(m_test)
        
        print(f"--- Processing Split ---")
        print(f"Left out participant: {left_out_participant}")
        print(f"Training on {len(np.unique(m_train))} other participants")

        # split concatenated dataset between PI and M
        X_train_M = X_train[:, :91]
        X_train_PI = X_train[:, 91:]

        X_test_M = X_test[:, :91]
        X_test_PI = X_test[:, 91:]

        yield X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test

def objective(trial, X_train, y_train, m_train, n_splits=N_SPLITS):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    max_features = trial.suggest_float("max_features", 0.1, 1.0)    
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    
    cv = GroupKFold(n_splits=n_splits)

    # Create model with suggested hyperparameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,        
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        verbose=1
    )
    
    # Evaluate using K-Fold cross-validation
    score = cross_val_score(
        clf,
        X_train,
        y_train,
        cv=cv,
        groups=m_train,
        scoring="accuracy").mean()
    
    # Optuna tries to maximize accuracy
    return score

def generate_oof_predictions(base_model_PI, base_model_M, X_PI, X_M, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)

    n_samples = X_PI.shape[0]
    n_classes = len(np.unique(y))

    # Allocate OOF prediction matrices
    X_PI_oof = np.zeros(X_PI.shape)
    p_PI_oof = np.zeros((n_samples, n_classes))
    X_M_oof = np.zeros(X_M.shape)
    p_M_oof = np.zeros((n_samples, n_classes))
    y_oof = np.zeros((n_samples, ))

    for fold, (fold_train_idx, fold_test_idx) in enumerate(gkf.split(X_PI, y, groups)):
        print(f"Fold {fold+1}/{n_splits}")

        # Split data
        X_PI_train, X_PI_test = X_PI[fold_train_idx], X_PI[fold_test_idx]
        X_M_train, X_M_test = X_M[fold_train_idx], X_M[fold_test_idx]
        y_train, y_test = y[fold_train_idx], y[fold_test_idx]

        # Predict on validation on fold for PI
        base_model_PI.fit(X_PI_train, y_train)

        X_PI_oof[fold_test_idx] = X_PI_test
        p_PI_oof[fold_test_idx] = base_model_PI.predict_proba(X_PI_test)        

        # Predict on validation on fold for M
        base_model_M.fit(X_M_train, y_train)

        X_M_oof[fold_test_idx] = X_M_test
        p_M_oof[fold_test_idx] = base_model_M.predict_proba(X_M_test)        

        # Label predict on fold
        y_oof[fold_test_idx] = y_test

    return X_PI_oof, p_PI_oof, X_M_oof, p_M_oof, y_oof

def stack_prediction(base_model_PI, base_mode_M, X_test_PI, X_test_M):
    p_test_PI = base_model_PI.predict_proba(X_test_PI)
    p_test_M = base_mode_M.predict_proba(X_test_M)
    
    meta_X = np.hstack((X_test_PI, X_test_M, p_test_PI, p_test_M))

    return meta_X

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

participant_ids = np.sort(np.unique(m_data))

print("Total participants:", len(participant_ids))

print("? Calculate PI+M LOOCV(Leave-One-Out)")
data_iterator = participant_loocv_iterator(X_data, y_data, m_data)

#for loop in range(args.loops):
for loop, (X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test) in enumerate(data_iterator, start=1):
    start_loop = time.perf_counter()

    metric = {}

    print("🔵 Loop: " + str(loop))

    print("🟢 Split dataset PI+M")
    #(X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test) = participant_group_split(X_data, y_data, m_data)

    print("\n")
    print(f"PI X Train size: {X_train_PI.shape}, PI y Train size: {y_train.shape}, PI X Test size: {X_test_PI.shape}, PI y Test size: {y_test.shape}")
    print(f"M X Train size: {X_train_M.shape}, M y Train size: {y_train.shape}, M X Test size: {X_test_M.shape}, M y Test size: {y_test.shape}")
    print("\n")

    print("🟢 Get best hyperparameters base model PI")
    study_PI = optuna.create_study(direction="maximize", study_name="3_stack_rf_PI")

    study_PI.optimize(lambda trial: objective(trial, X_train_PI, y_train, m_train), n_trials=N_TRIALS)  # You can increase n_trials for better tuning
    
    trial_PI = study_PI.best_trial

    print(f"Accuracy base model PI: {trial_PI.value}")
    print("Best hyperparameters base model PI: ")
    for key, value in trial_PI.params.items():
        print(f"    {key}: {value}")

    print("🟢 Training with best hyperparmeters base model PI")
    best_params_PI = trial_PI.params
    base_model_PI = RandomForestClassifier(**best_params_PI, n_jobs=-1)

    print("🟢 Train base model PI")
    base_model_PI.fit(X_train_PI, y_train)

    print("🟢 Evaluate base model PI")
    model_test_accuracy_PI = accuracy_score(y_test, base_model_PI.predict(X_test_PI))
    model_test_f1_score_PI = f1_score(y_test, base_model_PI.predict(X_test_PI), average='macro')
    
    print(f"Base model test Accuracy PI: {model_test_accuracy_PI}")
    print(f"Base model test F1 Score PI: {model_test_f1_score_PI}")

    print("🟢 Get best hyperparameters base model M")
    study_M = optuna.create_study(direction="maximize", study_name="3_stack_rf_M")

    study_M.optimize(lambda trial: objective(trial, X_train_M, y_train, m_train), n_trials=N_TRIALS)
    
    trial_M = study_M.best_trial

    print(f"Accuracy base model M: {trial_M.value}")
    print("Best hyperparameters base model M: ")
    for key, value in trial_M.params.items():
        print(f"    {key}: {value}")

    print("🟢 Training with best hyperparmeters base model M")
    best_params_M = trial_M.params
    base_model_M = RandomForestClassifier(**best_params_M, n_jobs=-1)

    print("🟢 Train base model M")
    base_model_M.fit(X_train_M, y_train)

    print("🟢 Evaluate base model M")
    model_test_accuracy_M = accuracy_score(y_test, base_model_M.predict(X_test_M))
    model_test_f1_score_M = f1_score(y_test, base_model_M.predict(X_test_M), average='macro')

    print(f"Base model test Accuracy M: {model_test_accuracy_M}")
    print(f"Base model test F1 Score M: {model_test_f1_score_M}")

    print("🟢 Generate training meta predictions for meta model concatenating PI and M predictions (OOF predictions of experts)")
    X_PI_oof, p_X_tr_PI, X_M_oof, p_X_tr_M, y_tr = generate_oof_predictions(base_model_PI, base_model_M, X_train_PI, X_train_M, y_train, m_train, n_splits=3)

    print("🟢 Training meta dataset")
    meta_X_tr = np.hstack([X_PI_oof, X_M_oof, p_X_tr_PI, p_X_tr_M])

    print("🟢 Training meta model")
    meta_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000
        ))
    ])

    print("🟢 Train meta model (Logistic Regression optimized hyperparameters) with concatenated probability distribution from PI and M")
    meta_model.fit(meta_X_tr, y_tr)

    print("🟢 Evaluate meta model on hold out dataset")
    X_test_meta = stack_prediction(base_model_PI, base_model_M, X_test_PI, X_test_M)

    meta_model_test_accuracy = accuracy_score(y_test, meta_model.predict(X_test_meta))
    meta_model_test_f1_score = f1_score(y_test, meta_model.predict(X_test_meta), average='macro')

    print(f"Meta model test Accuracy: {meta_model_test_accuracy}")
    print(f"Meta model test F1 Score: {meta_model_test_f1_score}")

    # save meta model metrics
    metric["loop"] = loop

    metric["base_model_accuracy_PI"] = model_test_accuracy_PI
    metric["base_model_f1_score_PI"] = model_test_f1_score_PI
    metric["base_model_accuracy_M"] = model_test_accuracy_M
    metric["base_model_f1_score_M"] = model_test_f1_score_M
    metric["meta_model_accuracy"] = meta_model_test_accuracy
    metric["meta_model_f1_score"] = meta_model_test_f1_score

    # add metrics to collection
    metrics.append(metric)
    
    if args.generate_plots == True:
        # create and plot confusion matrix from base model
        print("🟢 Confusion Matrix Meta model PI+M")
        cm = confusion_matrix(y_test, meta_model.predict(X_test_meta))

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True,        # show numbers in each cell
            fmt="d",           # integer formatting
            cmap="Blues",      # color palette
            xticklabels=ACTIVITIES, 
            yticklabels=ACTIVITIES
        )

        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix Heatmap")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(str(Path.cwd()) + "/images/confusion_matrix_" + str(loop) + ".png", dpi=300, bbox_inches="tight")
    
    elapsed_loop = time.perf_counter() - start_loop
    print(f"Loop time: {elapsed_loop:.2f} seconds")

# Save metrics
print("🟢 Save metrics for PI+M")
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
df_metrics.to_csv(str(Path.cwd()) + "/paper/3_statcking_rf/" + get_save_path(args.superclases) + "/metrics_loocv.csv", index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")