import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, LeaveOneGroupOut, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ACTIVITIES = ['FASE REPOSO CON K5', 'TAPIZ RODANTE',
              'INCREMENTAL CICLOERGOMETRO', 'YOGA', 'SENTADO VIENDO LA TV',
              'SENTADO LEYENDO', 'SENTADO USANDO PC', 'DE PIE USANDO PC',
              'DE PIE DOBLANDO TOALLAS', 'DE PIE MOVIENDO LIBROS',
              'DE PIE BARRIENDO', 'CAMINAR USUAL SPEED',
              'CAMINAR CON MÓVIL O LIBRO', 'CAMINAR CON LA COMPRA',
              'CAMINAR ZIGZAG', 'TROTAR', 'SUBIR Y BAJAR ESCALERAS']

ACTIVITIES_TO_BE_REMOVED=['TAPIZ RODANTE', 'YOGA']

SUPERCLASES_CAPTURED24 = ['WALKING', 'HOUSEHOLD-CHORES',
                          'STANDING', 'SLEEP', 'BICYCLING',
                          'SITTING', 'MIXED-ACTIVITY', 'SPORTS']

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

metrics = []

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Mixture of Experts with radom forests experts")

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

def build_gate_router(p_PI_val, p_M_val, y):
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

def mixture_of_experts_soft_predict_proba(X_test_PI, X_test_M):
    # Expert probabilities prediction (N,8)
    p_test_PI = expert_PI.predict_proba(X_test_PI)
    p_test_M = expert_M.predict_proba(X_test_M)

    # Gate probabilities prediction (N,2)
    w = gate.predict_proba(np.hstack([X_test_PI, X_test_M, p_test_PI, p_test_M]))

    # Extract expert weights (N, 1)
    w_PI = w[:, 1].reshape(-1, 1)
    w_M = w[:, 0].reshape(-1, 1)

    # Weighted mixture
    return w_PI * p_test_PI + w_M * p_test_M

def mixture_of_experts_hard_predict_proba(X_test_PI, X_test_M):
    # Expert probabilities prediction (N, 8)
    p_test_PI = expert_PI.predict_proba(X_test_PI)
    p_test_M = expert_M.predict_proba(X_test_M)

    # Gate probabilities prediction (N, 2)
    w = gate.predict_proba(np.hstack([X_test_PI, X_test_M, p_test_PI, p_test_M]))

    # Choose expert per sample (top-1)
    choose_PI = (w[:, 1] >= w[:, 0])  # True → expert PI, False → expert M

    # Allocate output
    p_final = np.zeros_like(p_test_PI)

    # Fill per-sample
    p_final[choose_PI] = p_test_PI[choose_PI]
    p_final[~choose_PI] = p_test_M[~choose_PI]

    return p_final

start_app = time.perf_counter()

args = parse_args(sys.argv[1:])

print("🟢 load stack PI+M")
stack_data_all = np.load(args.stack_all)

X_data_all = stack_data_all[WINDOW_DATA]
y_data_all = stack_data_all[WINDOW_LABELS]
m_data_all = stack_data_all[WINDOW_METADATA]

print("🟢 Remove some activities from stack")
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

print("🟢 Normalize PI and M Datasets") 
sc = StandardScaler()

X_data = sc.fit_transform(X_data)

print("Calculate PI+M LOOCV(Leave-One-Out)")
data_iterator = participant_loocv_iterator(X_data, y_data, m_data)

for loop, (X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test) in enumerate(data_iterator, start=1):
    start_loop = time.perf_counter()

    print("🔵 Loop: " + str(loop))

    metric = {}

    print("🟢 Get best hyperparameters model PI")
    study_PI = optuna.create_study(direction="maximize", study_name="4_mixture_of_experts_rf_PI")

    study_PI.optimize(lambda trial: objective(trial, X_train_PI, y_train, m_train), n_trials=N_TRIALS)  
    
    trial_PI = study_PI.best_trial

    print(f"Accuracy PI: {trial_PI.value}")
    print("Best hyperparameters PI: ")
    for key, value in trial_PI.params.items():
        print(f"    {key}: {value}")

    print("🟢 training model with best hyperparmeters PI")
    best_params_PI = trial_PI.params
    expert_PI = RandomForestClassifier(**best_params_PI, n_jobs=-1)

    expert_PI.fit(X_train_PI, y_train)

    print("🟢 Test expert model PI")
    acc_score_test_PI = accuracy_score(y_test, expert_PI.predict(X_test_PI))
    f1_score_test_PI = f1_score(y_test, expert_PI.predict(X_test_PI), average='macro') 

    print(f"Expert model test Accuracy PI: {acc_score_test_PI}")
    print(f"Expert model test F1 Score PI: {f1_score_test_PI}")

    print("🟢 Get best hyperparameters model M")
    study_M = optuna.create_study(direction="maximize", study_name="4_mixture_of_experts_rf_M")

    study_M.optimize(lambda trial: objective(trial, X_train_M, y_train, m_train), n_trials=N_TRIALS)  # You can increase n_trials for better tuning
    
    trial_M = study_M.best_trial

    print(f"Accuracy M: {trial_M.value}")
    print("Best hyperparameters M: ")
    for key, value in trial_M.params.items():
        print(f"    {key}: {value}")

    print("🟢 training model with best hyperparmeters M")
    best_params_M = trial_M.params
    expert_M = RandomForestClassifier(**best_params_M, n_jobs=-1)

    expert_M.fit(X_train_M, y_train)

    print("🟢 Test expert model PI")
    acc_score_test_M = accuracy_score(y_test, expert_M.predict(X_test_M))
    f1_score_test_M = f1_score(y_test, expert_M.predict(X_test_M), average='macro') 

    print(f"Expert model test Accuracy M: {acc_score_test_M}")
    print(f"Expert model test F1 Score M: {f1_score_test_M}")

    print("🟢 Training gate dataset")
    X_PI_oof, p_X_tr_PI, X_M_oof, p_X_tr_M, y_tr = generate_oof_predictions(expert_PI, expert_M, X_train_PI, X_train_M, y_train, m_train, n_splits=3)

    print("🟢 Training meta dataset")
    X_gate_train = np.hstack([X_PI_oof, X_M_oof, p_X_tr_PI, p_X_tr_M])
    y_gate_train = build_gate_router(p_X_tr_PI, p_X_tr_M, y_tr)

    print("🟢 Training gate")
    gate = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000
        ))
    ])

    gate.fit(X_gate_train, y_gate_train)

    print("🟢 Test MoE Soft")
    p_final_soft = mixture_of_experts_soft_predict_proba(X_test_PI, X_test_M)
    y_pred_soft = p_final_soft.argmax(axis=1)

    moe_acc_soft = accuracy_score(y_test, y_pred_soft)
    moe_f1_weight_soft = f1_score(y_test, y_pred_soft, average="weighted")
    print(f"Soft MoE Accuracy: {moe_acc_soft:.4f}, Soft MoE F1-score: {moe_f1_weight_soft:.4f}")

    print("🟢 Hard MoE Test")
    p_final_hard = mixture_of_experts_hard_predict_proba(X_test_PI, X_test_M)
    y_pred_hard = p_final_hard.argmax(axis=1)

    moe_acc_hard = accuracy_score(y_test, y_pred_hard)
    moe_f1_weight_hard = f1_score(y_test, y_pred_hard, average="weighted")
    print(f"Hard MoE Accuracy: {moe_acc_hard:.4f}, Hard MoE F1-score: {moe_f1_weight_hard:.4f}")

    print("🟢 Add metrics")
    metric["loop"] = loop

    metric["base_model_validate_accuracy_PI"] = acc_score_test_PI
    metric["base_model_validate_f1_score_PI"] = f1_score_test_PI
    metric["base_model_validate_accuracy_M"] = acc_score_test_M
    metric["base_model_train_f1_score_M"] = f1_score_test_M
    metric["moe_acc_soft"] = moe_acc_soft
    metric["moe_f1_weight_soft"] = moe_f1_weight_soft
    metric["moe_acc_hard"] = moe_acc_hard
    metric["moe_f1_weight_hard"] = moe_f1_weight_hard

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
df_metrics.to_csv(str(Path.cwd()) + "/paper/4_moe_rf/" + get_save_path(args.superclases) + "/metrics_loocv.csv", index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")