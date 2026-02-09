import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
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

N_ESTIMATORS=493     # More trees → more stability and accuracy (to a point), but slower.
MAX_DEPTH=6          # Lower → less overfitting (shallow trees). -> Resolve the overfitting.
MAX_FEATURES=0.2
MIN_SAMPLES_SPLIT=41 # Higher values = simpler model, less overfitting.
MIN_SAMPLES_LEAF=24  # Larger → smoother predictions, less overfitting.

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

def participant_group_split(X_data, y_data, m_data, val_size=0.2, test_size=0.2):
    # Transporm string labels to numbers
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)
     
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

def mixture_of_experts_soft_predict_proba(X_test_PI, X_test_M):
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

def mixture_of_experts_hard_predict_proba(X_test_PI, X_test_M):
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

for loop in range(args.loops):
    start_loop = time.perf_counter()

    print("🔵 Loop: " + str(loop))

    metric = {}

    print("🟢 Split Dataset (Training/Validation/Test)")
    (X_train_PI, X_validation_PI, X_test_PI,
    X_train_M, X_validation_M, X_test_M,
    y_train, y_validation, y_test,
    m_train, m_validation, m_test) = participant_group_split(X_data, y_data, m_data)

    print("🟢 train expert model PI")
    expert_PI = RandomForestClassifier(        
        n_estimators=N_ESTIMATORS,                     
        max_depth=MAX_DEPTH, 
        max_features= MAX_FEATURES,                
        min_samples_split=MIN_SAMPLES_SPLIT,        
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        verbose=1            
    )

    expert_PI.fit(X_train_PI, y_train)

    print("🟢 validation expert model PI")
    y_validation_pred_PI = expert_PI.predict(X_validation_PI)
    acc_score_val_PI = accuracy_score(y_validation, y_validation_pred_PI)
    f1_score_val_PI = f1_score(y_validation, y_validation_pred_PI, average='macro') 

    print("🟢 train validation expert model M")
    expert_M = RandomForestClassifier(        
        n_estimators=N_ESTIMATORS,                     
        max_depth=MAX_DEPTH, 
        max_features= MAX_FEATURES,                
        min_samples_split=MIN_SAMPLES_SPLIT,        
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        verbose=1        
    )

    expert_M.fit(X_train_M, y_train)

    print("🟢 validation expert model PI")
    y_validation_pred_M = expert_M.predict(X_validation_M)
    acc_score_val_M = accuracy_score(y_validation, y_validation_pred_M)
    f1_score_val_M = f1_score(y_validation, y_validation_pred_M, average='macro') 

    print("🟢 Build gate validation datasets")
    X_gate_val = np.hstack([
        np.vstack([X_train_PI, X_validation_PI]), 
        np.vstack([X_train_M, X_validation_M])
    ])
    y_gate_val = build_gate_router(
        expert_PI, 
        expert_M, 
        np.vstack([X_train_PI, X_validation_PI]), 
        np.vstack([X_train_M, X_validation_M]),
        np.concatenate([y_train, y_validation])
    )

    #X_gate_val = np.hstack([X_validation_PI, X_validation_M])
    #y_gate_val = build_gate_router(expert_PI, expert_M, X_validation_PI, X_validation_M, y_validation)

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
    X_gate_test = np.hstack([X_test_PI, X_test_M])
    y_gate_test = build_gate_router(expert_PI, expert_M, X_test_PI, X_test_M, y_test)

    gate_pred = gate.predict(X_gate_test)

    gate_acc = accuracy_score(y_gate_test, gate_pred)
    gate_f1_weight = f1_score(y_gate_test, gate_pred, average="weighted")
    print(f"Gate Accuracy: {gate_acc:.4f}, Gate F1-score: {gate_f1_weight:.4f}")

    print("🟢 Soft Validate MoE")
    p_final_soft = mixture_of_experts_soft_predict_proba(X_test_PI, X_test_M)

    y_pred_soft = p_final_soft.argmax(axis=1)

    moe_acc_soft = accuracy_score(y_test, y_pred_soft)
    moe_f1_weight_soft = f1_score(y_test, y_pred_soft, average="weighted")
    print(f"Soft MoE Accuracy: {moe_acc_soft:.4f}, Soft MoE F1-score: {moe_f1_weight_soft:.4f}")

    print("🟢 Hard Validate MoE")
    p_final_hard = mixture_of_experts_hard_predict_proba(X_test_PI, X_test_M)

    y_pred_hard = p_final_hard.argmax(axis=1)

    moe_acc_hard = accuracy_score(y_test, y_pred_hard)
    moe_f1_weight_hard = f1_score(y_test, y_pred_hard, average="weighted")
    print(f"Hard MoE Accuracy: {moe_acc_hard:.4f}, Hard MoE F1-score: {moe_f1_weight_hard:.4f}")

    # save meta model metrics
    metric["loop"] = loop

    metric["base_model_validate_accuracy_PI"] = acc_score_val_PI
    metric["base_model_validate_f1_score_PI"] = f1_score_val_PI
    metric["base_model_validate_accuracy_M"] = acc_score_val_M
    metric["base_model_train_f1_score_M"] = f1_score_val_M    
    metric["gate_acc"] = gate_acc
    metric["gate_f1_weight"] = gate_f1_weight
    metric["moe_acc_soft"] = moe_acc_soft
    metric["moe_f1_weight_soft"] = moe_f1_weight_soft
    metric["moe_acc_hard"] = moe_acc_hard
    metric["moe_f1_weight_hard"] = moe_f1_weight_hard

    # add metrics to collection
    metrics.append(metric)

    elapsed_loop = time.perf_counter() - start_app
    print(f"Loop time: {elapsed_loop:.2f} seconds")

print("🟢 Save metrics")

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

df_metrics.to_csv(str(Path.cwd()) + "/results/moe_rf_metrics.csv", index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")