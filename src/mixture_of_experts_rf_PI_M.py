import sys
import argparse
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
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

WINDOW_DATA = "arr_0"
WINDOW_LABELS = "arr_1"
WINDOW_METADATA = "arr_2"

N_ESTIMATORS=493     # More trees → more stability and accuracy (to a point), but slower.
MAX_DEPTH=6          # Lower → less overfitting (shallow trees). -> Resolve the overfitting.
MAX_FEATURES=0.2
MIN_SAMPLES_SPLIT=41 # Higher values = simpler model, less overfitting.
MIN_SAMPLES_LEAF=24  # Larger → smoother predictions, less overfitting.
N_JOBS=-1

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
        "-k-folds",
        "--k-folds",
        dest="k_folds",        
        type=int,
        default=3,       
        help=f"k-Folds for train."
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

    # Unique participants
    unique_groups = np.unique(m_data)

    n_total = len(unique_groups)
    n_val = int(n_total * val_size)    
    n_test = int(n_total * test_size)

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
        m_train, m_val, m_test
    )

def base_kfold_cross_validation(X_train, y_train, m_train, k):
    gkf = GroupKFold(n_splits=k)

    X_base_train = []
    y_base_train = []

    fold_acc_training_scores = [] 
    fold_f1_training_scores = [] 
    fold_acc_val_scores = []    
    fold_f1_val_scores = []    

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=m_train), start=1):
        # Split into training and validation folds (group-aware)
        X_training_fold, X_validation_fold = X_train[train_idx], X_train[val_idx]
        y_training_fold, y_validation_fold = y_train[train_idx], y_train[val_idx]

        # Create k-fold model
        fold_model = RandomForestClassifier(
            n_jobs=N_JOBS,        
            n_estimators=N_ESTIMATORS,                     
            max_depth=MAX_DEPTH,
            max_features= MAX_FEATURES,                 
            min_samples_split=MIN_SAMPLES_SPLIT,        
            min_samples_leaf=MIN_SAMPLES_LEAF   
        )

        # Train k-fold model with k-fold datasets
        fold_model.fit(X_training_fold, y_training_fold)

        for idx, importance in enumerate(fold_model.feature_importances_):
            print(f"Feature {idx}", importance)            

        # Calculate training k-fold metrics
        y_training_fold_pred = fold_model.predict(X_training_fold)
        acc_score_training = accuracy_score(y_training_fold, y_training_fold_pred)
        f1_score_training = f1_score(y_training_fold, y_training_fold_pred, average='macro') 

        fold_acc_training_scores.append(acc_score_training)
        fold_f1_training_scores.append(f1_score_training)

        print(f"🟡 Fold {fold}/{k}: Training Accuracy Score: {acc_score_training:.4f}, Training F1-score: {f1_score_training:.4f}")

        # Calculate validation k-fold metrics
        y_validation_fold_pred = fold_model.predict(X_validation_fold)
        acc_score_val = accuracy_score(y_validation_fold, y_validation_fold_pred)
        f1_score_val = f1_score(y_validation_fold, y_validation_fold_pred, average='macro') 

        fold_acc_val_scores.append(acc_score_val)
        fold_f1_val_scores.append(f1_score_val)

        print(f"🟡 Fold {fold}/{k}: Validation Accuracy Score: {acc_score_val:.4f}, Validation F1-score: {f1_score_val:.4f}")
        
    # create base model mean and standard deviation metrics
    metrics = {
        "base_model_accuracy_train_mean": float(np.mean(fold_acc_training_scores)),
        "base_model_f1_train_mean": float(np.mean(fold_f1_training_scores)),
        "base_model_accuracy_validate_mean": float(np.mean(fold_acc_val_scores)),
        "base_model_f1_validate_mean": float(np.mean(fold_f1_val_scores)),
    }
    
    # Create base model
    expert_model = RandomForestClassifier(
        n_jobs=N_JOBS,        
        n_estimators=N_ESTIMATORS,                     
        max_depth=MAX_DEPTH, 
        max_features= MAX_FEATURES,                
        min_samples_split=MIN_SAMPLES_SPLIT,        
        min_samples_leaf=MIN_SAMPLES_LEAF              
    )

    # # Other way to obtain grouped cross validation
    # scores = cross_validate(
    #     expert_model,
    #     X_train,
    #     y_train,
    #     cv=gkf,
    #     groups=m_train,
    #     scoring={
    #         "accuracy": "accuracy",
    #         "f1_macro": "f1_macro"
    #     }
    # )

    # print("Accuracy per fold:", scores["test_accuracy"])
    # print("F1-macro per fold:", scores["test_f1_macro"])

    # print("Mean accuracy:", scores["test_accuracy"].mean())
    # print("Mean F1-macro:", scores["test_f1_macro"].mean())

    # # Create base model
    # metrics = {
    #     "base_model_accuracy_validate_mean": float(scores["test_accuracy"].mean()),
    #     "base_model_f1_validate_mean": float(scores["test_f1_macro"].mean()),
    # }

    expert_model.fit(X_train, y_train)

    return expert_model, metrics     

def build_gate(model_PI, model_M, X_PI, X_M, y):
    pa_val = model_PI.predict_proba(X_PI)
    pb_val = model_M.predict_proba(X_M)

    # Per-sample correctness
    correct_a = (pa_val.argmax(axis=1) == y).astype(int)
    correct_b = (pb_val.argmax(axis=1) == y).astype(int)

    conf_a = pa_val.max(axis=1)
    conf_b = pb_val.max(axis=1)

    gate_y = np.zeros_like(y)

    mask_a = (correct_a == 1) & (correct_b == 0)
    gate_y[mask_a] = 1

    mask_b = (correct_b == 1) & (correct_a == 0)
    gate_y[mask_b] = 0

    mask_tie = (correct_a == correct_b)
    gate_y[mask_tie] = (conf_a[mask_tie] > conf_b[mask_tie]).astype(int)

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

print("🟢 Regroup in superclasses from PI and M Datasets")
ACTIVITIES = SUPERCLASES_CAPTURED24
(y_data) = superclases_captured24(y_data)

print("🟢 Normalize PI and M Datasets") 
sc = StandardScaler()

X_data = sc.fit_transform(X_data)

print("🟢 Split Dataset (Training/Validation/Test)")
(X_train_PI, X_validation_PI, X_test_PI,
 X_train_M, X_validation_M, X_test_M,
 y_train, y_validation, y_test,
 m_train, m_validation, m_test) = participant_group_split(X_data, y_data, m_data)

print("🟢 k-Fold validation and train expert model PI")
expert_PI, metrics_PI = base_kfold_cross_validation(X_train_PI, y_train, m_train, args.k_folds)
print("\n")

print("🟢 k-Fold validation and train expert model M")
expert_M, metrics_M =  base_kfold_cross_validation(X_train_M, y_train, m_train, args.k_folds)
print("\n")

print("🟢 Build gate validation datasets")
X_gate_val = np.hstack([X_validation_PI, X_validation_M])
y_gate_val = build_gate(expert_PI, expert_M, X_validation_PI, X_validation_M, y_validation)

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
y_gate_test = build_gate(expert_PI, expert_M, X_test_PI, X_test_M, y_test)

gate_pred = gate.predict(X_gate_test)

gate_acc = (gate_pred == y_gate_test).mean()
print("Gate accuracy:", gate_acc)

print("🟢 Soft Validate MoE")
p_final_soft = mixture_of_experts_soft_predict_proba(X_test_PI, X_test_M)

y_pred_soft = p_final_soft.argmax(axis=1)

accuracy_soft = (y_pred_soft == y_test).mean()
print("Soft MoE accuracy:", accuracy_soft)

print("🟢 Hard Validate MoE")
p_final_hard = mixture_of_experts_hard_predict_proba(X_test_PI, X_test_M)

y_pred_hard = p_final_hard.argmax(axis=1)

accuracy_hard = (y_pred_hard == y_test).mean()
print("Hard MoE accuracy:", accuracy_hard)
