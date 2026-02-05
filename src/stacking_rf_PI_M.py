import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

N_ESTIMATORS=493     # More trees → more stability and accuracy (to a point), but slower.
MAX_DEPTH=6          # Lower → less overfitting (shallow trees). -> Resolve the overfitting.
MAX_FEATURES=0.2
MIN_SAMPLES_SPLIT=41 # Higher values = simpler model, less overfitting.
MIN_SAMPLES_LEAF=24  # Larger → smoother predictions, less overfitting.
N_JOBS=-1

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
        help=f"Use Superclases: Captured24, CPA-METS"
    )     
    parser.add_argument(
        "-k-folds",
        "--k-folds",   
        dest="k_folds",        
        type=int,
        default=3,
        help=f"training k-folds."        
    )
    parser.add_argument(
        "-step-init",
        "--step-init",
        dest="step_init",        
        type=int,
        default=6,        
        help="Participant initial step."
    )    
    parser.add_argument(
        "-step",
        "--step",
        dest="step",        
        type=int,
        default=1,        
        help="Participant step."
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

def participant_group_split(X_data, y_data, m_data, test_size=0.2):
    # split concatate dataset between train and test
    unique_groups = np.unique(m_data)

    n_test = int(len(unique_groups) * test_size)
    test_groups = unique_groups[-n_test:]
    train_groups = unique_groups[:-n_test]

    train_idx = np.where(np.isin(m_data, train_groups))[0]
    test_idx  = np.where(np.isin(m_data, test_groups))[0]    

    X_train = X_data[train_idx]
    X_test = X_data[test_idx]
    
    y_train = y_data[train_idx]
    y_test = y_data[test_idx]

    m_train = m_data[train_idx]
    m_test = m_data[test_idx]

    print(f"Participants for train: {len(np.unique(m_data[train_idx]))}")
    print(f"Participants for test:  {len(np.unique(m_data[test_idx]))}")

    # split concatate dataset between PI and M
    X_train_M = X_train[:, :91]
    X_train_PI = X_train[:, 91:]

    X_test_M = X_test[:, :91]
    X_test_PI = X_test[:, 91:]
    
    return X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test 

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
    metric = {
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

    expert_model.fit(X_train, y_train)

    return expert_model, metric      

def merge_base_metrics(metric_M, metric_PI):
    merged = {}

    # Rename and add M metrics
    for key, value in metric_M.items():
        merged[f"{key}_M"] = value

    # Rename and add PI metrics
    for key, value in metric_PI.items():
        merged[f"{key}_PI"] = value

    return merged

args = parse_args(sys.argv[1:])

print("🟢 load stack PI+M")
stack_data_all = np.load(args.stack_all)

# get datasets from stack
X_data_all = stack_data_all[WINDOW_DATA]
y_data_all = stack_data_all[WINDOW_LABELS]
m_data_all = stack_data_all[WINDOW_METADATA]

# remove some activities
print("🟢 Remove some activities")
ACTIVITIES = [x for x in ACTIVITIES if x not in ACTIVITIES_TO_BE_REMOVED]

print("🟢 Get indices to remove for PI and M Datasets")
indices_to_remove = pretreatment(y_data_all)

X_data = np.delete(X_data_all, indices_to_remove, axis=0)
y_data = np.delete(y_data_all, indices_to_remove, axis=0)
m_data = np.delete(m_data_all, indices_to_remove, axis=0)

# Superclasses from PI and M
print("🟢 Superclasses from PI and M Datasets")
if (args.superclases == "Captured24"):
    ACTIVITIES = SUPERCLASES_CAPTURED24
    (y_data) = superclases_captured24(y_data)    
elif (args.superclases == "CPA-METS"):
    ACTIVITIES = SUPERCLASES_CPA_METS
    (y_data) = superclases_cpa_mets(y_data)

participant_ids = np.sort(np.unique(m_data))

print("Total participants:", len(participant_ids))

for n_participants in range(args.step_init, len(participant_ids) + 1, args.step):
    # select first n participants
    selected_participants = participant_ids[:n_participants]

    # rows belonging to those participants
    mask = np.isin(m_data, selected_participants)

    for loop in range(args.loops):
        print("🔵 Loop: " + str(loop))

        print("🟢 get subset participant dataset")
        X_sub_data = X_data[mask]
        y_sub_data = y_data[mask]
        m_sub_data = m_data[mask]

        print("🟢 Split dataset PI+M")
        (X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test) = participant_group_split(X_sub_data, y_sub_data, m_sub_data)

        print("\n")
        print(f"PI X Train size: {X_train_PI.shape}, PI y Train size: {y_train.shape}, PI X Test size: {X_test_PI.shape}, PI y Test size: {y_test.shape}")
        print(f"M X Train size: {X_train_M.shape}, M y Train size: {y_train.shape}, M X Test size: {X_test_M.shape}, M y Test size: {y_test.shape}")
        print("\n")

        print("🟢 k-Fold train base model PI")
        base_model_PI, metric_PI = base_kfold_cross_validation(X_train_PI, y_train, m_train, args.k_folds)
        print("\n")

        print("🟢 k-Fold train base model M")
        base_model_M, metric_M =  base_kfold_cross_validation(X_train_M, y_train, m_train, args.k_folds)
        print("\n")

        print("🟢 Base predictions on training set for PI and M")
        pa_tr_PI = base_model_PI.predict_proba(X_train_PI)
        pb_tr_M = base_model_M.predict_proba(X_train_M)
        
        stack_X_tr = np.hstack([pa_tr_PI, pb_tr_M])

        pa_te_PI = base_model_PI.predict_proba(X_test_PI)
        pb_te_M = base_model_M.predict_proba(X_test_M)
        
        stack_X_te = np.hstack([pa_te_PI, pb_te_M])

        print("🟢 Optimize meta model hyperparameters")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))
        ])

        param_grid = {
            "clf__C": [0.001, 0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"]
        }

        cv = GroupKFold(n_splits=5)

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        ) 

        print("🟢 Train meta model with concatenated probability distribution from PI and M")
        grid.fit(stack_X_tr, y_train, groups=m_train)

        print("Best params:", grid.best_params_)
        print("Best CV accuracy:", grid.best_score_)
        model_meta = grid.best_estimator_

        # merge base model metrics
        metric = merge_base_metrics(metric_M, metric_PI)

        # get meta model metrics     
        meta_train_acc = accuracy_score(y_train, model_meta.predict(stack_X_tr))
        meta_train_f1 = f1_score(y_train, model_meta.predict(stack_X_tr), average='macro')
        meta_test_acc = accuracy_score(y_test, model_meta.predict(stack_X_te))
        meta_test_f1 = f1_score(y_test, model_meta.predict(stack_X_te), average='macro')

        # save meta model metrics
        metric["meta_model_accuracy_train"] = meta_train_acc
        metric["meta_model_f1_train"] = meta_train_f1
        metric["meta_model_accuracy_test"] = meta_test_acc
        metric["meta_model_f1_test"] = meta_test_f1
        metric["loop"] = loop
        metric["participants"] = n_participants

        # track meta model metrics
        print("Train Accuracy Score: " + str(meta_train_acc))
        print("Train F1-score: " + str(meta_train_f1))
        print("Test accuracy score: " + str(meta_test_acc))
        print("Test F1-score: " + str(meta_test_f1))
        print("Loop: " + str(loop))
        print("Participants: " + str(n_participants))

        # create a metrics file
        with open(str(Path.cwd()) + "/results/rf_cascading_cross_PI_M_accuracy_" + str(args.k_folds) + "_folds_" + str(n_participants) + "_participants_" + str(loop) + ".txt", "w") as f:            
            f.write(f"Train Accuracy Score: {meta_train_acc}\n")
            f.write(f"Train F1-score: {meta_train_f1}\n")
            f.write(f"Test Accuracy Score: {meta_test_acc}\n")
            f.write(f"Test F1-score: {meta_test_f1}\n")
            f.write(f"participants: {n_participants}\n")            
            f.write(f"loop: {loop}\n")            

        print("\n")

        # create and plot confusion matrix from base model
        print("🟢 Confusion Matrix Meta model PI+M")
        cm = confusion_matrix(y_test, model_meta.predict(stack_X_te))

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

        plt.savefig(str(Path.cwd()) + "/images/confusion_matrix_" + str(args.k_folds) + "_folds_" + str(n_participants) + "_participants_" + str(loop) + ".png", dpi=300, bbox_inches="tight")

        # add metrics to collection
        metrics.append(metric)

# Save metrics
print("🟢 Save metrics for PI+M")
df = pd.DataFrame(metrics)

df.to_csv(str(Path.cwd()) + "/results/metrics_output.csv", index=False)
