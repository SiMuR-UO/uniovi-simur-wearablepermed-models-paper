import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
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

def pretreatment(y_data):
    # Get indices of elements to remove
    indices_to_remove = [i for i, lbl in enumerate(y_data) if lbl in ACTIVITIES_TO_BE_REMOVED]

    return indices_to_remove

def superclases_captured24(y_data):
    return np.array([MAPPING_CAPTURED24.get(label, "UNKNOWN") for label in y_data])

def superclases_cpa_mets(y_data):
    return np.array([MAPPING_CPA_METS.get(label, "UNKNOWN") for label in y_data])

def participant_group_split(X_data, y_data, m_data, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size)

    train_idx, test_idx = next(gss.split(X_data, y_data, m_data))

    X_train, X_test = X_data[train_idx], X_data[test_idx]
    y_train, y_test = y_data[train_idx], y_data[test_idx]
    m_train, m_test = m_data[train_idx], m_data[test_idx]

    print(f"Unique participants in train: {np.unique(m_train)}")
    print(f"Unique participants in test:  {np.unique(m_data[test_idx])}")

    # split concatate dataset between PI and M
    X_train_M = X_train[:, :91]
    X_train_PI = X_train[:, 91:]

    X_test_M = X_test[:, :91]
    X_test_PI = X_test[:, 91:]
    
    return X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test

def participant_cross_training(model, X_data, y_data, m_data, n_folds=3):
    gkf = GroupKFold(n_splits=n_folds)

    X_proba_all = []
    y_proba_all = []
    m_proba_all = []

    model_acc_scores = []
    model_f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_data, y_data, m_data), start=1):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]
        m_train, m_test = m_data[train_idx], m_data[test_idx]

        print(f"\n=== Fold {fold} ===")
        print("Train groups:", np.unique(m_train))
        print("Test groups: ", np.unique(m_test))

        # train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba_all.append(y_test)
        m_proba_all.append(m_test)

        # Predict probabilistic distribution
        X_proba = model.predict_proba(X_test)
        X_proba_all.append(X_proba)
        
        # test
        model_acc_scores.append(accuracy_score(y_test, y_pred))
        model_f1_scores.append(f1_score(y_test, y_pred, average='macro'))

    # Concatenate across folds
    X_proba_all = np.concatenate(X_proba_all, axis=0)
    y_proba_all = np.concatenate(y_proba_all, axis=0)
    m_proba_all = np.concatenate(m_proba_all, axis=0)

    model_acc_score_mean = float(np.mean(model_acc_scores))
    model_f1_score_mean = float(np.mean(model_f1_scores))

    # train the model
    model.fit(X_data, y_data)

    return model, X_proba_all, y_proba_all, m_proba_all, model_acc_score_mean, model_f1_score_mean

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

for loop in range(args.loops):
    start_loop = time.perf_counter()

    metric = {}

    print("🔵 Loop: " + str(loop))

    print("🟢 Split dataset PI+M")
    (X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test) = participant_group_split(X_data, y_data, m_data)

    print("\n")
    print(f"PI X Train size: {X_train_PI.shape}, PI y Train size: {y_train.shape}, PI X Test size: {X_test_PI.shape}, PI y Test size: {y_test.shape}")
    print(f"M X Train size: {X_train_M.shape}, M y Train size: {y_train.shape}, M X Test size: {X_test_M.shape}, M y Test size: {y_test.shape}")
    print("\n")

    print("🟢 Create model PI")
    model_PI = RandomForestClassifier(        
        n_estimators=N_ESTIMATORS,                     
        max_depth=MAX_DEPTH,
        max_features= MAX_FEATURES,                 
        min_samples_split=MIN_SAMPLES_SPLIT,        
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        verbose=1   
    )
    
    print("🟢 Cross Training for PI")
    (base_model_PI,
     p_X_tr_PI,
     p_y_tr,
     p_m_tr,
     model_test_accuracy_PI,
     model_test_f1_score_PI) = participant_cross_training(model_PI, X_train_PI, y_train, m_train)

    print("🟢 Create model M")
    model_M = RandomForestClassifier(        
        n_estimators=N_ESTIMATORS,                     
        max_depth=MAX_DEPTH,
        max_features= MAX_FEATURES,                 
        min_samples_split=MIN_SAMPLES_SPLIT,        
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        verbose=1   
    ) 

    print("🟢 Cross Training for M")
    (base_model_M,
     p_X_tr_M,
     p_y_tr,
     p_m_tr,
     model_test_accuracy_M,
     model_test_f1_score_M) = participant_cross_training(model_M, X_train_M, y_train, m_train)

    print("🟢 Base predictions on training for PI and M")
    
    stack_X_tr = np.hstack([p_X_tr_PI, p_X_tr_M])

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
    grid.fit(stack_X_tr, p_y_tr, groups=p_m_tr)
    model_meta = grid.best_estimator_

    print("Best params:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)

    print("🟢 Test meta model")
    pa_te_PI = base_model_PI.predict_proba(X_test_PI)
    pb_te_M = base_model_M.predict_proba(X_test_M)
    
    stack_X_te = np.hstack([pa_te_PI, pb_te_M])

    meta_model_test_accuracy = accuracy_score(y_test, model_meta.predict(stack_X_te))
    meta_model_test_f1_score = f1_score(y_test, model_meta.predict(stack_X_te), average='macro')

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

df_metrics.to_csv(str(Path.cwd()) + "/results/stacking_rf_metrics.csv", index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")