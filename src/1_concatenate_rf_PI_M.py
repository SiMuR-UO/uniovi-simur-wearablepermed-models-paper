import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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
    parser = argparse.ArgumentParser(description="Cascading Random Forest Model")
    parser.add_argument(
        "-stack-all",
        "--stack-all",
        dest="stack_all",       
        required=True,
        help=f"Stack all for train."
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
        type=int,
        default=5,
        dest="k_folds",       
        required=True,
        help=f"k-Folds for train."
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
        help="set loglevel to INFO.",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG.",
        action="store_const",
        const=logging.DEBUG,
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
    
    return X_train, X_test, y_train, y_test, m_train, m_test

def model_kfold_cross_validate(X_train, y_train, m_train, k):
    start_cross = time.perf_counter()

    # classifier model
    model = RandomForestClassifier(        
        n_estimators=N_ESTIMATORS,                     
        max_depth=MAX_DEPTH,
        max_features= MAX_FEATURES,                 
        min_samples_split=MIN_SAMPLES_SPLIT,        
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        verbose=1   
    )

    # Cross-validation strategy
    gkf = GroupKFold(n_splits=k, shuffle=True)
        
    # Execute cross-validation       
    cv_scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=gkf,
        groups=m_train,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro"
        },
        n_jobs=1
    )

    metrics = {
        "model_accuracy_test": float(cv_scores["test_accuracy"].mean()),
        "model_f1_score_test": float(cv_scores["test_f1_macro"].mean()),
    }

    # Train classifier model
    model.fit(X_train, y_train)

    # cross validation time tracking
    elapsed_cross = time.perf_counter() - start_cross
    print(f"Cross-validation time: {elapsed_cross:.2f} seconds")
    
    return model, metrics        

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

loops = []

model_train_accuracies = []
model_train_f1_scores = []
model_test_accuracies = []
model_test_f1_scores = []

start_app = time.perf_counter()

for loop in range(args.loops):
    print("🔵 Loop: " + str(loop))
    start_loop = time.perf_counter()

    print("🟢 Split dataset PI+M")
    (X_train, X_test, y_train, y_test, m_train, m_test) = participant_group_split(X_data, y_data, m_data)

    print("\n")
    print(f"X Train size: {X_train.shape}, y Train size: {y_train.shape}, X Test size: {X_test.shape}, y Test size: {y_test.shape}")
    print("\n")    
   
    print("🟢 k-Fold cross validation model")
    model, metrics = model_kfold_cross_validate(X_train, y_train, m_train, args.k_folds)

    print("🟢 Validate model")
    model_test_accuracy = accuracy_score(y_test, model.predict(X_test))
    model_test_f1_score = f1_score(y_test, model.predict(X_test), average='macro')

    print("🟢 Append model metrics")
    loops.append(loop)
    model_test_accuracies.append(metrics["model_accuracy_test"])
    model_test_f1_scores.append(metrics["model_f1_score_test"])

    elapsed_loop = time.perf_counter() - start_loop
    print(f"Loop time: {elapsed_loop:.2f} seconds")

print("🟢 Save metrics")
df_metrics = pd.DataFrame({   
    'loop': loops,
    'model_test_accuracy': model_test_accuracies,
    'model_test_f1_score': model_test_f1_scores,
})

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

df_metrics.to_csv(str(Path.cwd()) + "/results/concatenate_rf_metrics.csv", index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")