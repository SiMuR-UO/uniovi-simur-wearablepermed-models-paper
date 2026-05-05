import sys
import time
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, LeaveOneGroupOut, cross_val_score
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

N_TRIALS = 5 # You can increase n_trials for better tuning
N_SPLITS = 3

metrics = []

CSV_SUFFIX_FILE_NAME = "_all.csv"

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Individual Random Forest Model")
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
        help=f"Use Superclases: WearablePerMed, Captured24, CPA-METS"
    )
    parser.add_argument(
        "-segment-body",
        "--segment-body",
        required=True,
        dest="segment_body",    
        help=f"Segment Body: PI, M, C"
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

def participant_group_split(X_data, y_data, m_data, segment_body, test_size=0.2):
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
    
    if segment_body == 'PI':
        return X_train_PI, X_test_PI, y_train, y_test, m_train, m_test
    elif segment_body == 'M':
        return X_train_M, X_test_M, y_train, y_test, m_train, m_test
    else:
        raise Exception("Sorry, Segment body " + segment_body + " is not contemplated")

def participant_loocv_iterator(X_data, y_data, m_data, segment_body):
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

        # split concatate dataset between PI and M
        # X_train_M = X_train[:, :91]
        # X_train_PI = X_train[:, 91:]

        # X_test_M = X_test[:, :91]
        # X_test_PI = X_test[:, 91:]

        # if segment_body == 'PI':
        #     yield X_train_PI, X_test_PI, y_train, y_test, m_train, m_test
        # elif segment_body == 'M':
        #     yield X_train_M, X_test_M, y_train, y_test, m_train, m_test
        # else:
        #     raise Exception("Sorry, Segment body " + segment_body + " is not contemplated")
            
        yield X_train, X_test, y_train, y_test, m_train, m_test
           
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

print("Calculate PI+M LOOCV(Leave-One-Out)")
data_iterator = participant_loocv_iterator(X_data, y_data, m_data, args.segment_body)

#for loop in range(args.loops):
for loop, (X_train, X_test, y_train, y_test, m_train, m_test) in enumerate(data_iterator, start=1):    
    start_loop = time.perf_counter()
    print("🔵 Loop: " + str(loop))

    metric = {}

    #print("🟢 Split dataset PI+M")
    #(X_train, X_test, y_train, y_test, m_train, m_test) = participant_group_split(X_data, y_data, m_data, args.segment_body)
    print(f"X Train size: {X_train.shape}, y Train size: {y_train.shape}, X Test size: {X_test.shape}, y Test size: {y_test.shape}")
   
    print("🟢 Get best hyperparameters")
    study = optuna.create_study(direction="maximize", study_name="1_individual_rf")

    study.optimize(lambda trial: objective(trial, X_train, y_train, m_train), n_trials=N_TRIALS)
    
    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print("Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("🟢 training model with best hyperparmeters individual model")
    best_params = trial.params
    model = RandomForestClassifier(**best_params, n_jobs=-1)

    model.fit(X_train, y_train)

    print("🟢 Validate model")
    model_accuracy_test = accuracy_score(y_test, model.predict(X_test))
    model_f1_score_test = f1_score(y_test, model.predict(X_test), average='macro')

    # save meta model metrics
    metric["loop"] = loop

    metric["model_accuracy_test"] = model_accuracy_test
    metric["model_f1_score_test"] = model_f1_score_test

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
df_metrics.to_csv(str(Path.cwd()) + "/paper/1_individual/" + get_save_path(args.superclases) + "/metrics_loocv_" + args.segment_body.lower() + CSV_SUFFIX_FILE_NAME, index=False)

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")