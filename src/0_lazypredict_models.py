import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from lazypredict.Supervised import LazyClassifier

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
        help=f"Segment Body: PI, M"
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

print("🟢 Split dataset PI+M")
(X_train, X_test, y_train, y_test, m_train, m_test) = participant_group_split(X_data, y_data, m_data, args.segment_body)

clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

#                                Accuracy  Balanced Accuracy ROC AUC  F1 Score  Time Taken
# Model                                                                                   
# LGBMClassifier                     0.92               0.91    None      0.92        2.82
# ExtraTreesClassifier               0.92               0.90    None      0.92        1.65
# BaggingClassifier                  0.92               0.90    None      0.92       11.45
# SVC                                0.90               0.89    None      0.90        3.37
# RandomForestClassifier             0.91               0.89    None      0.91       11.96
# LinearSVC                          0.90               0.87    None      0.90        3.97
# LogisticRegression                 0.88               0.87    None      0.88       10.03
# SGDClassifier                      0.86               0.86    None      0.87        1.81
# DecisionTreeClassifier             0.89               0.86    None      0.89        1.84
# LinearDiscriminantAnalysis         0.86               0.84    None      0.87        0.21
# Perceptron                         0.85               0.83    None      0.85        0.45
# CalibratedClassifierCV             0.88               0.83    None      0.88       15.64
# ExtraTreeClassifier                0.83               0.82    None      0.84        0.09
# KNeighborsClassifier               0.86               0.80    None      0.85        0.33
# PassiveAggressiveClassifier        0.85               0.79    None      0.85        0.56
# BernoulliNB                        0.76               0.78    None      0.77        0.11
# GaussianNB                         0.68               0.77    None      0.68        0.11
# NearestCentroid                    0.72               0.76    None      0.73        0.10
# RidgeClassifier                    0.84               0.75    None      0.83        0.09
# RidgeClassifierCV                  0.84               0.75    None      0.83        0.45
# QuadraticDiscriminantAnalysis      0.73               0.74    None      0.73        0.22
# LabelPropagation                   0.67               0.62    None      0.69        7.54
# LabelSpreading                     0.67               0.62    None      0.69       17.39
# AdaBoostClassifier                 0.74               0.60    None      0.70        7.56
# DummyClassifier                    0.27               0.12    None      0.11        0.06