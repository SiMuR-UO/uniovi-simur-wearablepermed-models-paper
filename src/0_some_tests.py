import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

WINDOW_DATA = "arr_0"
WINDOW_LABELS = "arr_1"
WINDOW_METADATA = "arr_2"

def participant_group_split(X_data, y_data, m_data, test_size=0.2):
    # Transporm string labels to numbers
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size)

    train_idx, test_idx = next(gss.split(X_data, y_data, m_data))

    X_train, X_test = X_data[train_idx], X_data[test_idx]
    y_train, y_test = y_data[train_idx], y_data[test_idx]
    m_train, m_test = m_data[train_idx], m_data[test_idx]

    print(f"Unique participants in train: {np.unique(m_train)}")
    print(f"Unique participants in test:  {np.unique(m_data[test_idx])}")

    # split concatenated dataset between PI and M
    X_train_M = X_train[:, :91]
    X_train_PI = X_train[:, 91:]

    X_test_M = X_test[:, :91]
    X_test_PI = X_test[:, 91:]
    
    return X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test

print("🟢 load stack PI+M")
stack_data_all = np.load("/mnt/simur-fileserver/data/wearablepermed/output/desync_5_seconds_case/data_feature_all.npz")

X_data = stack_data_all[WINDOW_DATA]
y_data = stack_data_all[WINDOW_LABELS]
m_data = stack_data_all[WINDOW_METADATA]

print(f"Total Unique participants: {len(np.unique(m_data))}")
print(f"Unique participants: {np.unique(m_data)}")

print("🟢 Split dataset PI+M")
(X_train_PI, X_test_PI, X_train_M, X_test_M, y_train, y_test, m_train, m_test) = participant_group_split(X_data, y_data, m_data)

print("\n")
print(f"PI X Train size: {X_train_PI.shape}, PI y Train size: {y_train.shape}, PI X Test size: {X_test_PI.shape}, PI y Test size: {y_test.shape}")
print(f"M X Train size: {X_train_M.shape}, M y Train size: {y_train.shape}, M X Test size: {X_test_M.shape}, M y Test size: {y_test.shape}")
print("\n")

mask = m_data == "PMP1042"
X_selected = X_data[mask]