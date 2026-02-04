import numpy as np
from collections import defaultdict

print("🟢 load stack PI and M")
stack_data_PI = np.load("/home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-models/output/cases_dataset_PI/Modelo_BRF_8superclases_PI/data_feature_all.npz")
stack_data_M = np.load("/home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-models/output/cases_dataset_M/Modelo_BRF_8superclases_M/data_feature_all.npz")

# get datasets from stack
X_data_PI = stack_data_PI["arr_0"]
y_data_PI = stack_data_PI["arr_1"]
m_data_PI = stack_data_PI["arr_2"]

X_data_M = stack_data_M["arr_0"]
y_data_M = stack_data_M["arr_1"]
m_data_M = stack_data_M["arr_2"]

print("🟢 sort PI and M datasets by participant and activity to be concatenated")
m_data_sorted_idx_PI = np.lexsort((y_data_PI, m_data_PI))

X_data_PI = X_data_PI[m_data_sorted_idx_PI]
y_data_PI = y_data_PI[m_data_sorted_idx_PI]
m_data_PI = m_data_PI[m_data_sorted_idx_PI]

m_data_sorted_idx_M = np.lexsort((y_data_M, m_data_M))

X_data_M = X_data_M[m_data_sorted_idx_M]
y_data_M = y_data_M[m_data_sorted_idx_M]
m_data_M = m_data_M[m_data_sorted_idx_M]

print("🟢 Common participants")
common_participants = np.intersect1d(np.unique(m_data_PI), np.unique(m_data_M))
print("Total common participants: " + str(len(common_participants)))

X_list = []
y_list = []
m_list = []

print("🟢 concatenated each common participant")
for p in common_participants:
    # Filter participant by p
    pi_mask = m_data_PI == p
    X_pi_p = X_data_PI[pi_mask]
    y_pi_p = y_data_PI[pi_mask]

    m_mask = m_data_M == p
    X_m_p = X_data_M[m_mask]
    y_m_p = y_data_M[m_mask]

    # Find common activities for this participant
    common_labels = np.intersect1d(np.unique(y_pi_p), np.unique(y_m_p))
    if len(common_labels) == 0:
        continue

    # Keep only rows with common labels
    pi_rows = np.isin(y_pi_p, common_labels)
    m_rows = np.isin(y_m_p, common_labels)

    X_pi_p = X_pi_p[pi_rows]
    y_pi_p = y_pi_p[pi_rows]

    X_m_p = X_m_p[m_rows]
    y_m_p = y_m_p[m_rows]

    # Ensure same number of rows
    if len(y_pi_p) != len(y_m_p):
        # Optional: intersect exact rows per label order
        # This aligns PI and M by label
        common_rows = []
        for lbl in common_labels:
            pi_idx = np.where(y_pi_p == lbl)[0]
            m_idx = np.where(y_m_p == lbl)[0]
            min_len = min(len(pi_idx), len(m_idx))
            common_rows.append((pi_idx[:min_len], m_idx[:min_len]))
        
        # Rebuild arrays with same number of rows
        X_pi_new, X_m_new, y_new = [], [], []
        for pi_idx, m_idx in common_rows:
            X_pi_new.append(X_pi_p[pi_idx])
            X_m_new.append(X_m_p[m_idx])
            y_new.append(y_pi_p[pi_idx])
        
        X_pi_p = np.vstack(X_pi_new)
        X_m_p = np.vstack(X_m_new)
        y_pi_p = np.concatenate(y_new)

    # Concatenate horizontally: M(91), PI(91)
    X_concat = np.hstack((X_m_p, X_pi_p))

    # add sample
    X_list.append(X_concat)
    y_list.append(y_pi_p)
    m_list.append(np.full(len(y_pi_p), p))

print("🟢 concatenated all participants")
X_Final = np.vstack(X_list)
y_Final = np.concatenate(y_list)
m_Final = np.concatenate(m_list)

print(X_Final.shape, y_Final.shape, m_Final.shape)

print("🟢 save concatenated participants")
np.savez("/home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-models/output/cases_dataset_PI_M/case_PI_M_BRF_acc_gyr_8_superclasses/Modelo_5_BRF_concatenado_MM/X_y_m_final.npz", arr_0=X_Final, arr_1=y_Final, arr_2=m_Final)