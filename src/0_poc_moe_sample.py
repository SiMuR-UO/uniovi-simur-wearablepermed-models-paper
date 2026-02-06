import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def build_gate(expert_a, expert_b, Xa, Xb, y):
    pa_val = expert_a.predict_proba(Xa)
    pb_val = expert_b.predict_proba(Xb)

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

def mixture_of_experts_predict_proba(expert_a, expert_b, gate, Xa, Xb):
    # Expert probabilities
    pa = expert_a.predict_proba(Xa)
    pb = expert_b.predict_proba(Xb)

    # Gate weights
    w = gate.predict_proba(np.hstack([Xa, Xb]))
    w_a = w[:, 1].reshape(-1, 1)
    w_b = w[:, 0].reshape(-1, 1)

    # Weighted mixture
    return w_a * pa + w_b * pb

print("🟢 Generate mock data")
X, y = make_classification(
    n_samples=6000,
    n_features=24,
    n_classes=8,    
    n_informative=14,
    n_redundant=2,
    n_clusters_per_class=1,
    random_state=42
)

print("🟢 Generate training/validation/test data for a and b")
# Split features into two "datasets"
X_a = X[:, :12]   # dataset A (6000,12)
X_b = X[:, 12:]   # dataset B (6000,12)

# First split: train vs temp
Xa_tr, Xa_tmp, Xb_tr, Xb_tmp, y_tr, y_tmp = train_test_split(
    X_a, X_b, y, test_size=0.4, random_state=42
)

# Second split: gate-val vs test
Xa_val, Xa_te, Xb_val, Xb_te, y_val, y_te = train_test_split(
    Xa_tmp, Xb_tmp, y_tmp, test_size=0.5, random_state=42
)

print("🟢 Train experts")
expert_a = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

expert_b = LogisticRegression(max_iter=2000)

print("🟢 Validate experts")
expert_a.fit(Xa_tr, y_tr)
expert_b.fit(Xb_tr, y_tr)

y_a_pred = expert_a.predict(Xa_val)
acc_score_test_a = accuracy_score(y_val, y_a_pred)

y_b_pred = expert_a.predict(Xb_val)
acc_score_test_b = accuracy_score(y_val, y_b_pred)

print(f"Expert A validation accuracy: {acc_score_test_a:.4f}")
print(f"Expert B validation accuracy: {acc_score_test_b:.4f}")

print("🟢 Build gate validation datasets")
X_gate_val = np.hstack([Xa_val, Xb_val])
y_gate_val = build_gate(expert_a, expert_b, Xa_val, Xb_val, y_val)

print("🟢 Training gate")
gate = LogisticRegression()
gate.fit(X_gate_val, y_gate_val)

print("🟢 Validate gate test datasets")
X_gate_te = np.hstack([Xa_te, Xb_te])
y_gate_te = build_gate(expert_a, expert_b, Xa_te, Xb_te, y_te)

gate_pred = gate.predict(X_gate_te)

gate_acc = (gate_pred == y_gate_te).mean()
print("Gate test accuracy:", gate_acc)

print("🟢 Validate MoE")
p_final = mixture_of_experts_predict_proba(expert_a, expert_b, gate, Xa_te, Xb_te) # (1200,8)
y_pred = p_final.argmax(axis=1) # (1200,)

accuracy = (y_pred == y_te).mean()
print("MoE test accuracy:", accuracy)