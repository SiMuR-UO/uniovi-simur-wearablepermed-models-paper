import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🟢 Imports and toy data")
X, y = make_classification(
    n_samples=6000,
    n_features=24,
    n_informative=14,
    n_redundant=2,
    n_classes=8,
    n_clusters_per_class=1,
    random_state=42
)

# Split features into two "datasets"
X_a = X[:, :12]   # dataset A
X_b = X[:, 12:]   # dataset B

Xa_tr, Xa_te, Xb_tr, Xb_te, y_tr, y_te = train_test_split(
    X_a, X_b, y, test_size=0.2, random_state=42
)

print("🟢 Train the base models")
rf_params = dict(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

base_a = RandomForestClassifier(**rf_params)
base_b = RandomForestClassifier(**rf_params)

base_a.fit(Xa_tr, y_tr)
base_b.fit(Xb_tr, y_tr)

print("🟢 Build stacking features (probability stacking)")
pa_tr = base_a.predict_proba(Xa_tr)
pb_tr = base_b.predict_proba(Xb_tr)

stack_X_tr = np.hstack([pa_tr, pb_tr])

pa_te = base_a.predict_proba(Xa_te)
pb_te = base_b.predict_proba(Xb_te)

stack_X_te = np.hstack([pa_te, pb_te])

print("🟢 Meta-model (stacker)")
meta_model = LogisticRegression(
    max_iter=2000,
    multi_class="auto"
)

meta_model.fit(stack_X_tr, y_tr)

print("🟢 Evaluation")
y_pred = meta_model.predict(stack_X_te)

acc = accuracy_score(y_te, y_pred)
print("Stacking accuracy:", acc)