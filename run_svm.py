import json
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# -------- LOAD --------
def load(path):
    with open(path) as f:
        return json.load(f)

train = load("data/dfi_splits/train.json")
val = load("data/dfi_splits/val.json")


# -------- BUILD DATA --------
def build_xy_raw(data):
    X = []
    y = []

    for row in data:
        # LEFT sample
        X.append(list(row["dfi_left"]))
        y.append(0)  # left

        # RIGHT sample
        X.append(list(row["dfi_right"]))
        y.append(1)  # right

    return X, np.array(y)


def pad_or_truncate(raw_X, target_len):
    X = np.zeros((len(raw_X), target_len), dtype=float)
    for i, vec in enumerate(raw_X):
        limit = min(len(vec), target_len)
        if limit > 0:
            X[i, :limit] = np.array(vec[:limit], dtype=float)
    return X


X_train_raw, y_train = build_xy_raw(train)
X_val_raw, y_val = build_xy_raw(val)

# Raw deltas are variable-length, so we fix dimensionality with zero-padding.
max_len = max((len(v) for v in X_train_raw), default=0)
X_train = pad_or_truncate(X_train_raw, max_len)
X_val = pad_or_truncate(X_val_raw, max_len)


# -------- MODEL SEARCH --------
search_space = [
    {"kernel": "linear", "C": 0.1},
    {"kernel": "linear", "C": 1.0},
    {"kernel": "linear", "C": 3.0},
    {"kernel": "linear", "C": 10.0},
    {"kernel": "rbf", "C": 0.3, "gamma": "scale"},
    {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 3.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 3.0, "gamma": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.03},
    {"kernel": "poly", "C": 1.0, "degree": 2, "gamma": "scale"},
    {"kernel": "poly", "C": 3.0, "degree": 2, "gamma": "scale"},
]

best = None
results = []

for params in search_space:
    model = make_pipeline(
        StandardScaler(),
        SVC(**params)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    results.append((acc, params, model, y_pred))
    if best is None or acc > best[0]:
        best = (acc, params, model, y_pred)


# -------- EVAL (BEST) --------
best_acc, best_params, best_model, y_pred = best

print("Best params:", best_params)
print("Accuracy:", best_acc)

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix (rows=true, cols=pred):")
print(cm)

print("Top candidates:")
for rank, (acc, params, _, _) in enumerate(sorted(results, key=lambda x: x[0], reverse=True)[:5], start=1):
    print(f"{rank}. acc={acc:.4f} params={params}")

print("Left mean:", X_train[y_train == 0].mean(axis=0))
print("Right mean:", X_train[y_train == 1].mean(axis=0))
print(np.bincount(y_train))
# random baseline
rand_pred = np.random.randint(0, 2, size=len(y_val))
print("Random acc:", (rand_pred == y_val).mean())