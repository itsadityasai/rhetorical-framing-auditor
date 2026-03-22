import json
import numpy as np
from itertools import product

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# -------- CONFIG --------
TRAIN_PATH = "data/dfi_splits/train.json"
VAL_PATH = "data/dfi_splits/val.json"

AGGREGATE_FEATURES = False

SVM_KERNEL = "rbf"
SVM_C = 10
SVM_GAMMA = 0.1
SVM_DEGREE = 3 # ignored unless kernel = poly

SVM_KERNEL_OPTIONS = ["linear", "rbf", "poly"]
SWEEP_VALUES = [0.01, 0.1, 1, 10, 100, 1000, 10000]


# -------- LOAD --------
def load(path):
    with open(path) as f:
        return json.load(f)


# -------- BUILD DATA --------
def summarize_deltas(deltas):
    if len(deltas) == 0:
        return [0.0] * 10

    arr = np.array(deltas, dtype=float)
    pos = arr[arr > 0]
    neg = arr[arr < 0]

    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.max(arr)),
        float(np.min(arr)),
        float(np.sum(arr)),
        float(len(arr)),
        float(len(pos)),
        float(len(neg)),
        float(np.sum(pos)) if len(pos) > 0 else 0.0,
        float(np.sum(neg)) if len(neg) > 0 else 0.0,
    ]


def aggregate_features(side_deltas):
    side = summarize_deltas(side_deltas)
    return side


def build_xy_aggregated(data):
    X = []
    y = []

    for row in data:
        # LEFT sample
        X.append(aggregate_features(row["dfi_left"]))
        y.append(0)  # left

        # RIGHT sample
        X.append(aggregate_features(row["dfi_right"]))
        y.append(1)  # right

    return np.array(X, dtype=float), np.array(y)


def build_xy_raw(data):
    X = []
    y = []

    for row in data:
        X.append(list(row["dfi_left"]))
        y.append(0)

        X.append(list(row["dfi_right"]))
        y.append(1)

    return X, np.array(y)


def pad_or_truncate(raw_X, target_len):
    X = np.zeros((len(raw_X), target_len), dtype=float)
    for i, vec in enumerate(raw_X):
        limit = min(len(vec), target_len)
        if limit > 0:
            X[i, :limit] = np.array(vec[:limit], dtype=float)
    return X


def run(
    train_path=TRAIN_PATH,
    val_path=VAL_PATH,
    aggregate_features=AGGREGATE_FEATURES,
    kernel=SVM_KERNEL,
    C=SVM_C,
    gamma=SVM_GAMMA,
    degree=SVM_DEGREE,
):
    train = load(train_path)
    val = load(val_path)

    if aggregate_features:
        X_train, y_train = build_xy_aggregated(train)
        X_val, y_val = build_xy_aggregated(val)
    else:
        X_train_raw, y_train = build_xy_raw(train)
        X_val_raw, y_val = build_xy_raw(val)

        # Raw deltas are variable-length, so we fix dimensionality with zero-padding.
        max_len = max((len(v) for v in X_train_raw), default=0)
        X_train = pad_or_truncate(X_train_raw, max_len)
        X_val = pad_or_truncate(X_val_raw, max_len)

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
        )
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    rand_pred = np.random.randint(0, 2, size=len(y_val))
    rand_acc = float((rand_pred == y_val).mean())
    class_counts = np.bincount(y_train)

    print("\n=== SVM Validation Report ===")
    print(f"Features: {'aggregated' if aggregate_features else 'raw-padded'}")
    print(f"Train samples: {len(y_train)} | Val samples: {len(y_val)}")
    print(f"Input dim: {X_train.shape[1]}")
    print(f"Class balance (train) -> left: {class_counts[0]}, right: {class_counts[1]}")
    print(f"Params: kernel={kernel}, C={C}, gamma={gamma}, degree={degree}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Random baseline: {rand_acc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print("            pred_left  pred_right")
    print(f"true_left    {cm[0, 0]:>8}   {cm[0, 1]:>10}")
    print(f"true_right   {cm[1, 0]:>8}   {cm[1, 1]:>10}")

    return {
        "model": model,
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "random_accuracy": rand_acc,
    }


if __name__ == "__main__":
    total_runs = 0
    best = None

    for aggregate_features, kernel, C, gamma in product(
        [False, True],
        SVM_KERNEL_OPTIONS,
        SWEEP_VALUES,
        SWEEP_VALUES,
    ):
        degrees = range(1, 16) if kernel == "poly" else [SVM_DEGREE]

        for degree in degrees:
            total_runs += 1
            result = run(
                aggregate_features=aggregate_features,
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
            )

            acc = result["accuracy"]
            print(
                f"[{total_runs}] agg={aggregate_features} kernel={kernel} C={C} gamma={gamma} degree={degree} -> acc={acc:.4f}"
            )

            if best is None or acc > best["accuracy"]:
                best = {
                    "accuracy": acc,
                    "aggregate_features": aggregate_features,
                    "kernel": kernel,
                    "C": C,
                    "gamma": gamma,
                    "degree": degree,
                    "confusion_matrix": result["confusion_matrix"],
                }

    print("\n=== Sweep Complete ===")
    print(f"Total runs: {total_runs}")
    print(
        "Best config:",
        {
            "aggregate_features": best["aggregate_features"],
            "kernel": best["kernel"],
            "C": best["C"],
            "gamma": best["gamma"],
            "degree": best["degree"],
        },
    )
    print(f"Best accuracy: {best['accuracy']:.4f}")
    print("Best confusion matrix (rows=true, cols=pred):")
    print(np.array(best["confusion_matrix"]))