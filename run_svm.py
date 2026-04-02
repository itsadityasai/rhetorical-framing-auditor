import json
import numpy as np
import yaml

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

svm_params = params["svm"]
svm_model_params = svm_params["model"]
svm_grid_params = svm_params["grid_search"]


# -------- CONFIG --------
TRAIN_PATH = svm_params["train_path"]
VAL_PATH = svm_params["val_path"]

AGGREGATE_FEATURE_OPTIONS = svm_params["aggregate_feature_options"]
INITIAL_AGGREGATE_FEATURES = svm_params["initial_aggregate_features"]

SVM_KERNEL = svm_model_params["kernel"]
SVM_C = svm_model_params["C"]
SVM_GAMMA = svm_model_params["gamma"]
SVM_DEGREE = svm_model_params["degree"]  # ignored unless kernel = poly

C_VALUES = svm_grid_params["c_values"]
GAMMA_VALUES = svm_grid_params["gamma_values"]

GRID_N_JOBS = svm_grid_params["n_jobs"]
GRID_VERBOSE = svm_grid_params["verbose"]
SWEEP_RESULTS_PATH = params["paths"]["files"]["svm_sweep_results"]

RUN_INITIAL_TEST = svm_grid_params["run_initial_test"]
RUN_GRID_SEARCH = svm_grid_params["run_grid_search"]


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
    return summarize_deltas(side_deltas)


def build_xy_aggregated(data):
    X = []
    y = []

    for row in data:
        X.append(aggregate_features(row["dfi_left"]))
        y.append(0)

        X.append(aggregate_features(row["dfi_right"]))
        y.append(1)

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


def prepare_train_val(train_data, val_data, aggregate_features):
    if aggregate_features:
        X_train, y_train = build_xy_aggregated(train_data)
        X_val, y_val = build_xy_aggregated(val_data)
        return X_train, y_train, X_val, y_val

    X_train_raw, y_train = build_xy_raw(train_data)
    X_val_raw, y_val = build_xy_raw(val_data)

    max_len = max((len(v) for v in X_train_raw), default=0)
    X_train = pad_or_truncate(X_train_raw, max_len)
    X_val = pad_or_truncate(X_val_raw, max_len)
    return X_train, y_train, X_val, y_val


def run(
    train_path=TRAIN_PATH,
    val_path=VAL_PATH,
    aggregate_features=False,
    kernel=SVM_KERNEL,
    C=SVM_C,
    gamma=SVM_GAMMA,
    degree=SVM_DEGREE,
):
    train = load(train_path)
    val = load(val_path)

    X_train, y_train, X_val, y_val = prepare_train_val(train, val, aggregate_features)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print("\n=== SVM Validation Report ===")
    print(f"Features: {'aggregated' if aggregate_features else 'raw-padded'}")
    print(f"Train samples: {len(y_train)} | Val samples: {len(y_val)}")
    print(f"Input dim: {X_train.shape[1]}")
    print(f"Params: kernel={kernel}, C={C}, gamma={gamma}, degree={degree}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print("            pred_left  pred_right")
    print(f"true_left    {cm[0, 0]:>8}   {cm[0, 1]:>10}")
    print(f"true_right   {cm[1, 0]:>8}   {cm[1, 1]:>10}")

    return {
        "model": model,
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "params": {
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "degree": degree,
            "aggregate_features": aggregate_features,
        },
    }


def build_param_grid():
    return [
        {
            "svc__kernel": ["linear"],
            "svc__C": C_VALUES,
        },
        {
            "svc__kernel": ["rbf"],
            "svc__C": C_VALUES,
            "svc__gamma": GAMMA_VALUES,
        },
    ]


def count_candidates(param_grid):
    return sum(len(list(ParameterGrid([grid]))) for grid in param_grid)


def grid_search_for_mode(train_data, val_data, aggregate_features):
    X_train, y_train, X_val, y_val = prepare_train_val(train_data, val_data, aggregate_features)

    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])

    # Use train as training fold (-1) and val as validation fold (0)
    test_fold = np.array([-1] * len(y_train) + [0] * len(y_val))
    split = PredefinedSplit(test_fold)

    pipeline = make_pipeline(StandardScaler(), SVC())

    param_grid = build_param_grid()
    total_candidates = count_candidates(param_grid)
    print(f"Grid candidates for aggregate_features={aggregate_features}: {total_candidates}")
    print(f"Progress will be shown by sklearn as [CV ...; x/{total_candidates}]")

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=split,
        n_jobs=GRID_N_JOBS,
        verbose=GRID_VERBOSE,
        refit=False,
        return_train_score=False,
    )

    gs.fit(X_all, y_all)

    # Build per-combo records from native sklearn results
    records = []
    n = len(gs.cv_results_["params"])
    for i in range(n):
        rec = {
            "aggregate_features": aggregate_features,
            "params": gs.cv_results_["params"][i],
            "accuracy": float(gs.cv_results_["mean_test_score"][i]),
            "rank": int(gs.cv_results_["rank_test_score"][i]),
            "fit_time": float(gs.cv_results_["mean_fit_time"][i]),
            "score_time": float(gs.cv_results_["mean_score_time"][i]),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "input_dim": int(X_train.shape[1]),
        }
        records.append(rec)

    # Evaluate confusion matrix for best params in this mode
    best_idx = int(np.argmax(gs.cv_results_["mean_test_score"]))
    best_params = gs.cv_results_["params"][best_idx]
    best_kernel = best_params["svc__kernel"]
    best_C = best_params["svc__C"]
    best_gamma = best_params.get("svc__gamma", "scale")
    best_degree = best_params.get("svc__degree", 3)

    best_eval = run(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        aggregate_features=aggregate_features,
        kernel=best_kernel,
        C=best_C,
        gamma=best_gamma,
        degree=best_degree,
    )

    best_summary = {
        "aggregate_features": aggregate_features,
        "params": {
            "kernel": best_kernel,
            "C": best_C,
            "gamma": best_gamma,
            "degree": best_degree,
        },
        "accuracy": best_eval["accuracy"],
        "confusion_matrix": best_eval["confusion_matrix"],
    }

    return records, best_summary


if __name__ == "__main__":
    if RUN_INITIAL_TEST:
        print("\n===== Initial single run (no grid) =====")
        run(
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            aggregate_features=INITIAL_AGGREGATE_FEATURES,
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            degree=SVM_DEGREE,
        )

    if not RUN_GRID_SEARCH:
        print("\nGrid search skipped (RUN_GRID_SEARCH=False)")
        raise SystemExit(0)

    train = load(TRAIN_PATH)
    val = load(VAL_PATH)

    all_records = []
    best_summaries = []

    for mode in AGGREGATE_FEATURE_OPTIONS:
        print(f"\n===== Grid search mode: aggregate_features={mode} =====")
        records, best_summary = grid_search_for_mode(train, val, mode)
        all_records.extend(records)
        best_summaries.append(best_summary)

    best_overall = max(best_summaries, key=lambda x: x["accuracy"])

    payload = {
        "grid_n_jobs": GRID_N_JOBS,
        "grid_verbose": GRID_VERBOSE,
        "c_values": C_VALUES,
        "gamma_values": GAMMA_VALUES,
        "total_runs": len(all_records),
        "best_per_mode": best_summaries,
        "best_overall": best_overall,
        "results": sorted(all_records, key=lambda r: r["accuracy"], reverse=True),
    }

    with open(SWEEP_RESULTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Grid Search Complete ===")
    print(f"Total runs: {len(all_records)}")
    print("Best overall:", best_overall)
    print(f"Saved sweep results to {SWEEP_RESULTS_PATH}")