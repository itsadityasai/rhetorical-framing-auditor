"""
Aggregate Vector Random Forest for Bias Classification

This script uses the same aggregate feature extraction as aggregate-vector/train_aggregate_svm.py
but trains Random Forest instead of SVM.

Motivation:
- Random Forest achieved 87.78% on padded DFI vectors (vs 77.84% for SVM)
- Aggregate SVM achieved only 61.36%
- Question: Can Random Forest rescue the aggregate feature approach?

Aggregate features per triplet (for left-vs-center and right-vs-center):
- Mean, std, min, max of DFI deltas
- Skewness (asymmetry of distribution)
- % positive deltas (side > center prominence)
- % zero deltas (equal coverage)
- % negative deltas (center > side prominence)
- Log number of clusters (triplet complexity)

These 9 features are consistent across all triplets regardless of cluster count.
"""

import argparse
import copy
import json
import math
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV

# Add project root for module imports
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "params.yaml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(PROJECT_ROOT))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = "experiments/aggregate-rf/results/aggregate_rf_results.json"
DEFAULT_MODEL_DIR = "experiments/aggregate-rf/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Random Forest on aggregate DFI features for bias classification"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)

    # Random Forest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)  # None = unlimited
    parser.add_argument("--min-samples-split", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_split_triplet_idx(split_rows: List[dict]) -> Set[int]:
    return {row["triplet_idx"] for row in split_rows if "triplet_idx" in row}


def split_facts_by_existing_splits(
    facts_rows: List[dict],
    train_ids: Set[int],
    val_ids: Set[int],
    test_ids: Set[int],
) -> Tuple[List[dict], List[dict], List[dict]]:
    train_rows, val_rows, test_rows = [], [], []
    for row in facts_rows:
        idx = row.get("triplet_idx")
        if idx in train_ids:
            train_rows.append(row)
        elif idx in val_ids:
            val_rows.append(row)
        elif idx in test_ids:
            test_rows.append(row)
    return train_rows, val_rows, test_rows


# =============================================================================
# Prominence Functions
# =============================================================================


def W_log_depth(depth: int, sat_count: int) -> float:
    """Best-performing formula from Option B."""
    return 1.0 / (1.0 + math.log1p(depth))


def compute_side_prominence(
    edus: List[str], edu_lookup: Dict, side: str, W_func
) -> float:
    """Compute max prominence score for a side using given W function."""
    scores = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            depth = meta.get("depth", 0)
            sat_count = meta.get("satellite_edges_to_root", 0)
            scores.append(W_func(depth, sat_count))
    return max(scores) if scores else 0.0


def compute_side_coverage(edus: List[str], edu_lookup: Dict, side: str) -> int:
    """Binary coverage: 1 if side present, 0 otherwise."""
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            return 1
    return 0


# =============================================================================
# DFI Construction (raw deltas per cluster)
# =============================================================================


def build_raw_dfi(fact_row: Dict, W_func) -> Tuple[List[float], List[float]]:
    """Build raw DFI deltas for each cluster."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        W_left = compute_side_prominence(edus, edu_lookup, "left", W_func)
        W_center = compute_side_prominence(edus, edu_lookup, "center", W_func)
        W_right = compute_side_prominence(edus, edu_lookup, "right", W_func)

        deltas_left.append(W_left - W_center)
        deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


def build_raw_coverage_dfi(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build raw coverage-based DFI deltas."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        cov_left = compute_side_coverage(edus, edu_lookup, "left")
        cov_center = compute_side_coverage(edus, edu_lookup, "center")
        cov_right = compute_side_coverage(edus, edu_lookup, "right")

        deltas_left.append(cov_left - cov_center)
        deltas_right.append(cov_right - cov_center)

    return deltas_left, deltas_right


# =============================================================================
# Aggregate Feature Extraction
# =============================================================================


def normalize_vector(vec: List[float]) -> np.ndarray:
    """Normalize vector to have 0 mean and unit std."""
    arr = np.array(vec, dtype=float)
    if len(arr) == 0:
        return arr
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 1e-10:
        return (arr - mean) / std
    else:
        return arr - mean


def compute_aggregate_features(deltas: List[float]) -> List[float]:
    """
    Extract aggregate statistical features from DFI deltas.

    Returns fixed-length feature vector (9 features):
    - mean, std, min, max
    - skewness
    - % positive, % zero, % negative
    - log_num_clusters
    """
    if not deltas or len(deltas) == 0:
        return [0.0] * 9

    arr = np.array(deltas, dtype=float)
    n = len(arr)

    # Basic statistics
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # Skewness
    if std_val > 1e-10:
        skewness = float(stats.skew(arr))
    else:
        skewness = 0.0

    # Distribution of signs
    pct_positive = float(np.sum(arr > 0) / n)
    pct_zero = float(np.sum(np.abs(arr) < 1e-10) / n)
    pct_negative = float(np.sum(arr < 0) / n)

    # Number of clusters (log-scaled)
    log_num_clusters = math.log1p(n)

    return [
        mean_val,
        std_val,
        min_val,
        max_val,
        skewness,
        pct_positive,
        pct_zero,
        pct_negative,
        log_num_clusters,
    ]


def compute_aggregate_features_normalized(deltas: List[float]) -> List[float]:
    """Normalize deltas first, then compute aggregate features."""
    if not deltas or len(deltas) == 0:
        return [0.0] * 9
    normalized = normalize_vector(deltas)
    return compute_aggregate_features(normalized.tolist())


def compute_aggregate_features_extended(deltas: List[float]) -> List[float]:
    """
    Extended aggregate features (14 features):
    - All basic aggregate features (9)
    - Kurtosis, range, IQR, median, sum_abs
    """
    if not deltas or len(deltas) == 0:
        return [0.0] * 14

    arr = np.array(deltas, dtype=float)
    n = len(arr)
    basic = compute_aggregate_features(deltas)

    if np.std(arr) > 1e-10 and n >= 4:
        kurtosis = float(stats.kurtosis(arr))
    else:
        kurtosis = 0.0

    range_val = float(np.max(arr) - np.min(arr))
    if n >= 4:
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = float(q3 - q1)
    else:
        iqr = range_val

    median_val = float(np.median(arr))
    sum_abs = float(np.sum(np.abs(arr)))

    return basic + [kurtosis, range_val, iqr, median_val, sum_abs]


# =============================================================================
# Combined Feature Builders
# =============================================================================


def build_aggregate_coverage(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build aggregate features using coverage-only DFI."""
    deltas_left, deltas_right = build_raw_coverage_dfi(fact_row)
    return (
        compute_aggregate_features(deltas_left),
        compute_aggregate_features(deltas_right),
    )


def build_aggregate_coverage_normalized(
    fact_row: Dict,
) -> Tuple[List[float], List[float]]:
    """Build aggregate features using normalized coverage DFI."""
    deltas_left, deltas_right = build_raw_coverage_dfi(fact_row)
    return (
        compute_aggregate_features_normalized(deltas_left),
        compute_aggregate_features_normalized(deltas_right),
    )


def build_aggregate_log_depth(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build aggregate features using log-depth prominence."""
    deltas_left, deltas_right = build_raw_dfi(fact_row, W_log_depth)
    return (
        compute_aggregate_features(deltas_left),
        compute_aggregate_features(deltas_right),
    )


def build_aggregate_log_depth_normalized(
    fact_row: Dict,
) -> Tuple[List[float], List[float]]:
    """Build aggregate features using normalized log-depth DFI."""
    deltas_left, deltas_right = build_raw_dfi(fact_row, W_log_depth)
    return (
        compute_aggregate_features_normalized(deltas_left),
        compute_aggregate_features_normalized(deltas_right),
    )


def build_aggregate_combined(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    Combine coverage and structural aggregate features.
    Total: 9 coverage + 9 structural = 18 features per side.
    """
    cov_left, cov_right = build_aggregate_coverage(fact_row)
    str_left, str_right = build_aggregate_log_depth(fact_row)
    return (cov_left + str_left, cov_right + str_right)


def build_aggregate_extended(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build extended aggregate features (14 features)."""
    deltas_left, deltas_right = build_raw_dfi(fact_row, W_log_depth)
    return (
        compute_aggregate_features_extended(deltas_left),
        compute_aggregate_features_extended(deltas_right),
    )


def build_aggregate_combined_extended(
    fact_row: Dict,
) -> Tuple[List[float], List[float]]:
    """
    Combine coverage and extended structural aggregate features.
    Total: 9 coverage + 14 structural = 23 features per side.
    """
    cov_left, cov_right = build_aggregate_coverage(fact_row)
    ext_left, ext_right = build_aggregate_extended(fact_row)
    return (cov_left + ext_left[:14], cov_right + ext_right[:14])


# =============================================================================
# Training and Evaluation
# =============================================================================


def build_xy(rows: List[dict], key_left: str, key_right: str):
    x, y = [], []
    for row in rows:
        x.append(list(row[key_left]))
        y.append(0)  # left-vs-center
        x.append(list(row[key_right]))
        y.append(1)  # right-vs-center
    return np.array(x), np.array(y)


def evaluate(model, x: np.ndarray, y: np.ndarray) -> Dict:
    pred = model.predict(x)
    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def tune_random_forest(X_train, y_train, seed: int, cv_folds: int):
    """Tune random forest with grid search."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_and_evaluate_experiment(
    train_rows: List[dict],
    val_rows: List[dict],
    test_rows: List[dict],
    feature_builder: Callable,
    rf_cfg: Dict,
    experiment_name: str,
    tune: bool = False,
    cv_folds: int = 5,
) -> Tuple[object, int, Dict, Dict]:
    """Run a single experiment with given feature builder."""

    # Build features
    for row in train_rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)
    for row in val_rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)
    for row in test_rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)

    # Build X, y (fixed length - no padding needed!)
    x_train, y_train = build_xy(train_rows, "feat_left", "feat_right")
    x_val, y_val = build_xy(val_rows, "feat_left", "feat_right")
    x_test, y_test = build_xy(test_rows, "feat_left", "feat_right")

    input_dim = x_train.shape[1]
    print(f"  Input dimension: {input_dim} (fixed across all samples)")

    # Combine train+val for tuning/final training
    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.concatenate([y_train, y_val])

    best_params = None

    if tune:
        print(f"  Running hyperparameter tuning with {cv_folds}-fold CV...")
        model, best_params, cv_score = tune_random_forest(
            x_trainval, y_trainval, rf_cfg["seed"], cv_folds
        )
        print(f"  Best params: {best_params}")
        print(f"  Best CV score: {cv_score:.4f}")
    else:
        # Use specified hyperparameters
        model = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_split=rf_cfg["min_samples_split"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            random_state=rf_cfg["seed"],
            n_jobs=-1,
        )
        model.fit(x_trainval, y_trainval)
        best_params = {
            "n_estimators": rf_cfg["n_estimators"],
            "max_depth": rf_cfg["max_depth"],
            "min_samples_split": rf_cfg["min_samples_split"],
            "min_samples_leaf": rf_cfg["min_samples_leaf"],
        }

    metrics = {
        "train": evaluate(model, x_train, y_train),
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
        "trainval": evaluate(model, x_trainval, y_trainval),
    }

    # Get feature importances
    feature_importances = model.feature_importances_.tolist()

    return (
        model,
        input_dim,
        metrics,
        {"best_params": best_params, "feature_importances": feature_importances},
    )


def save_model(path: str, model, input_dim: int, experiment_name: str, rf_cfg: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model,
        "input_dim": int(input_dim),
        "experiment": experiment_name,
        "rf_config": rf_cfg,
        "created": datetime.now().isoformat(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    args = parse_args()

    rf_cfg = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth if args.max_depth != 0 else None,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "seed": args.seed,
    }

    run_log = init_run_logging(
        script_subdir="aggregate-rf",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "rf": rf_cfg,
            "tune": args.tune,
            "cv_folds": args.cv_folds,
        },
    )

    print("=" * 70)
    print("Aggregate Vector Random Forest for Bias Classification")
    print("=" * 70)
    print("\nThis experiment tests if Random Forest can improve on aggregate features.")
    print("Previous results:")
    print("  - Aggregate SVM: 61.36% test accuracy (failed)")
    print("  - Padded DFI + RF: 87.78% test accuracy (best)")
    print()

    # Load data
    print("Loading facts and splits...")
    facts = load_json(args.facts)
    train_split = load_json(os.path.join(args.split_dir, "train.json"))
    val_split = load_json(os.path.join(args.split_dir, "val.json"))
    test_split = load_json(os.path.join(args.split_dir, "test.json"))

    train_ids = get_split_triplet_idx(train_split)
    val_ids = get_split_triplet_idx(val_split)
    test_ids = get_split_triplet_idx(test_split)

    facts_train, facts_val, facts_test = split_facts_by_existing_splits(
        facts, train_ids, val_ids, test_ids
    )

    print(
        f"Data: train={len(facts_train)}, val={len(facts_val)}, test={len(facts_test)}"
    )

    # Define experiments
    experiments = [
        # Coverage-based aggregate
        (
            "agg_coverage",
            "Aggregate features from coverage-only DFI (9 features)",
            build_aggregate_coverage,
        ),
        (
            "agg_coverage_norm",
            "Aggregate features from NORMALIZED coverage DFI (9 features)",
            build_aggregate_coverage_normalized,
        ),
        # Structural (log-depth) aggregate
        (
            "agg_log_depth",
            "Aggregate features from log-depth DFI (9 features)",
            build_aggregate_log_depth,
        ),
        (
            "agg_log_depth_norm",
            "Aggregate features from NORMALIZED log-depth DFI (9 features)",
            build_aggregate_log_depth_normalized,
        ),
        # Combined
        (
            "agg_combined",
            "Combined coverage + structural aggregate (18 features)",
            build_aggregate_combined,
        ),
        # Extended features
        (
            "agg_extended",
            "Extended aggregate features from log-depth DFI (14 features)",
            build_aggregate_extended,
        ),
        # Combined extended
        (
            "agg_combined_extended",
            "Combined coverage + extended structural (23 features)",
            build_aggregate_combined_extended,
        ),
    ]

    os.makedirs(args.model_dir, exist_ok=True)
    results = {}

    for exp_name, description, feature_builder in experiments:
        print(f"\n{'=' * 70}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {description}")
        print("=" * 70)

        train_rows = copy.deepcopy(facts_train)
        val_rows = copy.deepcopy(facts_val)
        test_rows = copy.deepcopy(facts_test)

        model, input_dim, metrics, extra_info = train_and_evaluate_experiment(
            train_rows,
            val_rows,
            test_rows,
            feature_builder,
            rf_cfg,
            exp_name,
            tune=args.tune,
            cv_folds=args.cv_folds,
        )

        model_path = os.path.join(args.model_dir, f"{exp_name}.pkl")
        save_model(model_path, model, input_dim, exp_name, rf_cfg)

        results[exp_name] = {
            "description": description,
            "input_dim": input_dim,
            "metrics": metrics,
            "model_path": model_path,
            "best_params": extra_info["best_params"],
            "feature_importances": extra_info["feature_importances"],
        }

        print(
            f"  Val:  acc={metrics['val']['accuracy']:.4f}, f1={metrics['val']['macro_f1']:.4f}"
        )
        print(
            f"  Test: acc={metrics['test']['accuracy']:.4f}, f1={metrics['test']['macro_f1']:.4f}"
        )

    # Reference baselines
    baselines = {
        "aggregate_svm_best": {"test_acc": 0.6136, "test_f1": 0.6065},
        "coverage_only_padded_svm": {"test_acc": 0.7528, "test_f1": 0.7504},
        "padded_dfi_rf_tuned": {"test_acc": 0.8778, "test_f1": 0.8760},
    }

    # Compute deltas
    for exp_name, exp_data in results.items():
        exp_data["delta_vs_aggregate_svm"] = {
            "test_acc": exp_data["metrics"]["test"]["accuracy"]
            - baselines["aggregate_svm_best"]["test_acc"],
            "test_f1": exp_data["metrics"]["test"]["macro_f1"]
            - baselines["aggregate_svm_best"]["test_f1"],
        }
        exp_data["delta_vs_padded_rf"] = {
            "test_acc": exp_data["metrics"]["test"]["accuracy"]
            - baselines["padded_dfi_rf_tuned"]["test_acc"],
            "test_f1": exp_data["metrics"]["test"]["macro_f1"]
            - baselines["padded_dfi_rf_tuned"]["test_f1"],
        }

    output = {
        "setup": {
            "goal": "Test if Random Forest can improve aggregate feature classification",
            "hypothesis": "RF may capture non-linear patterns in aggregate features better than SVM",
            "facts": args.facts,
            "split_dir": args.split_dir,
            "rf": rf_cfg,
            "tuned": args.tune,
            "cv_folds": args.cv_folds,
            "created": datetime.now().isoformat(),
        },
        "aggregate_features": {
            "basic_9": [
                "mean",
                "std",
                "min",
                "max",
                "skewness",
                "pct_positive",
                "pct_zero",
                "pct_negative",
                "log_num_clusters",
            ],
            "extended_14": [
                "mean",
                "std",
                "min",
                "max",
                "skewness",
                "pct_positive",
                "pct_zero",
                "pct_negative",
                "log_num_clusters",
                "kurtosis",
                "range",
                "iqr",
                "median",
                "sum_abs",
            ],
        },
        "reference_baselines": baselines,
        "experiments": results,
    }

    save_json(args.out, output)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Aggregate Vector Random Forest Results")
    print("=" * 100)
    print(
        f"\n{'Experiment':<25} {'Dim':<6} {'Test Acc':<12} {'Test F1':<12} {'vs Agg-SVM':<12} {'vs Pad-RF':<12}"
    )
    print("-" * 100)

    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        test_f1 = exp_data["metrics"]["test"]["macro_f1"]
        delta_svm = exp_data["delta_vs_aggregate_svm"]["test_acc"]
        delta_rf = exp_data["delta_vs_padded_rf"]["test_acc"]
        dim = exp_data["input_dim"]
        print(
            f"{exp_name:<25} {dim:<6} {test_acc:<12.4f} {test_f1:<12.4f} {delta_svm:<+12.4f} {delta_rf:<+12.4f}"
        )

    print("-" * 100)
    print("Reference baselines:")
    print(f"  aggregate_svm_best:       61.36% test acc")
    print(f"  coverage_only_padded_svm: 75.28% test acc")
    print(f"  padded_dfi_rf_tuned:      87.78% test acc (BEST)")

    # Find best experiment
    best_exp = max(results.items(), key=lambda x: x[1]["metrics"]["test"]["accuracy"])
    print(f"\nBest aggregate RF experiment: {best_exp[0]}")
    print(f"  Test accuracy: {best_exp[1]['metrics']['test']['accuracy']:.4f}")
    print(
        f"  vs Aggregate SVM: {best_exp[1]['delta_vs_aggregate_svm']['test_acc']:+.4f}"
    )
    print(f"  vs Padded RF:     {best_exp[1]['delta_vs_padded_rf']['test_acc']:+.4f}")

    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out, "experiments": list(results.keys())})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
