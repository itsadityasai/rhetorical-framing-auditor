"""
Aggregate Vector SVM for Bias Classification

This script addresses a fundamental issue with the original DFI approach:
DFI vectors from different triplets have different facts/clusters, so the
1st entry in one DFI doesn't relate to the 1st entry in another DFI.
Padding and concatenating these vectors mixes unrelated dimensions.

SOLUTION:
1. Normalize each DFI vector (subtract mean, divide by std) to have 0 mean and unit std
2. Extract aggregate statistical features that are consistent across all triplets
3. Train SVM on these fixed-length aggregate vectors instead of variable-length padded DFI

Aggregate features per triplet (for left-vs-center and right-vs-center):
- Mean of DFI deltas
- Std of DFI deltas
- Min of DFI deltas
- Max of DFI deltas
- Skewness (asymmetry of distribution)
- % positive deltas (how often side has higher prominence than center)
- % zero deltas (coverage overlap)
- % negative deltas (how often center has higher prominence)
- Number of clusters (triplet complexity)

These 9 features are CONSISTENT across all triplets regardless of cluster count.
"""

import argparse
import json
import math
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add parent directory for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = "aggregate-vector/results/aggregate_svm_results.json"
DEFAULT_MODEL_DIR = "aggregate-vector/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SVM on aggregate DFI features for bias classification"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)

    parser.add_argument("--svm-kernel", default="rbf")
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--svm-gamma", type=float, default=0.1)
    parser.add_argument("--svm-degree", type=int, default=3)

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


def W_coverage_only(depth: int, sat_count: int) -> float:
    """Binary coverage (1 if present, 0 otherwise) - handled separately."""
    return 1.0  # Just presence indicator


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
    """
    Build raw DFI deltas for each cluster.

    Returns:
        (deltas_left, deltas_right): Raw delta values (variable length)
    """
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
    """
    Build raw coverage-based DFI deltas.

    Returns:
        (deltas_left, deltas_right): Binary coverage deltas
    """
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
    """
    Normalize vector to have 0 mean and unit std.
    Returns original if std is 0 (constant vector).
    """
    arr = np.array(vec, dtype=float)
    if len(arr) == 0:
        return arr

    mean = np.mean(arr)
    std = np.std(arr)

    if std > 1e-10:
        return (arr - mean) / std
    else:
        # Constant vector - just center it
        return arr - mean


def compute_aggregate_features(deltas: List[float]) -> List[float]:
    """
    Extract aggregate statistical features from DFI deltas.

    Returns fixed-length feature vector (9 features):
    - mean, std, min, max
    - skewness
    - % positive, % zero, % negative
    - num_clusters (log-scaled)
    """
    if not deltas or len(deltas) == 0:
        # Return zeros for empty vectors
        return [0.0] * 9

    arr = np.array(deltas, dtype=float)
    n = len(arr)

    # Basic statistics
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # Skewness (asymmetry)
    if std_val > 1e-10:
        skewness = float(stats.skew(arr))
    else:
        skewness = 0.0

    # Distribution of signs
    pct_positive = float(np.sum(arr > 0) / n)
    pct_zero = float(np.sum(np.abs(arr) < 1e-10) / n)
    pct_negative = float(np.sum(arr < 0) / n)

    # Number of clusters (log-scaled for stability)
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
    """
    Normalize deltas first, then compute aggregate features.
    """
    if not deltas or len(deltas) == 0:
        return [0.0] * 9

    # Normalize the vector
    normalized = normalize_vector(deltas)

    return compute_aggregate_features(normalized.tolist())


def compute_aggregate_features_extended(deltas: List[float]) -> List[float]:
    """
    Extended aggregate features (14 features):
    - All basic aggregate features (9)
    - Kurtosis (tail heaviness)
    - Range (max - min)
    - Interquartile range
    - Median
    - Sum of absolute values (total divergence)
    """
    if not deltas or len(deltas) == 0:
        return [0.0] * 14

    arr = np.array(deltas, dtype=float)
    n = len(arr)

    # Basic features
    basic = compute_aggregate_features(deltas)

    # Additional features
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


# =============================================================================
# Training and Evaluation
# =============================================================================


def build_xy(rows: List[dict], key_left: str, key_right: str):
    x, y = [], []
    for row in rows:
        x.append(list(row[key_left]))
        y.append(0)
        x.append(list(row[key_right]))
        y.append(1)
    return np.array(x), np.array(y)


def evaluate(model, x: np.ndarray, y: np.ndarray) -> Dict:
    pred = model.predict(x)
    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def train_and_evaluate_experiment(
    train_rows: List[dict],
    val_rows: List[dict],
    test_rows: List[dict],
    feature_builder,
    svm_cfg: Dict,
    experiment_name: str,
) -> Tuple[object, int, Dict]:
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

    # Train
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
        ),
    )
    model.fit(x_train, y_train)

    metrics = {
        "train": evaluate(model, x_train, y_train),
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
    }

    return model, input_dim, metrics


def save_model(path: str, model, input_dim: int, experiment_name: str, svm_cfg: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model,
        "input_dim": int(input_dim),
        "experiment": experiment_name,
        "svm": svm_cfg,
        "created": datetime.now().isoformat(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    args = parse_args()

    svm_cfg = {
        "kernel": args.svm_kernel,
        "C": args.svm_c,
        "gamma": args.svm_gamma,
        "degree": args.svm_degree,
    }

    run_log = init_run_logging(
        script_subdir="aggregate-vector",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "svm": svm_cfg,
        },
    )

    print("=" * 70)
    print("Aggregate Vector SVM for Bias Classification")
    print("=" * 70)
    print("\nThis experiment addresses the dimension mismatch problem:")
    print("- Different triplets have different numbers of clusters")
    print("- Raw DFI concatenation mixes unrelated dimensions")
    print("- SOLUTION: Extract fixed-length aggregate statistics from each DFI")
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
    ]

    os.makedirs(args.model_dir, exist_ok=True)
    results = {}

    for exp_name, description, feature_builder in experiments:
        print(f"\n{'=' * 70}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {description}")
        print("=" * 70)

        # Deep copy rows to avoid mutation issues
        import copy

        train_rows = copy.deepcopy(facts_train)
        val_rows = copy.deepcopy(facts_val)
        test_rows = copy.deepcopy(facts_test)

        model, input_dim, metrics = train_and_evaluate_experiment(
            train_rows, val_rows, test_rows, feature_builder, svm_cfg, exp_name
        )

        model_path = os.path.join(args.model_dir, f"{exp_name}.pkl")
        save_model(model_path, model, input_dim, exp_name, svm_cfg)

        results[exp_name] = {
            "description": description,
            "input_dim": input_dim,
            "metrics": metrics,
            "model_path": model_path,
        }

        print(
            f"  Val:  acc={metrics['val']['accuracy']:.4f}, f1={metrics['val']['macro_f1']:.4f}"
        )
        print(
            f"  Test: acc={metrics['test']['accuracy']:.4f}, f1={metrics['test']['macro_f1']:.4f}"
        )

    # Reference baselines
    baselines = {
        "coverage_only_padded": {"test_acc": 0.7528, "test_f1": 0.7504},
        "log_depth_padded": {"test_acc": 0.7642, "test_f1": 0.7620},
        "structural_baseline": {"test_acc": 0.6705, "test_f1": 0.6702},
    }

    # Compute deltas
    for exp_name, exp_data in results.items():
        exp_data["delta_vs_coverage_padded"] = {
            "test_acc": exp_data["metrics"]["test"]["accuracy"]
            - baselines["coverage_only_padded"]["test_acc"],
            "test_f1": exp_data["metrics"]["test"]["macro_f1"]
            - baselines["coverage_only_padded"]["test_f1"],
        }
        exp_data["delta_vs_log_depth_padded"] = {
            "test_acc": exp_data["metrics"]["test"]["accuracy"]
            - baselines["log_depth_padded"]["test_acc"],
            "test_f1": exp_data["metrics"]["test"]["macro_f1"]
            - baselines["log_depth_padded"]["test_f1"],
        }

    output = {
        "setup": {
            "goal": "Test aggregate feature extraction to fix dimension mismatch problem",
            "problem": "Different triplets have different cluster counts, making raw DFI concatenation mix unrelated dimensions",
            "solution": "Extract fixed-length aggregate statistics from each DFI",
            "facts": args.facts,
            "split_dir": args.split_dir,
            "svm": svm_cfg,
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
    print("\n" + "=" * 90)
    print("SUMMARY: Aggregate Vector Results")
    print("=" * 90)
    print(
        f"\n{'Experiment':<25} {'Dim':<6} {'Test Acc':<12} {'Test F1':<12} {'vs Cov-Pad':<12} {'vs LogD-Pad':<12}"
    )
    print("-" * 90)

    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        test_f1 = exp_data["metrics"]["test"]["macro_f1"]
        delta_cov = exp_data["delta_vs_coverage_padded"]["test_acc"]
        delta_log = exp_data["delta_vs_log_depth_padded"]["test_acc"]
        dim = exp_data["input_dim"]
        print(
            f"{exp_name:<25} {dim:<6} {test_acc:<12.4f} {test_f1:<12.4f} {delta_cov:<+12.4f} {delta_log:<+12.4f}"
        )

    print("-" * 90)
    print("Reference baselines (padded DFI approach):")
    print(f"  coverage_only_padded:  75.28% test acc")
    print(f"  log_depth_padded:      76.42% test acc")
    print(f"  structural_baseline:   67.05% test acc")

    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out, "experiments": list(results.keys())})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
