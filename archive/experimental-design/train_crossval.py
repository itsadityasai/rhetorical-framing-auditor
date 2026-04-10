"""
Experimental Design Improvements (Option D): Cross-validation and Bootstrap CI.

This script implements Option D to provide more robust evaluation:

1. K-Fold Cross-Validation:
   - Stratified k-fold instead of fixed train/val/test splits
   - Reports mean ± std accuracy across folds
   - Reduces variance from a single split

2. Bootstrap Confidence Intervals:
   - Resample test predictions with replacement
   - Compute 95% CI for accuracy estimates
   - Quantifies uncertainty in reported numbers

3. Relaxed Filtering (Optional):
   - Increase sample size by relaxing cluster constraints
   - Analyze performance vs. sample size tradeoff

Key insight: Original fixed split may have high variance due to small test set (176 triplets).
Cross-validation and bootstrap provide more reliable performance estimates.
"""

import argparse
import json
import os
import sys
import math
from datetime import datetime
from typing import Dict, List, Tuple, Callable
import copy

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add parent directory for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_OUT_PATH = "experimental-design/results/crossval_results.json"
DEFAULT_MODEL_DIR = "experimental-design/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validation and bootstrap CI experiments (Option D)"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)

    parser.add_argument("--svm-kernel", default="rbf")
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--svm-gamma", type=float, default=0.1)
    parser.add_argument("--svm-degree", type=int, default=3)

    parser.add_argument("--k-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000, help="Bootstrap iterations"
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# FEATURE BUILDERS (same as hybrid approach)
# =============================================================================


def get_cluster_coverage(edus: List[str], edu_lookup: Dict) -> Dict[str, int]:
    """Count EDUs per bias side in a cluster."""
    counts = {"left": 0, "center": 0, "right": 0}
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta is None:
            continue
        bias = meta.get("bias")
        if bias in VALID_BIASES:
            counts[bias] += 1
    return counts


def W_log_depth(depth: int, sat_count: int) -> float:
    """Logarithmic depth penalty (best from Option B)."""
    return 1.0 / (1.0 + math.log1p(depth))


def compute_side_prominence(
    edus: List[str], edu_lookup: Dict, side: str, W_func: Callable
) -> float:
    """Compute max prominence score for a side."""
    scores = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            depth = meta.get("depth", 0)
            sat_count = meta.get("satellite_edges_to_root", 0)
            scores.append(W_func(depth, sat_count))
    return max(scores) if scores else 0.0


def build_coverage_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build binary coverage features (delta encoding)."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        coverage = get_cluster_coverage(edus, edu_lookup)

        has_left = 1 if coverage["left"] > 0 else 0
        has_center = 1 if coverage["center"] > 0 else 0
        has_right = 1 if coverage["right"] > 0 else 0

        features_left.append(has_left - has_center)
        features_right.append(has_right - has_center)

    return features_left, features_right


def build_combined_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build combined features: coverage + structural (log depth)."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        coverage = get_cluster_coverage(edus, edu_lookup)

        # Coverage features
        has_left = 1 if coverage["left"] > 0 else 0
        has_center = 1 if coverage["center"] > 0 else 0
        has_right = 1 if coverage["right"] > 0 else 0

        cov_left = has_left - has_center
        cov_right = has_right - has_center

        # Structural features
        W_left = compute_side_prominence(edus, edu_lookup, "left", W_log_depth)
        W_center = compute_side_prominence(edus, edu_lookup, "center", W_log_depth)
        W_right = compute_side_prominence(edus, edu_lookup, "right", W_log_depth)

        str_left = W_left - W_center
        str_right = W_right - W_center

        # Combined: [cov, str] per cluster
        features_left.extend([cov_left, str_left])
        features_right.extend([cov_right, str_right])

    return features_left, features_right


def build_structural_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build structural prominence features using log depth formula."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        W_left = compute_side_prominence(edus, edu_lookup, "left", W_log_depth)
        W_center = compute_side_prominence(edus, edu_lookup, "center", W_log_depth)
        W_right = compute_side_prominence(edus, edu_lookup, "right", W_log_depth)

        deltas_left.append(W_left - W_center)
        deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


# =============================================================================
# DATA PREPARATION
# =============================================================================


def build_features_for_rows(rows: List[dict], feature_builder: Callable):
    """Apply feature builder to all rows."""
    for row in rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)


def build_xy(rows: List[dict], key_left: str, key_right: str):
    """Build X, y arrays from feature rows."""
    x, y = [], []
    for row in rows:
        x.append(list(row[key_left]))
        y.append(0)  # left-vs-center
        x.append(list(row[key_right]))
        y.append(1)  # right-vs-center
    return x, np.array(y)


def pad_or_truncate(raw_x: List[List[float]], target_len: int) -> np.ndarray:
    """Pad shorter vectors with zeros, truncate longer ones."""
    arr = np.zeros((len(raw_x), target_len), dtype=float)
    for i, vec in enumerate(raw_x):
        lim = min(len(vec), target_len)
        if lim > 0:
            arr[i, :lim] = np.array(vec[:lim], dtype=float)
    return arr


def prepare_dataset(
    facts_rows: List[dict], feature_builder: Callable
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Prepare full dataset with given feature builder."""
    rows = copy.deepcopy(facts_rows)
    build_features_for_rows(rows, feature_builder)
    x_raw, y = build_xy(rows, "feat_left", "feat_right")
    max_len = max((len(v) for v in x_raw), default=0)
    X = pad_or_truncate(x_raw, max_len)
    return X, y, max_len


# =============================================================================
# EXPERIMENT 1: K-Fold Cross-Validation
# =============================================================================


def run_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int,
    svm_cfg: Dict,
    seed: int,
    feature_name: str,
) -> Dict:
    """
    Run stratified k-fold cross-validation.
    Returns mean, std, and per-fold accuracies.
    """
    print(f"\n  Running {k_folds}-fold CV for {feature_name}...")

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
        ),
    )

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_accuracies = []
    fold_f1s = []
    fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone and train model
        fold_model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel=svm_cfg["kernel"],
                C=svm_cfg["C"],
                gamma=svm_cfg["gamma"],
                degree=svm_cfg["degree"],
            ),
        )
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        fold_accuracies.append(acc)
        fold_f1s.append(f1)
        fold_details.append(
            {
                "fold": fold_idx + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": float(acc),
                "f1": float(f1),
            }
        )

        print(f"    Fold {fold_idx + 1}: acc={acc:.4f}, f1={f1:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    print(f"  Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Mean F1: {mean_f1:.4f} +/- {std_f1:.4f}")

    return {
        "feature_set": feature_name,
        "k_folds": k_folds,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
        "min_accuracy": float(min(fold_accuracies)),
        "max_accuracy": float(max(fold_accuracies)),
        "fold_details": fold_details,
    }


# =============================================================================
# EXPERIMENT 2: Bootstrap Confidence Intervals
# =============================================================================


def run_bootstrap_ci(
    X: np.ndarray,
    y: np.ndarray,
    svm_cfg: Dict,
    n_bootstrap: int,
    seed: int,
    feature_name: str,
    test_ratio: float = 0.2,
) -> Dict:
    """
    Bootstrap confidence intervals for accuracy.
    1. Split data into train/test
    2. Train model on train set
    3. Resample test predictions n_bootstrap times
    4. Compute 95% CI
    """
    print(f"\n  Running bootstrap CI for {feature_name} ({n_bootstrap} iterations)...")

    np.random.seed(seed)

    # Split data (80/20)
    n = len(y)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
        ),
    )
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    point_estimate = accuracy_score(y_test, y_pred)

    # Bootstrap
    bootstrap_accs = []
    n_test = len(y_test)

    for _ in range(n_bootstrap):
        # Resample test indices with replacement
        boot_idx = np.random.choice(n_test, size=n_test, replace=True)
        boot_acc = accuracy_score(y_test[boot_idx], y_pred[boot_idx])
        bootstrap_accs.append(boot_acc)

    bootstrap_accs = np.array(bootstrap_accs)

    # Compute percentile CI
    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)
    ci_width = ci_upper - ci_lower

    # Standard error
    se = np.std(bootstrap_accs)

    print(f"    Point estimate: {point_estimate:.4f}")
    print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    CI width: {ci_width:.4f}")
    print(f"    Bootstrap SE: {se:.4f}")

    return {
        "feature_set": feature_name,
        "n_bootstrap": n_bootstrap,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "point_estimate": float(point_estimate),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "ci_width": float(ci_width),
        "bootstrap_se": float(se),
        "bootstrap_mean": float(np.mean(bootstrap_accs)),
        "bootstrap_std": float(np.std(bootstrap_accs)),
    }


# =============================================================================
# EXPERIMENT 3: Paired Bootstrap Test (Coverage vs Combined)
# =============================================================================


def run_paired_bootstrap_test(
    X_cov: np.ndarray,
    X_combined: np.ndarray,
    y: np.ndarray,
    svm_cfg: Dict,
    n_bootstrap: int,
    seed: int,
    test_ratio: float = 0.2,
) -> Dict:
    """
    Paired bootstrap test to check if combined significantly beats coverage.
    """
    print(f"\n  Running paired bootstrap test ({n_bootstrap} iterations)...")

    np.random.seed(seed)

    # Split data (80/20)
    n = len(y)
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_cov_train, X_cov_test = X_cov[train_idx], X_cov[test_idx]
    X_comb_train, X_comb_test = X_combined[train_idx], X_combined[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train both models
    model_cov = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
        ),
    )
    model_cov.fit(X_cov_train, y_train)

    model_comb = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
        ),
    )
    model_comb.fit(X_comb_train, y_train)

    # Get predictions
    pred_cov = model_cov.predict(X_cov_test)
    pred_comb = model_comb.predict(X_comb_test)

    acc_cov = accuracy_score(y_test, pred_cov)
    acc_comb = accuracy_score(y_test, pred_comb)
    observed_diff = acc_comb - acc_cov

    print(f"    Coverage accuracy: {acc_cov:.4f}")
    print(f"    Combined accuracy: {acc_comb:.4f}")
    print(f"    Observed difference: {observed_diff:+.4f}")

    # Paired bootstrap
    n_test = len(y_test)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n_test, size=n_test, replace=True)
        boot_acc_cov = accuracy_score(y_test[boot_idx], pred_cov[boot_idx])
        boot_acc_comb = accuracy_score(y_test[boot_idx], pred_comb[boot_idx])
        bootstrap_diffs.append(boot_acc_comb - boot_acc_cov)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # CI for difference
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    # P-value: proportion of bootstrap samples where diff <= 0 (one-sided test)
    p_value = np.mean(bootstrap_diffs <= 0)

    # Is combined significantly better?
    significant = ci_lower > 0  # 95% CI doesn't include 0

    print(f"    95% CI for difference: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"    P-value (one-sided): {p_value:.4f}")
    print(f"    Significant at alpha=0.05: {significant}")

    return {
        "coverage_accuracy": float(acc_cov),
        "combined_accuracy": float(acc_comb),
        "observed_difference": float(observed_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "p_value_one_sided": float(p_value),
        "significant_at_alpha_05": bool(significant),
        "n_bootstrap": n_bootstrap,
    }


# =============================================================================
# EXPERIMENT 4: Sample Size Analysis
# =============================================================================


def run_sample_size_analysis(
    X: np.ndarray,
    y: np.ndarray,
    svm_cfg: Dict,
    seed: int,
    feature_name: str,
) -> Dict:
    """
    Analyze how performance varies with training set size.
    Learning curve analysis.
    """
    print(f"\n  Running sample size analysis for {feature_name}...")

    np.random.seed(seed)

    # Test set: 20%
    n = len(y)
    indices = np.random.permutation(n)
    split_idx = int(n * 0.8)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    # Try different training sizes
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for frac in train_sizes:
        n_train = int(len(y_train_full) * frac)
        if n_train < 10:
            continue

        # Sample training data
        train_subset_idx = np.random.choice(
            len(y_train_full), size=n_train, replace=False
        )
        X_train = X_train_full[train_subset_idx]
        y_train = y_train_full[train_subset_idx]

        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel=svm_cfg["kernel"],
                C=svm_cfg["C"],
                gamma=svm_cfg["gamma"],
                degree=svm_cfg["degree"],
            ),
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        results.append(
            {
                "train_fraction": frac,
                "train_size": n_train,
                "accuracy": float(acc),
                "f1": float(f1),
            }
        )

        print(f"    {frac * 100:.0f}% ({n_train} samples): acc={acc:.4f}")

    return {
        "feature_set": feature_name,
        "test_size": len(test_idx),
        "learning_curve": results,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()

    svm_cfg = {
        "kernel": args.svm_kernel,
        "C": args.svm_c,
        "gamma": args.svm_gamma,
        "degree": args.svm_degree,
    }

    run_log = init_run_logging(
        script_subdir="experimental-design",
        hyperparams={
            "facts": args.facts,
            "out": args.out,
            "model_dir": args.model_dir,
            "svm": svm_cfg,
            "k_folds": args.k_folds,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
    )

    print("=" * 70)
    print("Option D: Experimental Design Improvements")
    print("=" * 70)

    # Load data
    print("\nLoading facts...")
    facts = load_json(args.facts)
    print(f"Loaded {len(facts)} fact triplets")

    os.makedirs(args.model_dir, exist_ok=True)

    # Prepare datasets with different feature sets
    print("\nPreparing feature sets...")

    X_cov, y_cov, dim_cov = prepare_dataset(facts, build_coverage_features)
    print(f"  Coverage: {X_cov.shape[0]} samples, {dim_cov} dims")

    X_str, y_str, dim_str = prepare_dataset(facts, build_structural_features)
    print(f"  Structural: {X_str.shape[0]} samples, {dim_str} dims")

    X_comb, y_comb, dim_comb = prepare_dataset(facts, build_combined_features)
    print(f"  Combined: {X_comb.shape[0]} samples, {dim_comb} dims")

    all_results = {}

    # =================================================================
    # Experiment 1: K-Fold Cross-Validation
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: K-Fold Cross-Validation")
    print("=" * 70)

    cv_results = {}
    cv_results["coverage"] = run_kfold_cv(
        X_cov, y_cov, args.k_folds, svm_cfg, args.seed, "coverage"
    )
    cv_results["structural"] = run_kfold_cv(
        X_str, y_str, args.k_folds, svm_cfg, args.seed, "structural"
    )
    cv_results["combined"] = run_kfold_cv(
        X_comb, y_comb, args.k_folds, svm_cfg, args.seed, "combined"
    )

    all_results["cross_validation"] = cv_results

    # =================================================================
    # Experiment 2: Bootstrap Confidence Intervals
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Bootstrap Confidence Intervals")
    print("=" * 70)

    bootstrap_results = {}
    bootstrap_results["coverage"] = run_bootstrap_ci(
        X_cov, y_cov, svm_cfg, args.n_bootstrap, args.seed, "coverage"
    )
    bootstrap_results["structural"] = run_bootstrap_ci(
        X_str, y_str, svm_cfg, args.n_bootstrap, args.seed, "structural"
    )
    bootstrap_results["combined"] = run_bootstrap_ci(
        X_comb, y_comb, svm_cfg, args.n_bootstrap, args.seed, "combined"
    )

    all_results["bootstrap_ci"] = bootstrap_results

    # =================================================================
    # Experiment 3: Paired Bootstrap Test
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Paired Bootstrap Test (Combined vs Coverage)")
    print("=" * 70)

    paired_test = run_paired_bootstrap_test(
        X_cov, X_comb, y_cov, svm_cfg, args.n_bootstrap, args.seed
    )
    all_results["paired_bootstrap_test"] = paired_test

    # =================================================================
    # Experiment 4: Sample Size Analysis
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Sample Size Analysis (Learning Curves)")
    print("=" * 70)

    size_analysis = {}
    size_analysis["coverage"] = run_sample_size_analysis(
        X_cov, y_cov, svm_cfg, args.seed, "coverage"
    )
    size_analysis["combined"] = run_sample_size_analysis(
        X_comb, y_comb, svm_cfg, args.seed, "combined"
    )

    all_results["sample_size_analysis"] = size_analysis

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Option D Results")
    print("=" * 70)

    print("\n1. K-Fold Cross-Validation (k={})".format(args.k_folds))
    print(f"   {'Feature Set':<15} {'Mean Acc':<12} {'Std':<10} {'Range':<20}")
    print("   " + "-" * 57)
    for name, res in cv_results.items():
        range_str = f"[{res['min_accuracy']:.4f}, {res['max_accuracy']:.4f}]"
        print(
            f"   {name:<15} {res['mean_accuracy']:<12.4f} {res['std_accuracy']:<10.4f} {range_str}"
        )

    print("\n2. Bootstrap 95% Confidence Intervals")
    print(f"   {'Feature Set':<15} {'Point Est':<12} {'95% CI':<25} {'Width':<10}")
    print("   " + "-" * 62)
    for name, res in bootstrap_results.items():
        ci_str = f"[{res['ci_95_lower']:.4f}, {res['ci_95_upper']:.4f}]"
        print(
            f"   {name:<15} {res['point_estimate']:<12.4f} {ci_str:<25} {res['ci_width']:<10.4f}"
        )

    print("\n3. Paired Bootstrap Test (Combined vs Coverage)")
    print(f"   Observed difference: {paired_test['observed_difference']:+.4f}")
    print(
        f"   95% CI: [{paired_test['ci_95_lower']:+.4f}, {paired_test['ci_95_upper']:+.4f}]"
    )
    print(f"   Significant at alpha=0.05: {paired_test['significant_at_alpha_05']}")

    # Reference baselines
    baselines = {
        "fixed_split_coverage": 0.7528,
        "fixed_split_combined": 0.7727,
        "fixed_split_log_depth": 0.7642,
    }

    # Save results
    output = {
        "setup": {
            "goal": "Experimental design improvements: cross-validation and bootstrap CI (Option D)",
            "facts": args.facts,
            "svm": svm_cfg,
            "k_folds": args.k_folds,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "total_samples": len(y_cov),
            "created": datetime.now().isoformat(),
        },
        "reference_baselines": baselines,
        "experiments": all_results,
    }

    save_json(args.out, output)
    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
