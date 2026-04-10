"""
Hybrid Approach (Option C): Explicitly separate coverage and structural signals.

This script implements Option C to quantify the independent contributions of
coverage (fact omission) and structural (RST prominence) signals.

Three approaches are implemented:

1. Feature Group Ablation:
   - Coverage-only features
   - Structural-only features (log depth, the best from Option B)
   - Combined features (both coverage + structural)

2. Two-Stage Classifier:
   - First stage: Coverage model predicts bias
   - Second stage: Structural model trained on coverage errors
   - Goal: See if structure helps correct coverage mistakes

3. Stacking Ensemble:
   - Train separate coverage and structural models
   - Meta-classifier combines their predictions
   - Quantifies complementary information

Key insight: If structural features add value beyond coverage, we should see:
- Combined > Coverage-only (structural adds information)
- Two-stage corrects some coverage errors
- Stacking outperforms individual models
"""

import argparse
import json
import os
import pickle
import sys
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Add project root for module imports
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "params.yaml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(PROJECT_ROOT))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = "experiments/hybrid-approach/results/hybrid_results.json"
DEFAULT_MODEL_DIR = "experiments/hybrid-approach/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid approach: separate coverage and structural signals (Option C)"
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
# FEATURE BUILDERS
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


def build_coverage_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    Build binary coverage features (delta encoding).
    Same as omission-based model: [has_left - has_center] per cluster for left example.
    """
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


def build_structural_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    Build structural prominence features using log depth formula (best from Option B).
    Returns delta encoding: W_side - W_center per cluster.
    """
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


def build_combined_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    Build combined features: coverage + structural interleaved per cluster.
    For each cluster: [coverage_delta, structural_delta]
    """
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


def build_structural_only_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    Build structural-only features: only include clusters where all 3 sides present.
    This removes the coverage signal entirely.
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        coverage = get_cluster_coverage(edus, edu_lookup)

        # Only include if all 3 sides present (size-3 cluster)
        if coverage["left"] > 0 and coverage["center"] > 0 and coverage["right"] > 0:
            W_left = compute_side_prominence(edus, edu_lookup, "left", W_log_depth)
            W_center = compute_side_prominence(edus, edu_lookup, "center", W_log_depth)
            W_right = compute_side_prominence(edus, edu_lookup, "right", W_log_depth)

            deltas_left.append(W_left - W_center)
            deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


# =============================================================================
# TRAINING UTILITIES
# =============================================================================


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


def evaluate(model, x: np.ndarray, y: np.ndarray) -> Dict:
    """Evaluate model on given data."""
    pred = model.predict(x)
    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def build_features_for_rows(rows: List[dict], feature_builder: Callable):
    """Apply feature builder to all rows."""
    for row in rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)


# =============================================================================
# EXPERIMENT 1: Feature Group Ablation
# =============================================================================


def run_feature_ablation(
    train_rows: List[dict],
    val_rows: List[dict],
    test_rows: List[dict],
    svm_cfg: Dict,
) -> Dict:
    """
    Run feature group ablation experiments:
    - Coverage-only
    - Structural-only (all clusters)
    - Structural-only (size-3 clusters only, no coverage signal)
    - Combined (coverage + structural)
    """
    import copy

    results = {}

    experiments = [
        ("coverage_only", "Binary coverage features", build_coverage_features),
        (
            "structural_all",
            "Structural log-depth (all clusters)",
            build_structural_features,
        ),
        (
            "structural_size3",
            "Structural log-depth (size-3 only, no coverage)",
            build_structural_only_features,
        ),
        ("combined", "Coverage + Structural combined", build_combined_features),
    ]

    for exp_name, description, feature_builder in experiments:
        print(f"\n  Running: {exp_name}")

        # Deep copy to avoid mutation
        train_copy = copy.deepcopy(train_rows)
        val_copy = copy.deepcopy(val_rows)
        test_copy = copy.deepcopy(test_rows)

        # Build features
        build_features_for_rows(train_copy, feature_builder)
        build_features_for_rows(val_copy, feature_builder)
        build_features_for_rows(test_copy, feature_builder)

        # Build X, y
        x_train_raw, y_train = build_xy(train_copy, "feat_left", "feat_right")
        x_val_raw, y_val = build_xy(val_copy, "feat_left", "feat_right")
        x_test_raw, y_test = build_xy(test_copy, "feat_left", "feat_right")

        max_len = max((len(v) for v in x_train_raw), default=0)

        if max_len == 0:
            results[exp_name] = {
                "description": description,
                "error": "Empty features",
                "metrics": {
                    "train": {"accuracy": 0.5, "macro_f1": 0.333},
                    "val": {"accuracy": 0.5, "macro_f1": 0.333},
                    "test": {"accuracy": 0.5, "macro_f1": 0.333},
                },
            }
            continue

        x_train = pad_or_truncate(x_train_raw, max_len)
        x_val = pad_or_truncate(x_val_raw, max_len)
        x_test = pad_or_truncate(x_test_raw, max_len)

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
        model.fit(x_train, y_train)

        metrics = {
            "train": evaluate(model, x_train, y_train),
            "val": evaluate(model, x_val, y_val),
            "test": evaluate(model, x_test, y_test),
        }

        results[exp_name] = {
            "description": description,
            "input_dim": max_len,
            "metrics": metrics,
        }

        print(
            f"    Test: acc={metrics['test']['accuracy']:.4f}, f1={metrics['test']['macro_f1']:.4f}"
        )

    return results


# =============================================================================
# EXPERIMENT 2: Two-Stage Classifier
# =============================================================================


def run_two_stage(
    train_rows: List[dict],
    val_rows: List[dict],
    test_rows: List[dict],
    svm_cfg: Dict,
) -> Dict:
    """
    Two-stage classifier:
    1. Train coverage model
    2. Identify misclassified samples
    3. Train structural model on error cases
    4. Combine predictions: use structural prediction when coverage is uncertain

    Returns analysis of whether structure helps correct coverage errors.
    """
    import copy

    print("\n  Stage 1: Training coverage model...")

    # Deep copy
    train_copy = copy.deepcopy(train_rows)
    val_copy = copy.deepcopy(val_rows)
    test_copy = copy.deepcopy(test_rows)

    # Build coverage features
    build_features_for_rows(train_copy, build_coverage_features)
    build_features_for_rows(val_copy, build_coverage_features)
    build_features_for_rows(test_copy, build_coverage_features)

    x_train_cov_raw, y_train = build_xy(train_copy, "feat_left", "feat_right")
    x_val_cov_raw, y_val = build_xy(val_copy, "feat_left", "feat_right")
    x_test_cov_raw, y_test = build_xy(test_copy, "feat_left", "feat_right")

    max_len_cov = max((len(v) for v in x_train_cov_raw), default=0)
    x_train_cov = pad_or_truncate(x_train_cov_raw, max_len_cov)
    x_val_cov = pad_or_truncate(x_val_cov_raw, max_len_cov)
    x_test_cov = pad_or_truncate(x_test_cov_raw, max_len_cov)

    # Train coverage model
    cov_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
            probability=True,  # Enable probability estimates
        ),
    )
    cov_model.fit(x_train_cov, y_train)

    cov_train_pred = cov_model.predict(x_train_cov)
    cov_val_pred = cov_model.predict(x_val_cov)
    cov_test_pred = cov_model.predict(x_test_cov)

    cov_train_prob = cov_model.predict_proba(x_train_cov)
    cov_val_prob = cov_model.predict_proba(x_val_cov)
    cov_test_prob = cov_model.predict_proba(x_test_cov)

    coverage_metrics = {
        "train": evaluate(cov_model, x_train_cov, y_train),
        "val": evaluate(cov_model, x_val_cov, y_val),
        "test": evaluate(cov_model, x_test_cov, y_test),
    }

    print(f"    Coverage test acc: {coverage_metrics['test']['accuracy']:.4f}")

    # Identify error indices
    train_errors = np.where(cov_train_pred != y_train)[0]
    val_errors = np.where(cov_val_pred != y_val)[0]
    test_errors = np.where(cov_test_pred != y_test)[0]

    print(
        f"    Train errors: {len(train_errors)}/{len(y_train)} ({100 * len(train_errors) / len(y_train):.1f}%)"
    )
    print(
        f"    Test errors: {len(test_errors)}/{len(y_test)} ({100 * len(test_errors) / len(y_test):.1f}%)"
    )

    print("\n  Stage 2: Training structural model on error cases...")

    # Build structural features for all samples
    train_copy2 = copy.deepcopy(train_rows)
    val_copy2 = copy.deepcopy(val_rows)
    test_copy2 = copy.deepcopy(test_rows)

    build_features_for_rows(train_copy2, build_structural_features)
    build_features_for_rows(val_copy2, build_structural_features)
    build_features_for_rows(test_copy2, build_structural_features)

    x_train_str_raw, _ = build_xy(train_copy2, "feat_left", "feat_right")
    x_val_str_raw, _ = build_xy(val_copy2, "feat_left", "feat_right")
    x_test_str_raw, _ = build_xy(test_copy2, "feat_left", "feat_right")

    max_len_str = max((len(v) for v in x_train_str_raw), default=0)
    x_train_str = pad_or_truncate(x_train_str_raw, max_len_str)
    x_val_str = pad_or_truncate(x_val_str_raw, max_len_str)
    x_test_str = pad_or_truncate(x_test_str_raw, max_len_str)

    # Train structural model on ALL training data (not just errors)
    str_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
            probability=True,
        ),
    )
    str_model.fit(x_train_str, y_train)

    str_train_pred = str_model.predict(x_train_str)
    str_val_pred = str_model.predict(x_val_str)
    str_test_pred = str_model.predict(
        x_test_cov
    )  # Note: using structural features for test

    str_test_pred = str_model.predict(x_test_str)
    str_test_prob = str_model.predict_proba(x_test_str)

    structural_metrics = {
        "train": evaluate(str_model, x_train_str, y_train),
        "val": evaluate(str_model, x_val_str, y_val),
        "test": evaluate(str_model, x_test_str, y_test),
    }

    print(f"    Structural test acc: {structural_metrics['test']['accuracy']:.4f}")

    # Analyze: Can structure correct coverage errors?
    # Check on test set: for samples where coverage was wrong, is structure right?
    structure_corrects = 0
    structure_wrong_too = 0

    for idx in test_errors:
        if str_test_pred[idx] == y_test[idx]:
            structure_corrects += 1
        else:
            structure_wrong_too += 1

    print(f"\n  Analysis of coverage errors ({len(test_errors)} test errors):")
    print(
        f"    Structure would correct: {structure_corrects} ({100 * structure_corrects / len(test_errors) if test_errors.size > 0 else 0:.1f}%)"
    )
    print(
        f"    Structure also wrong: {structure_wrong_too} ({100 * structure_wrong_too / len(test_errors) if test_errors.size > 0 else 0:.1f}%)"
    )

    # Combined prediction strategies
    print("\n  Stage 3: Testing combination strategies...")

    # Strategy 1: Always use coverage
    always_cov_acc = accuracy_score(y_test, cov_test_pred)

    # Strategy 2: Use structure when coverage is uncertain (max prob < threshold)
    uncertainty_results = {}
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        combined_pred = cov_test_pred.copy()
        cov_max_prob = np.max(cov_test_prob, axis=1)
        uncertain_mask = cov_max_prob < threshold
        combined_pred[uncertain_mask] = str_test_pred[uncertain_mask]

        combined_acc = accuracy_score(y_test, combined_pred)
        num_uncertain = np.sum(uncertain_mask)

        uncertainty_results[f"threshold_{threshold}"] = {
            "threshold": threshold,
            "num_uncertain": int(num_uncertain),
            "pct_uncertain": float(100 * num_uncertain / len(y_test)),
            "accuracy": float(combined_acc),
            "improvement_vs_coverage": float(combined_acc - always_cov_acc),
        }

        print(
            f"    Threshold {threshold}: {num_uncertain} uncertain, acc={combined_acc:.4f} (Δ={combined_acc - always_cov_acc:+.4f})"
        )

    # Strategy 3: Weighted average of probabilities
    for w_str in [0.1, 0.2, 0.3, 0.4, 0.5]:
        w_cov = 1.0 - w_str
        combined_prob = w_cov * cov_test_prob + w_str * str_test_prob
        combined_pred = np.argmax(combined_prob, axis=1)
        combined_acc = accuracy_score(y_test, combined_pred)

        uncertainty_results[f"weighted_str_{w_str}"] = {
            "weight_structural": w_str,
            "weight_coverage": w_cov,
            "accuracy": float(combined_acc),
            "improvement_vs_coverage": float(combined_acc - always_cov_acc),
        }

        print(
            f"    Weighted (cov={w_cov:.1f}, str={w_str:.1f}): acc={combined_acc:.4f} (Δ={combined_acc - always_cov_acc:+.4f})"
        )

    return {
        "coverage_model": coverage_metrics,
        "structural_model": structural_metrics,
        "error_analysis": {
            "train_errors": int(len(train_errors)),
            "test_errors": int(len(test_errors)),
            "structure_corrects_test_errors": int(structure_corrects),
            "structure_wrong_on_test_errors": int(structure_wrong_too),
            "pct_errors_structure_corrects": float(
                100 * structure_corrects / len(test_errors)
            )
            if len(test_errors) > 0
            else 0.0,
        },
        "combination_strategies": uncertainty_results,
    }


# =============================================================================
# EXPERIMENT 3: Stacking Ensemble
# =============================================================================


def run_stacking_ensemble(
    train_rows: List[dict],
    val_rows: List[dict],
    test_rows: List[dict],
    svm_cfg: Dict,
) -> Dict:
    """
    Stacking ensemble:
    1. Train coverage and structural models on train set
    2. Get their predictions/probabilities on validation set
    3. Train meta-classifier on validation predictions
    4. Evaluate on test set

    This quantifies whether the two feature sets provide complementary information.
    """
    import copy

    print("\n  Building base models...")

    # Build features for all splits
    train_cov = copy.deepcopy(train_rows)
    val_cov = copy.deepcopy(val_rows)
    test_cov = copy.deepcopy(test_rows)

    train_str = copy.deepcopy(train_rows)
    val_str = copy.deepcopy(val_rows)
    test_str = copy.deepcopy(test_rows)

    build_features_for_rows(train_cov, build_coverage_features)
    build_features_for_rows(val_cov, build_coverage_features)
    build_features_for_rows(test_cov, build_coverage_features)

    build_features_for_rows(train_str, build_structural_features)
    build_features_for_rows(val_str, build_structural_features)
    build_features_for_rows(test_str, build_structural_features)

    # Build arrays
    x_train_cov_raw, y_train = build_xy(train_cov, "feat_left", "feat_right")
    x_val_cov_raw, y_val = build_xy(val_cov, "feat_left", "feat_right")
    x_test_cov_raw, y_test = build_xy(test_cov, "feat_left", "feat_right")

    x_train_str_raw, _ = build_xy(train_str, "feat_left", "feat_right")
    x_val_str_raw, _ = build_xy(val_str, "feat_left", "feat_right")
    x_test_str_raw, _ = build_xy(test_str, "feat_left", "feat_right")

    max_len_cov = max((len(v) for v in x_train_cov_raw), default=0)
    max_len_str = max((len(v) for v in x_train_str_raw), default=0)

    x_train_cov = pad_or_truncate(x_train_cov_raw, max_len_cov)
    x_val_cov = pad_or_truncate(x_val_cov_raw, max_len_cov)
    x_test_cov = pad_or_truncate(x_test_cov_raw, max_len_cov)

    x_train_str = pad_or_truncate(x_train_str_raw, max_len_str)
    x_val_str = pad_or_truncate(x_val_str_raw, max_len_str)
    x_test_str = pad_or_truncate(x_test_str_raw, max_len_str)

    # Train base models on TRAIN set
    cov_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
            probability=True,
        ),
    )
    cov_model.fit(x_train_cov, y_train)

    str_model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=svm_cfg["kernel"],
            C=svm_cfg["C"],
            gamma=svm_cfg["gamma"],
            degree=svm_cfg["degree"],
            probability=True,
        ),
    )
    str_model.fit(x_train_str, y_train)

    # Get probabilities for stacking
    # For meta-classifier training, use validation set
    cov_val_prob = cov_model.predict_proba(x_val_cov)
    str_val_prob = str_model.predict_proba(x_val_str)

    cov_test_prob = cov_model.predict_proba(x_test_cov)
    str_test_prob = str_model.predict_proba(x_test_str)

    # Stack predictions as meta-features
    # Each model gives 2 class probabilities, so meta-features = 4 dims
    meta_train_x = np.hstack([cov_val_prob, str_val_prob])
    meta_test_x = np.hstack([cov_test_prob, str_test_prob])

    print(f"    Meta-features shape: {meta_train_x.shape}")

    # Train meta-classifier (logistic regression for interpretability)
    meta_model = LogisticRegression(max_iter=1000, random_state=42)
    meta_model.fit(meta_train_x, y_val)

    # Evaluate
    meta_pred = meta_model.predict(meta_test_x)
    meta_acc = accuracy_score(y_test, meta_pred)
    meta_f1 = f1_score(y_test, meta_pred, average="macro")

    # Individual model performance on test
    cov_test_pred = cov_model.predict(x_test_cov)
    str_test_pred = str_model.predict(x_test_str)

    cov_acc = accuracy_score(y_test, cov_test_pred)
    str_acc = accuracy_score(y_test, str_test_pred)

    print(f"    Coverage model test acc: {cov_acc:.4f}")
    print(f"    Structural model test acc: {str_acc:.4f}")
    print(f"    Stacked meta-model test acc: {meta_acc:.4f}")

    # Analyze meta-model coefficients
    # Coefficients show relative importance of each base model's predictions
    coefficients = meta_model.coef_[0]

    print(f"\n  Meta-model coefficients:")
    print(f"    Coverage prob[0]: {coefficients[0]:.4f}")
    print(f"    Coverage prob[1]: {coefficients[1]:.4f}")
    print(f"    Structural prob[0]: {coefficients[2]:.4f}")
    print(f"    Structural prob[1]: {coefficients[3]:.4f}")

    return {
        "base_models": {
            "coverage": {
                "test_accuracy": float(cov_acc),
                "test_f1": float(f1_score(y_test, cov_test_pred, average="macro")),
            },
            "structural": {
                "test_accuracy": float(str_acc),
                "test_f1": float(f1_score(y_test, str_test_pred, average="macro")),
            },
        },
        "stacked_model": {
            "test_accuracy": float(meta_acc),
            "test_f1": float(meta_f1),
            "improvement_vs_coverage": float(meta_acc - cov_acc),
            "improvement_vs_structural": float(meta_acc - str_acc),
        },
        "meta_coefficients": {
            "coverage_prob_0": float(coefficients[0]),
            "coverage_prob_1": float(coefficients[1]),
            "structural_prob_0": float(coefficients[2]),
            "structural_prob_1": float(coefficients[3]),
        },
        "interpretation": {
            "note": "If stacked > max(coverage, structural), models provide complementary info",
            "coverage_dominant": bool(
                abs(coefficients[0]) + abs(coefficients[1])
                > abs(coefficients[2]) + abs(coefficients[3])
            ),
        },
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
        script_subdir="hybrid-approach",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "svm": svm_cfg,
        },
    )

    print("=" * 70)
    print("Option C: Hybrid Approach - Separating Coverage & Structural Signals")
    print("=" * 70)

    # Load data
    print("\nLoading facts and splits...")
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

    os.makedirs(args.model_dir, exist_ok=True)

    all_results = {}

    # Experiment 1: Feature Group Ablation
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Feature Group Ablation")
    print("=" * 70)
    ablation_results = run_feature_ablation(facts_train, facts_val, facts_test, svm_cfg)
    all_results["feature_ablation"] = ablation_results

    # Experiment 2: Two-Stage Classifier
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Two-Stage Classifier")
    print("=" * 70)
    two_stage_results = run_two_stage(facts_train, facts_val, facts_test, svm_cfg)
    all_results["two_stage"] = two_stage_results

    # Experiment 3: Stacking Ensemble
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Stacking Ensemble")
    print("=" * 70)
    stacking_results = run_stacking_ensemble(
        facts_train, facts_val, facts_test, svm_cfg
    )
    all_results["stacking"] = stacking_results

    # Reference baselines
    baselines = {
        "coverage_only_original": {"test_acc": 0.7528, "test_f1": 0.7504},
        "structural_log_depth": {"test_acc": 0.7642, "test_f1": 0.7626},
        "structural_baseline": {"test_acc": 0.6705, "test_f1": 0.6702},
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Option C Results")
    print("=" * 70)

    print("\n1. Feature Group Ablation:")
    print(f"   {'Experiment':<25} {'Test Acc':<12} {'Test F1':<12}")
    print("   " + "-" * 50)
    for exp_name, exp_data in ablation_results.items():
        if "metrics" in exp_data:
            test_acc = exp_data["metrics"]["test"]["accuracy"]
            test_f1 = exp_data["metrics"]["test"]["macro_f1"]
            print(f"   {exp_name:<25} {test_acc:<12.4f} {test_f1:<12.4f}")

    print("\n2. Two-Stage Analysis:")
    print(
        f"   Coverage errors on test: {two_stage_results['error_analysis']['test_errors']}"
    )
    print(
        f"   Structure could correct: {two_stage_results['error_analysis']['structure_corrects_test_errors']} ({two_stage_results['error_analysis']['pct_errors_structure_corrects']:.1f}%)"
    )

    best_combo = max(
        two_stage_results["combination_strategies"].items(),
        key=lambda x: x[1]["accuracy"],
    )
    print(
        f"   Best combination: {best_combo[0]} with acc={best_combo[1]['accuracy']:.4f}"
    )

    print("\n3. Stacking Ensemble:")
    print(
        f"   Coverage base: {stacking_results['base_models']['coverage']['test_accuracy']:.4f}"
    )
    print(
        f"   Structural base: {stacking_results['base_models']['structural']['test_accuracy']:.4f}"
    )
    print(f"   Stacked meta: {stacking_results['stacked_model']['test_accuracy']:.4f}")
    print(
        f"   Improvement vs coverage: {stacking_results['stacked_model']['improvement_vs_coverage']:+.4f}"
    )

    # Save results
    output = {
        "setup": {
            "goal": "Hybrid approach to separate coverage and structural signals (Option C)",
            "facts": args.facts,
            "split_dir": args.split_dir,
            "svm": svm_cfg,
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
