"""
RST-Only Experiment: Isolating RST Contribution from Omission

This experiment filters the dataset to only include clusters where ALL three
bias sides (left, center, right) have coverage. By doing this, we eliminate
fact omission as a signal and can measure the pure RST structural contribution
to bias prediction.

Key insight: If omission is the primary signal (not RST positioning), then
filtering to only non-omitted facts should dramatically reduce accuracy.

Experimental design:
1. Filter clusters to only those with all 3 bias sides present
2. Filter triplets to only those with at least N full-coverage clusters
3. Run the same DFI approaches on this filtered dataset
4. Compare with full dataset results to quantify RST vs omission contribution

Reference baselines (full dataset, with omission signal):
- Ordered bipartite coverage: ~88-90%
- This will tell us how much of that accuracy comes from RST vs omission
"""

import argparse
import copy
import json
import math
import os
import pickle
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Add parent directory for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = "rst-vs-omission/results/rst_only_results.json"
DEFAULT_MODEL_DIR = "rst-vs-omission/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test RST-only signal by filtering to full-coverage clusters"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=1,
        help="Minimum number of full-coverage clusters required per triplet",
    )
    parser.add_argument(
        "--depth-agg",
        choices=["min", "max", "avg", "sum"],
        default="min",
        help="How to aggregate normalized depths within a cluster",
    )

    # Random Forest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
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


# =============================================================================
# FILTERING: Only clusters with ALL 3 bias sides
# =============================================================================


def get_cluster_coverage(edus: List[str], edu_lookup: Dict) -> Set[str]:
    """Get which bias sides are present in a cluster."""
    sides_present = set()
    for edu_id in edus:
        meta = edu_lookup.get(edu_id, {})
        bias = meta.get("bias", "")
        if bias in VALID_BIASES:
            sides_present.add(bias)
    return sides_present


def filter_to_full_coverage_clusters(fact_row: Dict) -> Dict:
    """
    Filter a fact row to only include clusters where all 3 bias sides are present.
    Returns a modified copy of the fact row with filtered clusters and edu_lookup.
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    filtered_clusters = {}
    filtered_edu_ids = set()

    for cluster_id, edus in clusters.items():
        coverage = get_cluster_coverage(edus, edu_lookup)

        # Only keep clusters with all 3 sides
        if coverage == VALID_BIASES:
            filtered_clusters[cluster_id] = edus
            filtered_edu_ids.update(edus)

    # Filter edu_lookup to only include EDUs from filtered clusters
    filtered_edu_lookup = {
        edu_id: meta
        for edu_id, meta in edu_lookup.items()
        if edu_id in filtered_edu_ids
    }

    # Create modified row
    filtered_row = copy.deepcopy(fact_row)
    filtered_row["clusters"] = filtered_clusters
    filtered_row["edu_lookup"] = filtered_edu_lookup
    filtered_row["original_cluster_count"] = len(clusters)
    filtered_row["filtered_cluster_count"] = len(filtered_clusters)

    return filtered_row


def filter_dataset(
    facts_rows: List[dict], min_clusters: int = 1
) -> Tuple[List[dict], Dict]:
    """
    Filter dataset to only include triplets with at least min_clusters
    full-coverage clusters.

    Returns:
        filtered_rows: List of filtered fact rows
        stats: Dictionary with filtering statistics
    """
    filtered_rows = []

    total_original_clusters = 0
    total_filtered_clusters = 0

    for row in facts_rows:
        filtered_row = filter_to_full_coverage_clusters(row)

        total_original_clusters += filtered_row["original_cluster_count"]
        total_filtered_clusters += filtered_row["filtered_cluster_count"]

        # Only keep triplets with at least min_clusters full-coverage clusters
        if filtered_row["filtered_cluster_count"] >= min_clusters:
            filtered_rows.append(filtered_row)

    stats = {
        "original_triplets": len(facts_rows),
        "filtered_triplets": len(filtered_rows),
        "triplet_retention_rate": len(filtered_rows) / len(facts_rows)
        if facts_rows
        else 0,
        "original_clusters": total_original_clusters,
        "filtered_clusters": total_filtered_clusters,
        "cluster_retention_rate": total_filtered_clusters / total_original_clusters
        if total_original_clusters
        else 0,
        "min_clusters_required": min_clusters,
    }

    return filtered_rows, stats


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
    """Best-performing prominence formula: 1 / (1 + log(1 + depth))"""
    return 1.0 / (1.0 + math.log1p(depth))


def get_edu_prominence(edu_meta: Dict) -> float:
    """Compute prominence score for a single EDU."""
    if not edu_meta:
        return 0.0
    depth = edu_meta.get("depth", 0)
    sat_count = edu_meta.get("satellite_edges_to_root", 0)
    return W_log_depth(depth, sat_count)


def get_side_edus(edus: List[str], edu_lookup: Dict, side: str) -> List[Dict]:
    """Get all EDU metadata for a given side in a cluster."""
    result = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            result.append(meta)
    return result


def get_side_scores(edus: List[str], edu_lookup: Dict, side: str) -> List[float]:
    """Get prominence scores for all EDUs of a given side in a cluster."""
    scores = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            scores.append(get_edu_prominence(meta))
    return scores


# =============================================================================
# ORDERING STRATEGY: By Normalized Depth
# =============================================================================


def compute_max_depths_per_article(edu_lookup: Dict) -> Dict[str, int]:
    """Compute the maximum depth for each article (bias)."""
    max_depths = {"left": 0, "center": 0, "right": 0}
    for edu_id, meta in edu_lookup.items():
        bias = meta.get("bias", "")
        depth = meta.get("depth", 0)
        if bias in max_depths:
            max_depths[bias] = max(max_depths[bias], depth)

    # Ensure no division by zero
    for bias in max_depths:
        if max_depths[bias] == 0:
            max_depths[bias] = 1

    return max_depths


def get_normalized_depth(edu_meta: Dict, max_depths: Dict[str, int]) -> float:
    """Get normalized depth for an EDU: depth / max_depth_for_article."""
    if not edu_meta:
        return 1.0

    bias = edu_meta.get("bias", "")
    depth = edu_meta.get("depth", 0)
    max_depth = max_depths.get(bias, 1)

    return depth / max_depth


def aggregate_cluster_depth(
    cluster_edus: List[str],
    edu_lookup: Dict,
    max_depths: Dict[str, int],
    agg_method: str = "min",
) -> float:
    """Aggregate normalized depths for all EDUs in a cluster."""
    normalized_depths = []
    for edu_id in cluster_edus:
        meta = edu_lookup.get(edu_id)
        if meta:
            nd = get_normalized_depth(meta, max_depths)
            normalized_depths.append(nd)

    if not normalized_depths:
        return 1.0

    if agg_method == "min":
        return min(normalized_depths)
    elif agg_method == "max":
        return max(normalized_depths)
    elif agg_method == "avg":
        return sum(normalized_depths) / len(normalized_depths)
    elif agg_method == "sum":
        return sum(normalized_depths)
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")


def order_clusters_by_depth(
    clusters: Dict,
    edu_lookup: Dict,
    agg_method: str = "min",
) -> OrderedDict:
    """Order clusters by aggregated normalized depth (shallowest first)."""
    max_depths = compute_max_depths_per_article(edu_lookup)

    cluster_depths = {}
    for cluster_id, edus in clusters.items():
        cluster_depths[cluster_id] = aggregate_cluster_depth(
            edus, edu_lookup, max_depths, agg_method
        )

    sorted_cluster_ids = sorted(
        clusters.keys(), key=lambda cid: (cluster_depths[cid], str(cid))
    )

    ordered = OrderedDict()
    for cid in sorted_cluster_ids:
        ordered[cid] = clusters[cid]

    return ordered


# =============================================================================
# DFI Feature Builders (RST-focused - no omission signal)
# =============================================================================


def build_rst_prominence_delta(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Pure RST prominence delta: avg(prominence) per side.
    Since all clusters have all 3 sides, this measures pure RST positioning.
    """
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in ordered_clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # AVG prominence
        W_left = (sum(scores_left) / len(scores_left)) if scores_left else 0.0
        W_center = (sum(scores_center) / len(scores_center)) if scores_center else 0.0
        W_right = (sum(scores_right) / len(scores_right)) if scores_right else 0.0

        deltas_left.append(W_left - W_center)
        deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


def build_rst_max_prominence(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Max prominence per cluster: captures most prominent mention per side.
    """
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in ordered_clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # MAX prominence
        W_left = max(scores_left) if scores_left else 0.0
        W_center = max(scores_center) if scores_center else 0.0
        W_right = max(scores_right) if scores_right else 0.0

        deltas_left.append(W_left - W_center)
        deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


def build_rst_repetition_delta(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Repetition count delta: how many times each side mentions each fact.
    Even without omission, sides may emphasize facts by repeating them.
    """
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in ordered_clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # COUNT (repetition)
        count_left = len(scores_left)
        count_center = len(scores_center)
        count_right = len(scores_right)

        deltas_left.append(count_left - count_center)
        deltas_right.append(count_right - count_center)

    return deltas_left, deltas_right


def build_rst_combined_3d(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Combined RST features: [max_prominence, avg_prominence, repetition] per cluster.
    3D feature per cluster, capturing multiple RST signals.
    """
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in ordered_clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # MAX prominence
        max_left = max(scores_left) if scores_left else 0.0
        max_center = max(scores_center) if scores_center else 0.0
        max_right = max(scores_right) if scores_right else 0.0

        # AVG prominence
        avg_left = (sum(scores_left) / len(scores_left)) if scores_left else 0.0
        avg_center = (sum(scores_center) / len(scores_center)) if scores_center else 0.0
        avg_right = (sum(scores_right) / len(scores_right)) if scores_right else 0.0

        # COUNT (repetition)
        count_left = len(scores_left)
        count_center = len(scores_center)
        count_right = len(scores_right)

        features_left.extend(
            [
                max_left - max_center,
                avg_left - avg_center,
                count_left - count_center,
            ]
        )

        features_right.extend(
            [
                max_right - max_center,
                avg_right - avg_center,
                count_right - count_center,
            ]
        )

    return features_left, features_right


def build_rst_depth_features(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Raw depth-based features: avg_depth, min_depth, max_depth per side.
    Captures where in the RST tree each side places each fact.
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    max_depths = compute_max_depths_per_article(edu_lookup)

    features_left = []
    features_right = []

    for cluster_id, edus in ordered_clusters.items():
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        def get_depths(edu_list):
            return [get_normalized_depth(e, max_depths) for e in edu_list]

        depths_left = get_depths(left_edus)
        depths_center = get_depths(center_edus)
        depths_right = get_depths(right_edus)

        # Min depth (shallowest = most prominent)
        min_left = min(depths_left) if depths_left else 1.0
        min_center = min(depths_center) if depths_center else 1.0
        min_right = min(depths_right) if depths_right else 1.0

        # Avg depth
        avg_left = sum(depths_left) / len(depths_left) if depths_left else 1.0
        avg_center = sum(depths_center) / len(depths_center) if depths_center else 1.0
        avg_right = sum(depths_right) / len(depths_right) if depths_right else 1.0

        features_left.extend(
            [
                min_center - min_left,  # Positive if left is shallower (more prominent)
                avg_center - avg_left,
            ]
        )

        features_right.extend(
            [
                min_center - min_right,
                avg_center - avg_right,
            ]
        )

    return features_left, features_right


def build_rst_role_features(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    RST role-based features: nucleus vs satellite proportion.
    Nuclei are the central units, satellites are supporting.
    """
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in ordered_clusters.items():
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        def nucleus_ratio(edu_list):
            if not edu_list:
                return 0.5  # Neutral default
            nucleus_count = sum(1 for e in edu_list if e.get("role") == "N")
            return nucleus_count / len(edu_list)

        nuc_left = nucleus_ratio(left_edus)
        nuc_center = nucleus_ratio(center_edus)
        nuc_right = nucleus_ratio(right_edus)

        features_left.append(nuc_left - nuc_center)
        features_right.append(nuc_right - nuc_center)

    return features_left, features_right


def build_rst_full_features(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Full RST feature set: prominence, depth, role, repetition.
    Comprehensive RST signal extraction.
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    max_depths = compute_max_depths_per_article(edu_lookup)

    features_left = []
    features_right = []

    for cluster_id, edus in ordered_clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        # 1. Max prominence
        max_left = max(scores_left) if scores_left else 0.0
        max_center = max(scores_center) if scores_center else 0.0
        max_right = max(scores_right) if scores_right else 0.0

        # 2. Avg prominence
        avg_left = (sum(scores_left) / len(scores_left)) if scores_left else 0.0
        avg_center = (sum(scores_center) / len(scores_center)) if scores_center else 0.0
        avg_right = (sum(scores_right) / len(scores_right)) if scores_right else 0.0

        # 3. Repetition count
        count_left = len(scores_left)
        count_center = len(scores_center)
        count_right = len(scores_right)

        # 4. Min depth (normalized)
        def get_depths(edu_list):
            return [get_normalized_depth(e, max_depths) for e in edu_list]

        depths_left = get_depths(left_edus)
        depths_center = get_depths(center_edus)
        depths_right = get_depths(right_edus)

        min_depth_left = min(depths_left) if depths_left else 1.0
        min_depth_center = min(depths_center) if depths_center else 1.0
        min_depth_right = min(depths_right) if depths_right else 1.0

        # 5. Nucleus ratio
        def nucleus_ratio(edu_list):
            if not edu_list:
                return 0.5
            nucleus_count = sum(1 for e in edu_list if e.get("role") == "N")
            return nucleus_count / len(edu_list)

        nuc_left = nucleus_ratio(left_edus)
        nuc_center = nucleus_ratio(center_edus)
        nuc_right = nucleus_ratio(right_edus)

        features_left.extend(
            [
                max_left - max_center,
                avg_left - avg_center,
                count_left - count_center,
                min_depth_center - min_depth_left,  # Positive if left is shallower
                nuc_left - nuc_center,
            ]
        )

        features_right.extend(
            [
                max_right - max_center,
                avg_right - avg_center,
                count_right - count_center,
                min_depth_center - min_depth_right,
                nuc_right - nuc_center,
            ]
        )

    return features_left, features_right


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
    feature_builder: Callable,
    rf_cfg: Dict,
    experiment_name: str,
    depth_agg: str,
    classifier_type: str = "rf",
) -> Tuple[object, int, Dict, List[float]]:
    """Run a single experiment with given feature builder."""

    # Build features with ordered clusters
    for row in train_rows:
        ordered = order_clusters_by_depth(
            row.get("clusters", {}), row.get("edu_lookup", {}), depth_agg
        )
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)
    for row in val_rows:
        ordered = order_clusters_by_depth(
            row.get("clusters", {}), row.get("edu_lookup", {}), depth_agg
        )
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)
    for row in test_rows:
        ordered = order_clusters_by_depth(
            row.get("clusters", {}), row.get("edu_lookup", {}), depth_agg
        )
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)

    # Build X, y
    x_train_raw, y_train = build_xy(train_rows, "feat_left", "feat_right")
    x_val_raw, y_val = build_xy(val_rows, "feat_left", "feat_right")
    x_test_raw, y_test = build_xy(test_rows, "feat_left", "feat_right")

    # Get max length from training data
    max_len = max((len(v) for v in x_train_raw), default=0)

    if max_len == 0:
        print(f"  WARNING: No features generated (max_len=0)")
        return None, 0, {"train": {}, "val": {}, "test": {}}, []

    # Pad/truncate
    X_train = pad_or_truncate(x_train_raw, max_len)
    X_val = pad_or_truncate(x_val_raw, max_len)
    X_test = pad_or_truncate(x_test_raw, max_len)

    input_dim = X_train.shape[1]
    print(f"  Input dimension: {input_dim}")

    # Combine train+val for final training
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # Train classifier
    if classifier_type == "rf":
        model = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_split=rf_cfg["min_samples_split"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            random_state=rf_cfg["seed"],
            n_jobs=-1,
        )
    elif classifier_type == "lr":
        model = LogisticRegression(
            max_iter=1000,
            random_state=rf_cfg["seed"],
        )
    elif classifier_type == "svm":
        model = SVC(
            kernel="rbf",
            random_state=rf_cfg["seed"],
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    model.fit(X_trainval, y_trainval)

    metrics = {
        "train": evaluate(model, X_train, y_train),
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
        "trainval": evaluate(model, X_trainval, y_trainval),
    }

    # Feature importances (RF only)
    feature_importances = []
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_.tolist()

    return model, input_dim, metrics, feature_importances


def main():
    args = parse_args()

    rf_cfg = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth if args.max_depth != 0 else None,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "seed": args.seed,
    }

    depth_agg = args.depth_agg
    min_clusters = args.min_clusters

    run_log = init_run_logging(
        script_subdir="rst-vs-omission",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "min_clusters": min_clusters,
            "depth_aggregation": depth_agg,
            "rf": rf_cfg,
        },
    )

    print("=" * 80)
    print("RST-ONLY EXPERIMENT: Isolating RST Contribution from Omission")
    print("=" * 80)
    print("\nHypothesis: By filtering to only clusters where ALL 3 bias sides")
    print("are present, we eliminate omission as a signal and can measure")
    print("the pure RST structural contribution to bias prediction.")
    print()
    print(f"Configuration:")
    print(f"  - Minimum full-coverage clusters per triplet: {min_clusters}")
    print(f"  - Depth aggregation: {depth_agg}")
    print()
    print("Reference baselines (FULL dataset, WITH omission signal):")
    print("  - Ordered bipartite coverage: ~88-90%")
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

    # Split first, then filter (to maintain split integrity)
    facts_train_orig, facts_val_orig, facts_test_orig = split_facts_by_existing_splits(
        facts, train_ids, val_ids, test_ids
    )

    print(f"\nOriginal data (before filtering):")
    print(f"  Train: {len(facts_train_orig)} triplets")
    print(f"  Val: {len(facts_val_orig)} triplets")
    print(f"  Test: {len(facts_test_orig)} triplets")

    # Filter each split separately
    facts_train, train_stats = filter_dataset(facts_train_orig, min_clusters)
    facts_val, val_stats = filter_dataset(facts_val_orig, min_clusters)
    facts_test, test_stats = filter_dataset(facts_test_orig, min_clusters)

    print(f"\nFiltered data (only full-coverage clusters, min {min_clusters}):")
    print(
        f"  Train: {len(facts_train)} triplets ({train_stats['triplet_retention_rate']:.1%} retained)"
    )
    print(
        f"  Val: {len(facts_val)} triplets ({val_stats['triplet_retention_rate']:.1%} retained)"
    )
    print(
        f"  Test: {len(facts_test)} triplets ({test_stats['triplet_retention_rate']:.1%} retained)"
    )
    print(
        f"\n  Train clusters: {train_stats['filtered_clusters']} ({train_stats['cluster_retention_rate']:.1%} of original)"
    )
    print(
        f"  Val clusters: {val_stats['filtered_clusters']} ({val_stats['cluster_retention_rate']:.1%} of original)"
    )
    print(
        f"  Test clusters: {test_stats['filtered_clusters']} ({test_stats['cluster_retention_rate']:.1%} of original)"
    )

    if len(facts_test) == 0:
        print("\nERROR: No test triplets remain after filtering!")
        return

    # Define experiments - RST-only feature builders
    experiments = [
        ("rst_prominence_avg", "Avg prominence delta", build_rst_prominence_delta),
        ("rst_prominence_max", "Max prominence delta", build_rst_max_prominence),
        ("rst_repetition", "Repetition count delta", build_rst_repetition_delta),
        ("rst_combined_3d", "Combined (max+avg+count)", build_rst_combined_3d),
        ("rst_depth_features", "Depth-based features", build_rst_depth_features),
        ("rst_role_features", "Role (nucleus/satellite)", build_rst_role_features),
        ("rst_full_features", "Full RST features (5D)", build_rst_full_features),
    ]

    os.makedirs(args.model_dir, exist_ok=True)
    results = {}

    # Run experiments with different classifiers
    classifiers = ["rf", "lr", "svm"]

    for exp_name, description, feature_builder in experiments:
        for clf_type in classifiers:
            full_exp_name = f"{exp_name}_{clf_type}"

            print(f"\n{'=' * 80}")
            print(f"Experiment: {full_exp_name}")
            print(f"Description: {description} + {clf_type.upper()}")
            print("=" * 80)

            train_rows = copy.deepcopy(facts_train)
            val_rows = copy.deepcopy(facts_val)
            test_rows = copy.deepcopy(facts_test)

            model, input_dim, metrics, feature_importances = (
                train_and_evaluate_experiment(
                    train_rows,
                    val_rows,
                    test_rows,
                    feature_builder,
                    rf_cfg,
                    full_exp_name,
                    depth_agg,
                    clf_type,
                )
            )

            if model is None:
                print("  SKIPPED: No features generated")
                continue

            model_path = os.path.join(args.model_dir, f"{full_exp_name}.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "input_dim": input_dim}, f)

            results[full_exp_name] = {
                "description": description,
                "classifier": clf_type,
                "input_dim": input_dim,
                "metrics": metrics,
                "model_path": model_path,
            }

            if feature_importances:
                results[full_exp_name]["feature_importances_top10"] = sorted(
                    enumerate(feature_importances), key=lambda x: -x[1]
                )[:10]

            print(
                f"  Val:  acc={metrics['val']['accuracy']:.4f}, f1={metrics['val']['macro_f1']:.4f}"
            )
            print(
                f"  Test: acc={metrics['test']['accuracy']:.4f}, f1={metrics['test']['macro_f1']:.4f}"
            )

    # Reference baselines
    reference_full_dataset = 0.8977  # Bipartite coverage (full dataset)
    random_baseline = 0.50

    # Compute deltas
    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        exp_data["delta_vs_full_dataset"] = test_acc - reference_full_dataset
        exp_data["delta_vs_random"] = test_acc - random_baseline

    output = {
        "setup": {
            "goal": "Isolate RST contribution by filtering to full-coverage clusters only",
            "description": "Only clusters where ALL 3 bias sides (left, center, right) are present",
            "hypothesis": "If omission is primary signal, accuracy should drop significantly",
            "min_clusters_per_triplet": min_clusters,
            "depth_aggregation": depth_agg,
            "facts": args.facts,
            "split_dir": args.split_dir,
            "rf": rf_cfg,
            "created": datetime.now().isoformat(),
        },
        "filtering_stats": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        },
        "reference_baselines": {
            "full_dataset_bipartite_coverage": {"test_acc": reference_full_dataset},
            "random_chance": {"test_acc": random_baseline},
        },
        "experiments": results,
    }

    save_json(args.out, output)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: RST-Only Experiment (No Omission Signal)")
    print("=" * 100)
    print(f"\nFiltering: Only clusters with ALL 3 bias sides present")
    print(f"Min clusters per triplet: {min_clusters}")
    print(f"Test triplets: {len(facts_test)} (of original {len(facts_test_orig)})")
    print(
        f"Test clusters: {test_stats['filtered_clusters']} ({test_stats['cluster_retention_rate']:.1%} retained)"
    )
    print()
    print(
        f"{'Experiment':<35} {'Dim':<8} {'Test Acc':<12} {'vs Full Data':<15} {'vs Random':<12}"
    )
    print("-" * 100)

    for exp_name, exp_data in sorted(
        results.items(), key=lambda x: -x[1]["metrics"]["test"]["accuracy"]
    ):
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        delta_full = exp_data["delta_vs_full_dataset"]
        delta_random = exp_data["delta_vs_random"]
        dim = exp_data["input_dim"]
        print(
            f"  {exp_name:<33} {dim:<8} {test_acc:<12.4f} {delta_full:<+15.4f} {delta_random:<+12.4f}"
        )

    print("-" * 100)
    print(f"Reference: Full dataset bipartite coverage = {reference_full_dataset:.2%}")
    print(f"Reference: Random chance = {random_baseline:.2%}")

    # Find best experiment
    best_exp = max(results.items(), key=lambda x: x[1]["metrics"]["test"]["accuracy"])
    print(f"\nBest RST-only experiment: {best_exp[0]}")
    print(f"  Test accuracy: {best_exp[1]['metrics']['test']['accuracy']:.4f}")
    print(f"  vs Full dataset: {best_exp[1]['delta_vs_full_dataset']:+.4f}")

    # Calculate RST contribution
    best_rst_acc = best_exp[1]["metrics"]["test"]["accuracy"]
    rst_contribution = (
        (best_rst_acc - random_baseline)
        / (reference_full_dataset - random_baseline)
        * 100
    )
    print(f"\n{'=' * 80}")
    print("RST vs OMISSION CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"\nFull dataset accuracy: {reference_full_dataset:.2%}")
    print(f"RST-only accuracy: {best_rst_acc:.2%}")
    print(f"Random baseline: {random_baseline:.2%}")
    print(f"\nEstimated RST contribution: {rst_contribution:.1f}%")
    print(f"Estimated Omission contribution: {100 - rst_contribution:.1f}%")

    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out, "experiments": list(results.keys())})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
