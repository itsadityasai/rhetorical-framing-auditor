"""
DFI Alternatives: Testing 3 Different Feature Vector Construction Methods

This script tests three alternative approaches to constructing DFI feature vectors
from clustered facts, using Random Forest as the classifier.

BASELINE (current approach):
- For each cluster, compute W_doc = max(prominence scores) for each bias
- DFI delta = W_side - W_center (single scalar per cluster)

ALTERNATIVE 1: Cumulative Prominence
- Change: W_doc = sum(prominence scores) instead of max
- Optional: Normalize by EDU count to prevent long-article bias
- Rationale: Captures both structural prominence + repetition frequency

ALTERNATIVE 2: Distributional DFI (3D per cluster)
- Change: For each cluster, output 3 metrics instead of 1:
  - max(W): Peak prominence (was this in headline?)
  - sum(W): Total structural emphasis
  - count(EDUs): Pure repetition frequency
- Rationale: Let the classifier learn which aspect matters most

ALTERNATIVE 3: Bipartite Decomposition
- Change: Match EDUs 1-to-1-to-1 within clusters using semantic similarity
- Leftover EDUs become separate clusters with 0s for missing sides
- Rationale: Explicitly encode coverage asymmetry (our strongest signal)

Reference baseline: Padded DFI + RF = 87.78% test accuracy
"""

import argparse
import copy
import json
import math
import os
import pickle
import sys
from datetime import datetime
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Add parent directory for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = "dfi-alternatives/results/dfi_alternatives_results.json"
DEFAULT_MODEL_DIR = "dfi-alternatives/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test 3 DFI alternatives with Random Forest"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)

    # Random Forest hyperparameters (from tuned model)
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
# BASELINE: Original DFI (max prominence)
# =============================================================================


def build_baseline_dfi(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    BASELINE: Original approach using max(prominence) per cluster.
    Returns (deltas_left, deltas_right) for left-vs-center and right-vs-center.
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        W_left = max(scores_left) if scores_left else 0.0
        W_center = max(scores_center) if scores_center else 0.0
        W_right = max(scores_right) if scores_right else 0.0

        deltas_left.append(W_left - W_center)
        deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


def build_baseline_coverage(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """BASELINE: Coverage-only (binary presence)."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        has_left = 1 if get_side_scores(edus, edu_lookup, "left") else 0
        has_center = 1 if get_side_scores(edus, edu_lookup, "center") else 0
        has_right = 1 if get_side_scores(edus, edu_lookup, "right") else 0

        deltas_left.append(has_left - has_center)
        deltas_right.append(has_right - has_center)

    return deltas_left, deltas_right


def build_baseline_combined(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """BASELINE: Combined coverage + structural (2 features per cluster)."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # Coverage
        has_left = 1 if scores_left else 0
        has_center = 1 if scores_center else 0
        has_right = 1 if scores_right else 0

        cov_left = has_left - has_center
        cov_right = has_right - has_center

        # Structural (max prominence)
        W_left = max(scores_left) if scores_left else 0.0
        W_center = max(scores_center) if scores_center else 0.0
        W_right = max(scores_right) if scores_right else 0.0

        str_left = W_left - W_center
        str_right = W_right - W_center

        # Combined: [cov, str] per cluster
        features_left.extend([cov_left, str_left])
        features_right.extend([cov_right, str_right])

    return features_left, features_right


# =============================================================================
# ALTERNATIVE 1: Cumulative Prominence (sum instead of max)
# =============================================================================


def build_alt1_cumulative(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 1: Cumulative prominence using sum(scores) instead of max.
    Captures both structural prominence AND repetition frequency.
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # SUM instead of MAX
        W_left = sum(scores_left) if scores_left else 0.0
        W_center = sum(scores_center) if scores_center else 0.0
        W_right = sum(scores_right) if scores_right else 0.0

        deltas_left.append(W_left - W_center)
        deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


def build_alt1_cumulative_normalized(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 1b: Cumulative prominence normalized by EDU count.
    Prevents long articles from automatically scoring higher.
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # Sum normalized by count (average prominence)
        W_left = (sum(scores_left) / len(scores_left)) if scores_left else 0.0
        W_center = (sum(scores_center) / len(scores_center)) if scores_center else 0.0
        W_right = (sum(scores_right) / len(scores_right)) if scores_right else 0.0

        deltas_left.append(W_left - W_center)
        deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


def build_alt1_combined(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 1c: Coverage + Cumulative prominence combined.
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # Coverage
        has_left = 1 if scores_left else 0
        has_center = 1 if scores_center else 0
        has_right = 1 if scores_right else 0

        cov_left = has_left - has_center
        cov_right = has_right - has_center

        # Cumulative prominence (SUM)
        W_left = sum(scores_left) if scores_left else 0.0
        W_center = sum(scores_center) if scores_center else 0.0
        W_right = sum(scores_right) if scores_right else 0.0

        str_left = W_left - W_center
        str_right = W_right - W_center

        features_left.extend([cov_left, str_left])
        features_right.extend([cov_right, str_right])

    return features_left, features_right


# =============================================================================
# ALTERNATIVE 2: Distributional DFI (3D per cluster)
# =============================================================================


def build_alt2_distributional(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 2: Distributional DFI with 3 metrics per cluster.
    For each cluster: [max_delta, sum_delta, count_delta]
    - max: Peak prominence (headline placement)
    - sum: Total structural emphasis
    - count: Pure repetition frequency
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # MAX prominence
        max_left = max(scores_left) if scores_left else 0.0
        max_center = max(scores_center) if scores_center else 0.0
        max_right = max(scores_right) if scores_right else 0.0

        # SUM prominence
        sum_left = sum(scores_left) if scores_left else 0.0
        sum_center = sum(scores_center) if scores_center else 0.0
        sum_right = sum(scores_right) if scores_right else 0.0

        # COUNT (repetition frequency)
        count_left = len(scores_left)
        count_center = len(scores_center)
        count_right = len(scores_right)

        # Deltas for left-vs-center
        features_left.extend(
            [
                max_left - max_center,
                sum_left - sum_center,
                count_left - count_center,
            ]
        )

        # Deltas for right-vs-center
        features_right.extend(
            [
                max_right - max_center,
                sum_right - sum_center,
                count_right - count_center,
            ]
        )

    return features_left, features_right


def build_alt2_distributional_coverage(
    fact_row: Dict,
) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 2b: Distributional DFI with coverage included.
    For each cluster: [coverage_delta, max_delta, sum_delta, count_delta]
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # Coverage (binary)
        cov_left = 1 if scores_left else 0
        cov_center = 1 if scores_center else 0
        cov_right = 1 if scores_right else 0

        # MAX prominence
        max_left = max(scores_left) if scores_left else 0.0
        max_center = max(scores_center) if scores_center else 0.0
        max_right = max(scores_right) if scores_right else 0.0

        # SUM prominence
        sum_left = sum(scores_left) if scores_left else 0.0
        sum_center = sum(scores_center) if scores_center else 0.0
        sum_right = sum(scores_right) if scores_right else 0.0

        # COUNT
        count_left = len(scores_left)
        count_center = len(scores_center)
        count_right = len(scores_right)

        # 4D feature vector per cluster
        features_left.extend(
            [
                cov_left - cov_center,
                max_left - max_center,
                sum_left - sum_center,
                count_left - count_center,
            ]
        )

        features_right.extend(
            [
                cov_right - cov_center,
                max_right - max_center,
                sum_right - sum_center,
                count_right - count_center,
            ]
        )

    return features_left, features_right


def build_alt2_distributional_normalized(
    fact_row: Dict,
) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 2c: Distributional DFI with normalized values.
    - max: normalized to [0,1] range
    - avg: mean prominence (sum/count)
    - log_count: log(1+count) for stability
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        scores_left = get_side_scores(edus, edu_lookup, "left")
        scores_center = get_side_scores(edus, edu_lookup, "center")
        scores_right = get_side_scores(edus, edu_lookup, "right")

        # MAX prominence (already in [0,1] due to W_log_depth)
        max_left = max(scores_left) if scores_left else 0.0
        max_center = max(scores_center) if scores_center else 0.0
        max_right = max(scores_right) if scores_right else 0.0

        # AVG prominence
        avg_left = (sum(scores_left) / len(scores_left)) if scores_left else 0.0
        avg_center = (sum(scores_center) / len(scores_center)) if scores_center else 0.0
        avg_right = (sum(scores_right) / len(scores_right)) if scores_right else 0.0

        # LOG COUNT
        log_count_left = math.log1p(len(scores_left))
        log_count_center = math.log1p(len(scores_center))
        log_count_right = math.log1p(len(scores_right))

        features_left.extend(
            [
                max_left - max_center,
                avg_left - avg_center,
                log_count_left - log_count_center,
            ]
        )

        features_right.extend(
            [
                max_right - max_center,
                avg_right - avg_center,
                log_count_right - log_count_center,
            ]
        )

    return features_left, features_right


# =============================================================================
# ALTERNATIVE 3: Bipartite Decomposition (1-to-1-to-1 matching)
# =============================================================================


def compute_edu_similarity(edu1_meta: Dict, edu2_meta: Dict) -> float:
    """
    Compute similarity between two EDUs based on structural features.
    Since we don't have embeddings, use structural proximity as proxy.
    """
    if not edu1_meta or not edu2_meta:
        return 0.0

    # Use inverse of depth difference as similarity
    depth1 = edu1_meta.get("depth", 0)
    depth2 = edu2_meta.get("depth", 0)
    depth_sim = 1.0 / (1.0 + abs(depth1 - depth2))

    # Use prominence similarity
    prom1 = get_edu_prominence(edu1_meta)
    prom2 = get_edu_prominence(edu2_meta)
    prom_sim = 1.0 - abs(prom1 - prom2)

    return (depth_sim + prom_sim) / 2.0


def greedy_bipartite_match(
    left_edus: List[Dict],
    center_edus: List[Dict],
    right_edus: List[Dict],
) -> Tuple[List[Tuple], List[Dict], List[Dict], List[Dict]]:
    """
    Greedy 1-to-1-to-1 matching of EDUs across the three sides.

    Returns:
        matched_triplets: List of (left_edu, center_edu, right_edu) tuples
        leftover_left: Unmatched left EDUs
        leftover_center: Unmatched center EDUs
        leftover_right: Unmatched right EDUs
    """
    # Make copies to avoid modifying originals
    left_pool = list(left_edus)
    center_pool = list(center_edus)
    right_pool = list(right_edus)

    matched_triplets = []

    # Greedy matching: while all three pools have EDUs
    while left_pool and center_pool and right_pool:
        # Find best triplet based on average pairwise similarity
        best_score = -1
        best_triplet = None
        best_indices = None

        for i, l_edu in enumerate(left_pool):
            for j, c_edu in enumerate(center_pool):
                for k, r_edu in enumerate(right_pool):
                    # Average pairwise similarity
                    sim_lc = compute_edu_similarity(l_edu, c_edu)
                    sim_lr = compute_edu_similarity(l_edu, r_edu)
                    sim_cr = compute_edu_similarity(c_edu, r_edu)
                    score = (sim_lc + sim_lr + sim_cr) / 3.0

                    if score > best_score:
                        best_score = score
                        best_triplet = (l_edu, c_edu, r_edu)
                        best_indices = (i, j, k)

        if best_triplet:
            matched_triplets.append(best_triplet)
            # Remove matched EDUs (in reverse order to preserve indices)
            left_pool.pop(best_indices[0])
            center_pool.pop(best_indices[1])
            right_pool.pop(best_indices[2])

    return matched_triplets, left_pool, center_pool, right_pool


def build_alt3_bipartite(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 3: Bipartite decomposition with 1-to-1-to-1 matching.

    For each cluster:
    1. Match EDUs 1-to-1-to-1 based on structural similarity
    2. Matched triplets contribute prominence deltas
    3. Leftover EDUs become "omission" features (explicit 0s for missing sides)
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        # Get EDU metadata for each side
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        # Perform greedy matching
        matched, leftover_l, leftover_c, leftover_r = greedy_bipartite_match(
            left_edus, center_edus, right_edus
        )

        # Process matched triplets
        for l_edu, c_edu, r_edu in matched:
            W_left = get_edu_prominence(l_edu)
            W_center = get_edu_prominence(c_edu)
            W_right = get_edu_prominence(r_edu)

            features_left.append(W_left - W_center)
            features_right.append(W_right - W_center)

        # Process leftover EDUs as explicit omissions
        # Leftover left EDUs: center omitted this mention
        for l_edu in leftover_l:
            W_left = get_edu_prominence(l_edu)
            features_left.append(W_left - 0.0)  # Center = 0 (omitted)

        # Leftover center EDUs: both sides omitted (or partially)
        for c_edu in leftover_c:
            W_center = get_edu_prominence(c_edu)
            features_left.append(0.0 - W_center)  # Left omitted
            features_right.append(0.0 - W_center)  # Right omitted

        # Leftover right EDUs: center omitted this mention
        for r_edu in leftover_r:
            W_right = get_edu_prominence(r_edu)
            features_right.append(W_right - 0.0)  # Center = 0 (omitted)

    return features_left, features_right


def build_alt3_bipartite_coverage(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 3b: Bipartite with coverage features.
    Each matched/unmatched EDU contributes a coverage delta.
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        matched, leftover_l, leftover_c, leftover_r = greedy_bipartite_match(
            left_edus, center_edus, right_edus
        )

        # Matched triplets: all present, delta = 0
        for _ in matched:
            features_left.append(0)  # 1 - 1 = 0
            features_right.append(0)  # 1 - 1 = 0

        # Leftover left: left has, center doesn't
        for _ in leftover_l:
            features_left.append(1)  # 1 - 0 = 1

        # Leftover center: center has, sides don't
        for _ in leftover_c:
            features_left.append(-1)  # 0 - 1 = -1
            features_right.append(-1)  # 0 - 1 = -1

        # Leftover right: right has, center doesn't
        for _ in leftover_r:
            features_right.append(1)  # 1 - 0 = 1

    return features_left, features_right


def build_alt3_bipartite_combined(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    ALTERNATIVE 3c: Bipartite with both coverage and prominence.
    Each matched/unmatched EDU contributes [coverage_delta, prominence_delta].
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        matched, leftover_l, leftover_c, leftover_r = greedy_bipartite_match(
            left_edus, center_edus, right_edus
        )

        # Matched triplets
        for l_edu, c_edu, r_edu in matched:
            W_left = get_edu_prominence(l_edu)
            W_center = get_edu_prominence(c_edu)
            W_right = get_edu_prominence(r_edu)

            # [coverage, prominence] for left-vs-center
            features_left.extend([0, W_left - W_center])
            # [coverage, prominence] for right-vs-center
            features_right.extend([0, W_right - W_center])

        # Leftover left EDUs
        for l_edu in leftover_l:
            W_left = get_edu_prominence(l_edu)
            features_left.extend([1, W_left])  # Center = 0

        # Leftover center EDUs
        for c_edu in leftover_c:
            W_center = get_edu_prominence(c_edu)
            features_left.extend([-1, -W_center])
            features_right.extend([-1, -W_center])

        # Leftover right EDUs
        for r_edu in leftover_r:
            W_right = get_edu_prominence(r_edu)
            features_right.extend([1, W_right])  # Center = 0

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
) -> Tuple[object, int, Dict, List[float]]:
    """Run a single experiment with given feature builder."""

    # Build features
    for row in train_rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)
    for row in val_rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)
    for row in test_rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)

    # Build X, y
    x_train_raw, y_train = build_xy(train_rows, "feat_left", "feat_right")
    x_val_raw, y_val = build_xy(val_rows, "feat_left", "feat_right")
    x_test_raw, y_test = build_xy(test_rows, "feat_left", "feat_right")

    # Get max length from training data
    max_len = max((len(v) for v in x_train_raw), default=0)

    # Pad/truncate
    X_train = pad_or_truncate(x_train_raw, max_len)
    X_val = pad_or_truncate(x_val_raw, max_len)
    X_test = pad_or_truncate(x_test_raw, max_len)

    input_dim = X_train.shape[1]
    print(f"  Input dimension: {input_dim}")

    # Combine train+val for final training
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        random_state=rf_cfg["seed"],
        n_jobs=-1,
    )
    model.fit(X_trainval, y_trainval)

    metrics = {
        "train": evaluate(model, X_train, y_train),
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
        "trainval": evaluate(model, X_trainval, y_trainval),
    }

    # Feature importances
    feature_importances = model.feature_importances_.tolist()

    return model, input_dim, metrics, feature_importances


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
        script_subdir="dfi-alternatives",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "rf": rf_cfg,
        },
    )

    print("=" * 80)
    print("DFI Alternatives: Testing 3 Feature Vector Construction Methods")
    print("=" * 80)
    print("\nReference baseline: Padded DFI + RF = 87.78% test accuracy")
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
        # BASELINE
        ("baseline_max", "BASELINE: max(prominence) per cluster", build_baseline_dfi),
        ("baseline_coverage", "BASELINE: coverage-only", build_baseline_coverage),
        (
            "baseline_combined",
            "BASELINE: coverage + max(prominence)",
            build_baseline_combined,
        ),
        # ALTERNATIVE 1: Cumulative Prominence
        ("alt1_cumulative", "ALT1: sum(prominence) per cluster", build_alt1_cumulative),
        (
            "alt1_cumulative_norm",
            "ALT1: avg(prominence) per cluster",
            build_alt1_cumulative_normalized,
        ),
        ("alt1_combined", "ALT1: coverage + sum(prominence)", build_alt1_combined),
        # ALTERNATIVE 2: Distributional DFI
        (
            "alt2_distributional",
            "ALT2: 3D [max, sum, count] per cluster",
            build_alt2_distributional,
        ),
        (
            "alt2_dist_coverage",
            "ALT2: 4D [cov, max, sum, count] per cluster",
            build_alt2_distributional_coverage,
        ),
        (
            "alt2_dist_normalized",
            "ALT2: 3D [max, avg, log_count] per cluster",
            build_alt2_distributional_normalized,
        ),
        # ALTERNATIVE 3: Bipartite Decomposition
        (
            "alt3_bipartite",
            "ALT3: 1-to-1-to-1 matching (prominence)",
            build_alt3_bipartite,
        ),
        (
            "alt3_bipartite_cov",
            "ALT3: 1-to-1-to-1 matching (coverage)",
            build_alt3_bipartite_coverage,
        ),
        (
            "alt3_bipartite_combined",
            "ALT3: 1-to-1-to-1 matching (cov + prominence)",
            build_alt3_bipartite_combined,
        ),
    ]

    os.makedirs(args.model_dir, exist_ok=True)
    results = {}

    for exp_name, description, feature_builder in experiments:
        print(f"\n{'=' * 80}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {description}")
        print("=" * 80)

        train_rows = copy.deepcopy(facts_train)
        val_rows = copy.deepcopy(facts_val)
        test_rows = copy.deepcopy(facts_test)

        model, input_dim, metrics, feature_importances = train_and_evaluate_experiment(
            train_rows, val_rows, test_rows, feature_builder, rf_cfg, exp_name
        )

        model_path = os.path.join(args.model_dir, f"{exp_name}.pkl")
        save_model(model_path, model, input_dim, exp_name, rf_cfg)

        results[exp_name] = {
            "description": description,
            "input_dim": input_dim,
            "metrics": metrics,
            "model_path": model_path,
            "feature_importances_top10": sorted(
                enumerate(feature_importances), key=lambda x: -x[1]
            )[:10],
        }

        print(
            f"  Val:  acc={metrics['val']['accuracy']:.4f}, f1={metrics['val']['macro_f1']:.4f}"
        )
        print(
            f"  Test: acc={metrics['test']['accuracy']:.4f}, f1={metrics['test']['macro_f1']:.4f}"
        )

    # Reference baseline
    reference_baseline = 0.8778  # Padded DFI + RF (tuned)

    # Compute deltas
    for exp_name, exp_data in results.items():
        exp_data["delta_vs_baseline_rf"] = (
            exp_data["metrics"]["test"]["accuracy"] - reference_baseline
        )

    output = {
        "setup": {
            "goal": "Test 3 alternative DFI construction methods",
            "alternatives": {
                "alt1": "Cumulative Prominence: sum() instead of max()",
                "alt2": "Distributional DFI: 3D [max, sum, count] per cluster",
                "alt3": "Bipartite Decomposition: 1-to-1-to-1 EDU matching",
            },
            "facts": args.facts,
            "split_dir": args.split_dir,
            "rf": rf_cfg,
            "created": datetime.now().isoformat(),
        },
        "reference_baseline": {
            "padded_dfi_rf_tuned": {"test_acc": reference_baseline},
        },
        "experiments": results,
    }

    save_json(args.out, output)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: DFI Alternatives Results")
    print("=" * 100)
    print(
        f"\n{'Experiment':<30} {'Dim':<8} {'Test Acc':<12} {'Test F1':<12} {'vs RF Baseline':<15}"
    )
    print("-" * 100)

    # Group by alternative
    for group, prefix in [
        ("BASELINE", "baseline"),
        ("ALT 1: Cumulative", "alt1"),
        ("ALT 2: Distributional", "alt2"),
        ("ALT 3: Bipartite", "alt3"),
    ]:
        print(f"\n{group}:")
        for exp_name, exp_data in results.items():
            if exp_name.startswith(prefix):
                test_acc = exp_data["metrics"]["test"]["accuracy"]
                test_f1 = exp_data["metrics"]["test"]["macro_f1"]
                delta = exp_data["delta_vs_baseline_rf"]
                dim = exp_data["input_dim"]
                print(
                    f"  {exp_name:<28} {dim:<8} {test_acc:<12.4f} {test_f1:<12.4f} {delta:<+15.4f}"
                )

    print("-" * 100)
    print(f"Reference: Padded DFI + RF (tuned) = {reference_baseline:.2%}")

    # Find best experiment
    best_exp = max(results.items(), key=lambda x: x[1]["metrics"]["test"]["accuracy"])
    print(f"\nBest experiment: {best_exp[0]}")
    print(f"  Test accuracy: {best_exp[1]['metrics']['test']['accuracy']:.4f}")
    print(f"  vs RF Baseline: {best_exp[1]['delta_vs_baseline_rf']:+.4f}")

    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out, "experiments": list(results.keys())})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
