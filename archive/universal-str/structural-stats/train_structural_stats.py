"""
Universal Structure Approach C: Structural Statistics Array

Instead of per-fact features, this approach computes summary statistics that
capture the OVERALL structural treatment of facts in an article. This creates
a fixed-size feature vector regardless of how many facts exist in the triplet.

Feature Vector (5 columns):
1. Max Prominence Delta - Is there a heavily promoted fact? (center ignored)
2. Average Repetition Delta - How often are facts repeated vs center?
3. Variance of Depth - Structural shuffle vs mirror of center?
4. Omission Count - How many facts completely hidden?
5. Average Deep-Burial Delta - Facts pushed to satellites/deep nodes?

This directly tests whether aggregate structural patterns distinguish biased
from centrist coverage, independent of specific topic/facts.

Reference baselines:
- Unordered padded DFI + RF: 87.78%
- Unordered bipartite coverage: 89.77%
"""

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import SVC

# Add project root for module imports
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "params.yaml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(PROJECT_ROOT))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = (
    "archive/universal-str/structural-stats/results/structural_stats_results.json"
)
DEFAULT_MODEL_DIR = "archive/universal-str/structural-stats/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Structure: Structural Statistics Array"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)

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
# Prominence Calculation
# =============================================================================


def W_log_depth(depth: int) -> float:
    """Prominence formula: 1 / (1 + log(1 + depth))"""
    return 1.0 / (1.0 + math.log1p(depth))


def get_edu_prominence(edu_meta: Dict) -> float:
    """Compute prominence score for a single EDU."""
    if not edu_meta:
        return 0.0
    depth = edu_meta.get("depth", 0)
    return W_log_depth(depth)


# =============================================================================
# Structural Statistics Feature Construction
# =============================================================================


def build_structural_stats_basic(
    fact_row: Dict,
) -> Tuple[List[float], List[float], Dict]:
    """
    Build the basic 5-column structural statistics array.

    Features:
    1. Max Prominence Delta - max(side_prominence - center_prominence)
    2. Average Repetition Delta - avg(count_side - count_center) per cluster
    3. Variance of Depth - var(side_depths) - var(center_depths)
    4. Omission Count - clusters where center has fact but side doesn't
    5. Average Deep-Burial Delta - avg delta for facts at depth >= 10
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    # Collect data across all clusters
    left_prominence_deltas = []
    right_prominence_deltas = []
    left_count_deltas = []
    right_count_deltas = []
    left_depths = []
    right_depths = []
    center_depths = []
    left_omissions = 0
    right_omissions = 0
    left_deep_deltas = []  # deltas for facts at depth >= 10
    right_deep_deltas = []

    DEEP_THRESHOLD = 10

    for cluster_id, edus in clusters.items():
        left_proms = []
        center_proms = []
        right_proms = []
        left_d = []
        center_d = []
        right_d = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            depth = meta.get("depth", 0)
            prom = get_edu_prominence(meta)

            if bias == "left":
                left_proms.append(prom)
                left_d.append(depth)
                left_depths.append(depth)
            elif bias == "center":
                center_proms.append(prom)
                center_d.append(depth)
                center_depths.append(depth)
            elif bias == "right":
                right_proms.append(prom)
                right_d.append(depth)
                right_depths.append(depth)

        has_left = len(left_proms) > 0
        has_center = len(center_proms) > 0
        has_right = len(right_proms) > 0

        # Prominence deltas
        if has_left and has_center:
            avg_left = sum(left_proms) / len(left_proms)
            avg_center = sum(center_proms) / len(center_proms)
            left_prominence_deltas.append(avg_left - avg_center)
        elif has_left and not has_center:
            left_prominence_deltas.append(sum(left_proms) / len(left_proms))

        if has_right and has_center:
            avg_right = sum(right_proms) / len(right_proms)
            avg_center = sum(center_proms) / len(center_proms)
            right_prominence_deltas.append(avg_right - avg_center)
        elif has_right and not has_center:
            right_prominence_deltas.append(sum(right_proms) / len(right_proms))

        # Count deltas (repetition)
        left_count_deltas.append(len(left_proms) - len(center_proms))
        right_count_deltas.append(len(right_proms) - len(center_proms))

        # Omissions
        if has_center and not has_left:
            left_omissions += 1
        if has_center and not has_right:
            right_omissions += 1

        # Deep burial deltas (facts at depth >= threshold)
        if has_center:
            avg_center_prom = sum(center_proms) / len(center_proms)
            for i, d in enumerate(left_d):
                if d >= DEEP_THRESHOLD:
                    left_deep_deltas.append(left_proms[i] - avg_center_prom)
            for i, d in enumerate(right_d):
                if d >= DEEP_THRESHOLD:
                    right_deep_deltas.append(right_proms[i] - avg_center_prom)

    # Compute final statistics
    # 1. Max Prominence Delta
    max_prom_left = max(left_prominence_deltas) if left_prominence_deltas else 0.0
    max_prom_right = max(right_prominence_deltas) if right_prominence_deltas else 0.0

    # 2. Average Repetition Delta
    avg_rep_left = (
        sum(left_count_deltas) / len(left_count_deltas) if left_count_deltas else 0.0
    )
    avg_rep_right = (
        sum(right_count_deltas) / len(right_count_deltas) if right_count_deltas else 0.0
    )

    # 3. Variance of Depth (side - center)
    var_left = np.var(left_depths) if len(left_depths) > 1 else 0.0
    var_right = np.var(right_depths) if len(right_depths) > 1 else 0.0
    var_center = np.var(center_depths) if len(center_depths) > 1 else 0.0
    depth_var_delta_left = var_left - var_center
    depth_var_delta_right = var_right - var_center

    # 4. Omission Count (already computed)

    # 5. Average Deep-Burial Delta
    avg_deep_left = (
        sum(left_deep_deltas) / len(left_deep_deltas) if left_deep_deltas else 0.0
    )
    avg_deep_right = (
        sum(right_deep_deltas) / len(right_deep_deltas) if right_deep_deltas else 0.0
    )

    left_features = [
        max_prom_left,
        avg_rep_left,
        depth_var_delta_left,
        float(left_omissions),
        avg_deep_left,
    ]

    right_features = [
        max_prom_right,
        avg_rep_right,
        depth_var_delta_right,
        float(right_omissions),
        avg_deep_right,
    ]

    debug_info = {
        "left_omissions": left_omissions,
        "right_omissions": right_omissions,
        "num_clusters": len(clusters),
    }

    return left_features, right_features, debug_info


def build_structural_stats_extended(
    fact_row: Dict,
) -> Tuple[List[float], List[float], Dict]:
    """
    Extended structural statistics with additional features (10 total).

    Features:
    1. Max Prominence Delta
    2. Min Prominence Delta (most buried fact)
    3. Average Prominence Delta
    4. Std Prominence Delta
    5. Average Repetition Delta
    6. Depth Variance Delta
    7. Average Depth Delta
    8. Omission Count
    9. Exclusive Count (facts side has but center doesn't)
    10. Average Deep-Burial Delta
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    left_prominence_deltas = []
    right_prominence_deltas = []
    left_count_deltas = []
    right_count_deltas = []
    left_depths = []
    right_depths = []
    center_depths = []
    left_omissions = 0
    right_omissions = 0
    left_exclusives = 0
    right_exclusives = 0
    left_deep_deltas = []
    right_deep_deltas = []

    DEEP_THRESHOLD = 10

    for cluster_id, edus in clusters.items():
        left_proms = []
        center_proms = []
        right_proms = []
        left_d = []
        center_d = []
        right_d = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            depth = meta.get("depth", 0)
            prom = get_edu_prominence(meta)

            if bias == "left":
                left_proms.append(prom)
                left_d.append(depth)
                left_depths.append(depth)
            elif bias == "center":
                center_proms.append(prom)
                center_d.append(depth)
                center_depths.append(depth)
            elif bias == "right":
                right_proms.append(prom)
                right_d.append(depth)
                right_depths.append(depth)

        has_left = len(left_proms) > 0
        has_center = len(center_proms) > 0
        has_right = len(right_proms) > 0

        # Prominence deltas
        if has_left and has_center:
            avg_left = sum(left_proms) / len(left_proms)
            avg_center = sum(center_proms) / len(center_proms)
            left_prominence_deltas.append(avg_left - avg_center)
        elif has_left and not has_center:
            left_prominence_deltas.append(sum(left_proms) / len(left_proms))
            left_exclusives += 1

        if has_right and has_center:
            avg_right = sum(right_proms) / len(right_proms)
            avg_center = sum(center_proms) / len(center_proms)
            right_prominence_deltas.append(avg_right - avg_center)
        elif has_right and not has_center:
            right_prominence_deltas.append(sum(right_proms) / len(right_proms))
            right_exclusives += 1

        # Count deltas
        left_count_deltas.append(len(left_proms) - len(center_proms))
        right_count_deltas.append(len(right_proms) - len(center_proms))

        # Omissions
        if has_center and not has_left:
            left_omissions += 1
        if has_center and not has_right:
            right_omissions += 1

        # Deep burial
        if has_center:
            avg_center_prom = sum(center_proms) / len(center_proms)
            for i, d in enumerate(left_d):
                if d >= DEEP_THRESHOLD:
                    left_deep_deltas.append(left_proms[i] - avg_center_prom)
            for i, d in enumerate(right_d):
                if d >= DEEP_THRESHOLD:
                    right_deep_deltas.append(right_proms[i] - avg_center_prom)

    # Compute statistics
    def safe_stat(arr, func, default=0.0):
        return func(arr) if arr else default

    left_features = [
        safe_stat(left_prominence_deltas, max),  # 1. Max
        safe_stat(left_prominence_deltas, min),  # 2. Min
        safe_stat(left_prominence_deltas, np.mean),  # 3. Avg
        safe_stat(left_prominence_deltas, np.std)
        if len(left_prominence_deltas) > 1
        else 0.0,  # 4. Std
        safe_stat(left_count_deltas, np.mean),  # 5. Avg repetition
        (np.var(left_depths) if len(left_depths) > 1 else 0.0)
        - (
            np.var(center_depths) if len(center_depths) > 1 else 0.0
        ),  # 6. Depth var delta
        (np.mean(left_depths) if left_depths else 0.0)
        - (np.mean(center_depths) if center_depths else 0.0),  # 7. Avg depth delta
        float(left_omissions),  # 8. Omissions
        float(left_exclusives),  # 9. Exclusives
        safe_stat(left_deep_deltas, np.mean),  # 10. Deep burial
    ]

    right_features = [
        safe_stat(right_prominence_deltas, max),
        safe_stat(right_prominence_deltas, min),
        safe_stat(right_prominence_deltas, np.mean),
        safe_stat(right_prominence_deltas, np.std)
        if len(right_prominence_deltas) > 1
        else 0.0,
        safe_stat(right_count_deltas, np.mean),
        (np.var(right_depths) if len(right_depths) > 1 else 0.0)
        - (np.var(center_depths) if len(center_depths) > 1 else 0.0),
        (np.mean(right_depths) if right_depths else 0.0)
        - (np.mean(center_depths) if center_depths else 0.0),
        float(right_omissions),
        float(right_exclusives),
        safe_stat(right_deep_deltas, np.mean),
    ]

    debug_info = {
        "num_features": 10,
        "feature_names": [
            "max_prom_delta",
            "min_prom_delta",
            "avg_prom_delta",
            "std_prom_delta",
            "avg_rep_delta",
            "depth_var_delta",
            "avg_depth_delta",
            "omissions",
            "exclusives",
            "deep_burial_delta",
        ],
    }

    return left_features, right_features, debug_info


def build_structural_stats_role_aware(
    fact_row: Dict,
) -> Tuple[List[float], List[float], Dict]:
    """
    Role-aware structural statistics that use nucleus/satellite information.

    Features:
    1-5: Basic stats (same as basic)
    6. Nucleus Count Delta - difference in nucleus EDUs
    7. Satellite Count Delta - difference in satellite EDUs
    8. Nucleus/Satellite Ratio Delta
    9. Avg Satellite Edge Count Delta
    10. Max Satellite Edge Count
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    left_prominence_deltas = []
    right_prominence_deltas = []
    left_count_deltas = []
    right_count_deltas = []
    left_depths = []
    right_depths = []
    center_depths = []
    left_omissions = 0
    right_omissions = 0
    left_deep_deltas = []
    right_deep_deltas = []

    # Role-specific counts
    left_nucleus = 0
    left_satellite = 0
    center_nucleus = 0
    center_satellite = 0
    right_nucleus = 0
    right_satellite = 0

    # Satellite edge counts
    left_sat_edges = []
    center_sat_edges = []
    right_sat_edges = []

    DEEP_THRESHOLD = 10

    for cluster_id, edus in clusters.items():
        left_proms = []
        center_proms = []
        right_proms = []
        left_d = []
        center_d = []
        right_d = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            depth = meta.get("depth", 0)
            role = meta.get("role", "N")
            sat_edges = meta.get("satellite_edges_to_root", 0)
            prom = get_edu_prominence(meta)

            if bias == "left":
                left_proms.append(prom)
                left_d.append(depth)
                left_depths.append(depth)
                left_sat_edges.append(sat_edges)
                if role == "N":
                    left_nucleus += 1
                else:
                    left_satellite += 1
            elif bias == "center":
                center_proms.append(prom)
                center_d.append(depth)
                center_depths.append(depth)
                center_sat_edges.append(sat_edges)
                if role == "N":
                    center_nucleus += 1
                else:
                    center_satellite += 1
            elif bias == "right":
                right_proms.append(prom)
                right_d.append(depth)
                right_depths.append(depth)
                right_sat_edges.append(sat_edges)
                if role == "N":
                    right_nucleus += 1
                else:
                    right_satellite += 1

        has_left = len(left_proms) > 0
        has_center = len(center_proms) > 0
        has_right = len(right_proms) > 0

        if has_left and has_center:
            avg_left = sum(left_proms) / len(left_proms)
            avg_center = sum(center_proms) / len(center_proms)
            left_prominence_deltas.append(avg_left - avg_center)

        if has_right and has_center:
            avg_right = sum(right_proms) / len(right_proms)
            avg_center = sum(center_proms) / len(center_proms)
            right_prominence_deltas.append(avg_right - avg_center)

        left_count_deltas.append(len(left_proms) - len(center_proms))
        right_count_deltas.append(len(right_proms) - len(center_proms))

        if has_center and not has_left:
            left_omissions += 1
        if has_center and not has_right:
            right_omissions += 1

        if has_center:
            avg_center_prom = sum(center_proms) / len(center_proms)
            for i, d in enumerate(left_d):
                if d >= DEEP_THRESHOLD:
                    left_deep_deltas.append(left_proms[i] - avg_center_prom)
            for i, d in enumerate(right_d):
                if d >= DEEP_THRESHOLD:
                    right_deep_deltas.append(right_proms[i] - avg_center_prom)

    # Compute statistics
    def safe_stat(arr, func, default=0.0):
        return func(arr) if arr else default

    # Nucleus/Satellite ratios
    def safe_ratio(num, denom, default=0.0):
        return num / denom if denom > 0 else default

    left_ns_ratio = safe_ratio(left_nucleus, left_nucleus + left_satellite)
    center_ns_ratio = safe_ratio(center_nucleus, center_nucleus + center_satellite)
    right_ns_ratio = safe_ratio(right_nucleus, right_nucleus + right_satellite)

    left_features = [
        safe_stat(left_prominence_deltas, max),  # 1. Max prom delta
        safe_stat(left_count_deltas, np.mean),  # 2. Avg repetition
        (np.var(left_depths) if len(left_depths) > 1 else 0.0)
        - (
            np.var(center_depths) if len(center_depths) > 1 else 0.0
        ),  # 3. Depth var delta
        float(left_omissions),  # 4. Omissions
        safe_stat(left_deep_deltas, np.mean),  # 5. Deep burial
        float(left_nucleus - center_nucleus),  # 6. Nucleus count delta
        float(left_satellite - center_satellite),  # 7. Satellite count delta
        left_ns_ratio - center_ns_ratio,  # 8. N/S ratio delta
        safe_stat(left_sat_edges, np.mean)
        - safe_stat(center_sat_edges, np.mean),  # 9. Avg sat edge delta
        safe_stat(left_sat_edges, max),  # 10. Max sat edges
    ]

    right_features = [
        safe_stat(right_prominence_deltas, max),
        safe_stat(right_count_deltas, np.mean),
        (np.var(right_depths) if len(right_depths) > 1 else 0.0)
        - (np.var(center_depths) if len(center_depths) > 1 else 0.0),
        float(right_omissions),
        safe_stat(right_deep_deltas, np.mean),
        float(right_nucleus - center_nucleus),
        float(right_satellite - center_satellite),
        right_ns_ratio - center_ns_ratio,
        safe_stat(right_sat_edges, np.mean) - safe_stat(center_sat_edges, np.mean),
        safe_stat(right_sat_edges, max),
    ]

    debug_info = {
        "num_features": 10,
        "feature_names": [
            "max_prom_delta",
            "avg_rep_delta",
            "depth_var_delta",
            "omissions",
            "deep_burial",
            "nucleus_delta",
            "satellite_delta",
            "ns_ratio_delta",
            "avg_sat_edge_delta",
            "max_sat_edges",
        ],
    }

    return left_features, right_features, debug_info


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


def evaluate(model, x: np.ndarray, y: np.ndarray) -> Dict:
    pred = model.predict(x)
    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def save_model(path: str, model, input_dim: int, experiment_name: str, config: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model,
        "input_dim": int(input_dim),
        "experiment": experiment_name,
        "config": config,
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
        script_subdir="universal-str/structural-stats",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "rf": rf_cfg,
        },
    )

    print("=" * 80)
    print("UNIVERSAL STRUCTURE: STRUCTURAL STATISTICS ARRAY")
    print("=" * 80)
    print("\nKey Insight: Aggregate structural patterns should distinguish bias.")
    print("Fixed-size feature vector captures overall treatment of facts.")
    print()
    print("Basic features (5):")
    print("  1. Max Prominence Delta - heavily promoted fact?")
    print("  2. Avg Repetition Delta - repetition pattern?")
    print("  3. Depth Variance Delta - structural shuffle?")
    print("  4. Omission Count - hidden facts?")
    print("  5. Avg Deep-Burial Delta - buried facts?")
    print()
    print("Reference baselines:")
    print("  - Unordered padded DFI + RF: 87.78%")
    print("  - Unordered bipartite coverage: 89.77%")
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

    # Define experiments - feature builders
    feature_experiments = [
        ("stats_basic_5", "Basic 5-feature stats", build_structural_stats_basic),
        (
            "stats_extended_10",
            "Extended 10-feature stats",
            build_structural_stats_extended,
        ),
        (
            "stats_role_aware",
            "Role-aware stats (nucleus/satellite)",
            build_structural_stats_role_aware,
        ),
    ]

    # Define model types to try
    model_types = ["RF", "LR", "SVM"]

    os.makedirs(args.model_dir, exist_ok=True)
    results = {}

    for feat_name, feat_desc, feature_builder in feature_experiments:
        print(f"\n{'=' * 80}")
        print(f"Feature Set: {feat_name}")
        print(f"Description: {feat_desc}")
        print("=" * 80)

        # Build features
        for row in facts_train:
            row["feat_left"], row["feat_right"], _ = feature_builder(row)
        for row in facts_val:
            row["feat_left"], row["feat_right"], _ = feature_builder(row)
        for row in facts_test:
            row["feat_left"], row["feat_right"], _ = feature_builder(row)

        # Build X, y
        x_train_raw, y_train = build_xy(facts_train, "feat_left", "feat_right")
        x_val_raw, y_val = build_xy(facts_val, "feat_left", "feat_right")
        x_test_raw, y_test = build_xy(facts_test, "feat_left", "feat_right")

        X_train = np.array(x_train_raw, dtype=float)
        X_val = np.array(x_val_raw, dtype=float)
        X_test = np.array(x_test_raw, dtype=float)

        input_dim = X_train.shape[1]
        print(f"  Input dimension: {input_dim}")

        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        # Try each model type
        for model_type in model_types:
            exp_name = f"{feat_name}_{model_type}"
            print(f"\n  Training {model_type}...")

            if model_type == "RF":
                model = RandomForestClassifier(
                    n_estimators=rf_cfg["n_estimators"],
                    max_depth=rf_cfg["max_depth"],
                    min_samples_split=rf_cfg["min_samples_split"],
                    min_samples_leaf=rf_cfg["min_samples_leaf"],
                    random_state=rf_cfg["seed"],
                    n_jobs=-1,
                )
            elif model_type == "LR":
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=rf_cfg["seed"],
                )
            elif model_type == "SVM":
                model = SVC(
                    kernel="rbf",
                    random_state=rf_cfg["seed"],
                )

            model.fit(X_trainval, y_trainval)

            metrics = {
                "train": evaluate(model, X_train, y_train),
                "val": evaluate(model, X_val, y_val),
                "test": evaluate(model, X_test, y_test),
                "trainval": evaluate(model, X_trainval, y_trainval),
            }

            # Feature importances (only for RF)
            if model_type == "RF":
                feature_importances = model.feature_importances_.tolist()
            elif model_type == "LR":
                feature_importances = model.coef_[0].tolist()
            else:
                feature_importances = []

            model_path = os.path.join(args.model_dir, f"{exp_name}.pkl")
            save_model(
                model_path,
                model,
                input_dim,
                exp_name,
                {
                    "feature_set": feat_name,
                    "model_type": model_type,
                    "rf": rf_cfg if model_type == "RF" else None,
                },
            )

            results[exp_name] = {
                "feature_set": feat_name,
                "feature_description": feat_desc,
                "model_type": model_type,
                "input_dim": input_dim,
                "metrics": metrics,
                "model_path": model_path,
                "feature_importances": feature_importances[:10]
                if feature_importances
                else [],
            }

            print(
                f"    Val:  acc={metrics['val']['accuracy']:.4f}, f1={metrics['val']['macro_f1']:.4f}"
            )
            print(
                f"    Test: acc={metrics['test']['accuracy']:.4f}, f1={metrics['test']['macro_f1']:.4f}"
            )

    # Reference baselines
    reference_unordered = 0.8778
    reference_bipartite = 0.8977

    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        exp_data["delta_vs_unordered_baseline"] = test_acc - reference_unordered
        exp_data["delta_vs_unordered_bipartite"] = test_acc - reference_bipartite

    output = {
        "setup": {
            "goal": "Test if aggregate structural statistics enable universal bias detection",
            "approach": "Fixed-size feature vector with summary statistics",
            "hypothesis": "Overall structural patterns distinguish biased from centrist coverage",
            "facts": args.facts,
            "split_dir": args.split_dir,
            "rf": rf_cfg,
            "created": datetime.now().isoformat(),
        },
        "reference_baselines": {
            "unordered_padded_dfi_rf": {"test_acc": reference_unordered},
            "unordered_bipartite_coverage": {"test_acc": reference_bipartite},
        },
        "experiments": results,
    }

    save_json(args.out, output)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY: Universal Structure - Structural Statistics")
    print("=" * 100)
    print(
        f"\n{'Experiment':<30} {'Model':<6} {'Dim':<6} {'Test Acc':<10} {'vs Baseline':<12} {'vs Best':<12}"
    )
    print("-" * 100)

    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        delta_baseline = exp_data["delta_vs_unordered_baseline"]
        delta_bipartite = exp_data["delta_vs_unordered_bipartite"]
        dim = exp_data["input_dim"]
        model_type = exp_data["model_type"]
        feat_set = exp_data["feature_set"]
        print(
            f"  {feat_set:<28} {model_type:<6} {dim:<6} {test_acc:<10.4f} {delta_baseline:<+12.4f} {delta_bipartite:<+12.4f}"
        )

    print("-" * 100)
    print(f"Reference: Unordered padded DFI + RF = {reference_unordered:.2%}")
    print(f"Reference: Unordered bipartite coverage = {reference_bipartite:.2%}")

    best_exp = max(results.items(), key=lambda x: x[1]["metrics"]["test"]["accuracy"])
    print(f"\nBest experiment: {best_exp[0]}")
    print(f"  Test accuracy: {best_exp[1]['metrics']['test']['accuracy']:.4f}")
    print(f"  vs Baseline: {best_exp[1]['delta_vs_unordered_baseline']:+.4f}")
    print(f"  vs Best: {best_exp[1]['delta_vs_unordered_bipartite']:+.4f}")

    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out, "experiments": list(results.keys())})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
