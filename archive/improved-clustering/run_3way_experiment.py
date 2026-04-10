"""
Experiment: Pure RST Framing Analysis on 3-Way Clusters

This script tests whether RST structural framing alone (without omission signal)
can predict media bias, using only clusters where all 3 biases are present.

Key insight: Since every cluster has exactly 1 EDU from each bias, the
"coverage" features are always 0 (no omission). This isolates the pure
structural framing signal.

Experiments:
1. Bipartite approach with min-depth ordering (finalized approach)
2. Original ordering (deterministic by cluster ID, for comparison)

Both use Random Forest classifier.

Expected outcome: Lower accuracy than full dataset (~89.77%) because
we've removed the omission signal which contributes ~72% of predictive power.
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Add parent directory for module imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from modules.run_logger import close_run_logging, init_run_logging, log_run_results


# Paths
DEFAULT_CLUSTERS_PATH = os.path.join(SCRIPT_DIR, "output/3way_clusters.json")
DEFAULT_SPLIT_DIR = os.path.join(PROJECT_ROOT, "data/valid_dfi_splits_recluster_gpu")
DEFAULT_OUT_PATH = os.path.join(SCRIPT_DIR, "output/3way_framing_results.json")
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "output/models")
RST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/rst_output")

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test pure RST framing on 3-way clusters (no omission signal)"
    )
    parser.add_argument("--clusters", default=DEFAULT_CLUSTERS_PATH)
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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
# RST Feature Enrichment
# =============================================================================


def roles(nuclearity):
    if nuclearity == "NS":
        return "N", "S"
    if nuclearity == "SN":
        return "S", "N"
    if nuclearity == "NN":
        return "N", "N"
    return None, None


def satellite_counts(edus, relations):
    parent_of = {}
    role_of = {}

    for rel in relations:
        parent = rel.get("parent")
        left = rel.get("left")
        right = rel.get("right")
        left_role, right_role = roles(rel.get("nuclearity"))

        if left is not None:
            parent_of[left] = parent
            role_of[left] = left_role
        if right is not None:
            parent_of[right] = parent
            role_of[right] = right_role

    sat_edges = {}
    local_role = {}

    for edu in edus:
        edu_id = edu["id"]
        local_role[edu_id] = role_of.get(edu_id)

        count = 0
        cur = edu_id
        seen = set()
        while cur in parent_of and cur not in seen:
            seen.add(cur)
            if role_of.get(cur) == "S":
                count += 1
            cur = parent_of[cur]

        sat_edges[edu_id] = count

    return sat_edges, local_role


def article_id_from_edu_id(edu_id: str) -> str:
    """Extract article ID from edu_id (format: articleId_eduLocalId)."""
    parts = edu_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else edu_id


def enrich_with_rst_features(cluster_result: dict) -> dict:
    """Add RST features (depth, role, satellite_edges_to_root) to edu_lookup."""
    triplet = cluster_result.get("triplet", {})
    clusters = cluster_result.get("clusters", {})
    edu_lookup = cluster_result.get("edu_lookup", {})

    # Collect all article IDs from edu_lookup
    article_ids = set()
    for edu_id in edu_lookup.keys():
        article_ids.add(article_id_from_edu_id(edu_id))

    # Load RST data for each article
    rst_data = {}
    for article_id in article_ids:
        rst_path = os.path.join(RST_OUTPUT_DIR, f"{article_id}.json")
        if os.path.exists(rst_path):
            with open(rst_path, "r", encoding="utf-8") as f:
                rst_data[article_id] = json.load(f)

    # Build enriched lookup
    enriched_lookup = {}
    for edu_id, edu_info in edu_lookup.items():
        article_id = article_id_from_edu_id(edu_id)
        local_id = int(edu_id.rsplit("_", 1)[1]) if "_" in edu_id else 0

        # Get RST features
        rst = rst_data.get(article_id, {})
        edus = rst.get("edus", [])
        relations = rst.get("relations", [])
        sat_edges, local_role = satellite_counts(edus, relations)

        # Find the EDU by local_id
        depth = 0
        role = None
        sat_count = 0
        for edu in edus:
            if edu.get("id") == local_id:
                depth = edu.get("depth", 0)
                role = local_role.get(local_id)
                sat_count = sat_edges.get(local_id, 0)
                break

        enriched_lookup[edu_id] = {
            "text": edu_info.get("text", ""),
            "bias": edu_info.get("bias", ""),
            "depth": depth,
            "role": role,
            "satellite_edges_to_root": sat_count,
        }

    return {
        "triplet_idx": cluster_result.get("triplet_idx"),
        "triplet": triplet,
        "clusters": clusters,
        "edu_lookup": enriched_lookup,
    }


# =============================================================================
# Prominence Functions
# =============================================================================


def W_log_depth(depth: int, sat_count: int) -> float:
    """Best-performing prominence formula: 1 / (1 + log(1 + depth))"""
    return 1.0 / (1.0 + math.log1p(depth))


def get_edu_prominence(edu_meta: Dict) -> float:
    if not edu_meta:
        return 0.0
    depth = edu_meta.get("depth", 0)
    sat_count = edu_meta.get("satellite_edges_to_root", 0)
    return W_log_depth(depth, sat_count)


def get_side_edus(edus: List[str], edu_lookup: Dict, side: str) -> List[Dict]:
    result = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            result.append(meta)
    return result


def get_side_scores(edus: List[str], edu_lookup: Dict, side: str) -> List[float]:
    scores = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            scores.append(get_edu_prominence(meta))
    return scores


# =============================================================================
# Ordering by Min Depth
# =============================================================================


def compute_max_depths_per_article(edu_lookup: Dict) -> Dict[str, int]:
    max_depths = {"left": 0, "center": 0, "right": 0}
    for edu_id, meta in edu_lookup.items():
        bias = meta.get("bias", "")
        depth = meta.get("depth", 0)
        if bias in max_depths:
            max_depths[bias] = max(max_depths[bias], depth)
    for bias in max_depths:
        if max_depths[bias] == 0:
            max_depths[bias] = 1
    return max_depths


def get_normalized_depth(edu_meta: Dict, max_depths: Dict[str, int]) -> float:
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
    return min(normalized_depths)


def order_clusters_by_depth(
    clusters: Dict, edu_lookup: Dict, agg_method: str = "min"
) -> OrderedDict:
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


def keep_original_order(clusters: Dict, edu_lookup: Dict) -> OrderedDict:
    """Keep original cluster ID order (deterministic)."""
    sorted_cluster_ids = sorted(clusters.keys(), key=lambda cid: int(cid))
    ordered = OrderedDict()
    for cid in sorted_cluster_ids:
        ordered[cid] = clusters[cid]
    return ordered


# =============================================================================
# Bipartite Approach (Finalized)
# =============================================================================


def greedy_bipartite_match(
    left_edus: List[Dict],
    center_edus: List[Dict],
    right_edus: List[Dict],
) -> Tuple[List[Tuple], List[Dict], List[Dict], List[Dict]]:
    """Greedy 1-to-1-to-1 matching of EDUs across the three sides."""
    left_pool = list(left_edus)
    center_pool = list(center_edus)
    right_pool = list(right_edus)

    matched_triplets = []

    while left_pool and center_pool and right_pool:
        best_score = -1
        best_triplet = None
        best_indices = None

        for i, l_edu in enumerate(left_pool):
            for j, c_edu in enumerate(center_pool):
                for k, r_edu in enumerate(right_pool):
                    depth_l = l_edu.get("depth", 0)
                    depth_c = c_edu.get("depth", 0)
                    depth_r = r_edu.get("depth", 0)

                    sim_lc = 1.0 / (1.0 + abs(depth_l - depth_c))
                    sim_lr = 1.0 / (1.0 + abs(depth_l - depth_r))
                    sim_cr = 1.0 / (1.0 + abs(depth_c - depth_r))
                    score = (sim_lc + sim_lr + sim_cr) / 3.0

                    if score > best_score:
                        best_score = score
                        best_triplet = (l_edu, c_edu, r_edu)
                        best_indices = (i, j, k)

        if best_triplet:
            matched_triplets.append(best_triplet)
            left_pool.pop(best_indices[0])
            center_pool.pop(best_indices[1])
            right_pool.pop(best_indices[2])

    return matched_triplets, left_pool, center_pool, right_pool


def build_bipartite_coverage(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Bipartite decomposition with coverage features.

    For 3-way clusters: Each cluster has exactly 1 EDU from each side.
    After matching, there are no leftovers - all coverage deltas are 0.
    The only signal comes from structural differences (prominence).
    """
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in ordered_clusters.items():
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        matched, leftover_l, leftover_c, leftover_r = greedy_bipartite_match(
            left_edus, center_edus, right_edus
        )

        # For 3-way clusters with 1 EDU each: always 1 matched, 0 leftovers
        # Coverage deltas are all 0 - signal comes only from ordering
        for _ in matched:
            features_left.append(0)
            features_right.append(0)

        # These should be empty for 3-way clusters
        for _ in leftover_l:
            features_left.append(1)
        for _ in leftover_c:
            features_left.append(-1)
            features_right.append(-1)
        for _ in leftover_r:
            features_right.append(1)

    return features_left, features_right


def build_bipartite_structural(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Bipartite with pure structural (prominence) features.

    For each matched triplet, compute prominence delta:
    delta_left = prominence(left) - prominence(center)
    delta_right = prominence(right) - prominence(center)
    """
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in ordered_clusters.items():
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        matched, leftover_l, leftover_c, leftover_r = greedy_bipartite_match(
            left_edus, center_edus, right_edus
        )

        # For matched triplets: compute prominence delta
        for l_edu, c_edu, r_edu in matched:
            prom_l = get_edu_prominence(l_edu)
            prom_c = get_edu_prominence(c_edu)
            prom_r = get_edu_prominence(r_edu)

            features_left.append(prom_l - prom_c)
            features_right.append(prom_r - prom_c)

        # Leftovers (should be empty for 3-way clusters)
        for edu in leftover_l:
            features_left.append(get_edu_prominence(edu))
        for edu in leftover_c:
            features_left.append(-get_edu_prominence(edu))
            features_right.append(-get_edu_prominence(edu))
        for edu in leftover_r:
            features_right.append(get_edu_prominence(edu))

    return features_left, features_right


def build_bipartite_combined(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Bipartite with combined coverage + structural features.
    For 3-way clusters, coverage is always 0, so this tests if
    combining the explicit 0 with structural helps.
    """
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in ordered_clusters.items():
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        matched, leftover_l, leftover_c, leftover_r = greedy_bipartite_match(
            left_edus, center_edus, right_edus
        )

        for l_edu, c_edu, r_edu in matched:
            # Coverage: all present = 0
            cov_left = 0
            cov_right = 0

            # Structural: prominence delta
            prom_l = get_edu_prominence(l_edu)
            prom_c = get_edu_prominence(c_edu)
            prom_r = get_edu_prominence(r_edu)

            str_left = prom_l - prom_c
            str_right = prom_r - prom_c

            features_left.extend([cov_left, str_left])
            features_right.extend([cov_right, str_right])

        # Leftovers
        for edu in leftover_l:
            features_left.extend([1, get_edu_prominence(edu)])
        for edu in leftover_c:
            features_left.extend([-1, -get_edu_prominence(edu)])
            features_right.extend([-1, -get_edu_prominence(edu)])
        for edu in leftover_r:
            features_right.extend([1, get_edu_prominence(edu)])

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
    ordering_fn: Callable,
    rf_cfg: Dict,
    experiment_name: str,
) -> Tuple[object, int, Dict, List[float]]:
    """Run a single experiment."""

    # Build features
    for row in train_rows:
        ordered = ordering_fn(row.get("clusters", {}), row.get("edu_lookup", {}))
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)
    for row in val_rows:
        ordered = ordering_fn(row.get("clusters", {}), row.get("edu_lookup", {}))
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)
    for row in test_rows:
        ordered = ordering_fn(row.get("clusters", {}), row.get("edu_lookup", {}))
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)

    # Build X, y
    x_train_raw, y_train = build_xy(train_rows, "feat_left", "feat_right")
    x_val_raw, y_val = build_xy(val_rows, "feat_left", "feat_right")
    x_test_raw, y_test = build_xy(test_rows, "feat_left", "feat_right")

    max_len = max((len(v) for v in x_train_raw), default=0)
    if max_len == 0:
        raise RuntimeError("No features generated")

    X_train = pad_or_truncate(x_train_raw, max_len)
    X_val = pad_or_truncate(x_val_raw, max_len)
    X_test = pad_or_truncate(x_test_raw, max_len)

    input_dim = X_train.shape[1]
    print(f"  Input dimension: {input_dim}")

    # Combine train+val
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

    feature_importances = model.feature_importances_.tolist()
    return model, input_dim, metrics, feature_importances


def save_model(path: str, model, input_dim: int, experiment_name: str, rf_cfg: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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
        script_subdir="improved-clustering/3way_framing",
        hyperparams={
            "clusters": args.clusters,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "rf": rf_cfg,
        },
    )

    print("=" * 80)
    print("EXPERIMENT: Pure RST Framing on 3-Way Clusters")
    print("=" * 80)
    print("\nGoal: Test if RST structural framing alone can predict bias")
    print("(without omission signal, since all clusters have all 3 biases)")
    print()
    print("Reference baselines (full dataset with omission):")
    print("  - Full dataset (bipartite coverage): 89.77%")
    print("  - RST-only (full coverage clusters): 61.16%")
    print()

    # Load clusters
    print(f"Loading 3-way clusters from {args.clusters}...")
    clusters_raw = load_json(args.clusters)
    print(f"Loaded {len(clusters_raw)} triplet results")

    # Enrich with RST features
    print("Enriching with RST features...")
    clusters_enriched = []
    for i, cr in enumerate(clusters_raw):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(clusters_raw)}")
        clusters_enriched.append(enrich_with_rst_features(cr))
    print(f"Enriched {len(clusters_enriched)} triplets")

    # Load splits
    print("\nLoading train/val/test splits...")
    train_split = load_json(os.path.join(args.split_dir, "train.json"))
    val_split = load_json(os.path.join(args.split_dir, "val.json"))
    test_split = load_json(os.path.join(args.split_dir, "test.json"))

    train_ids = get_split_triplet_idx(train_split)
    val_ids = get_split_triplet_idx(val_split)
    test_ids = get_split_triplet_idx(test_split)

    facts_train, facts_val, facts_test = split_facts_by_existing_splits(
        clusters_enriched, train_ids, val_ids, test_ids
    )

    print(
        f"Data: train={len(facts_train)}, val={len(facts_val)}, test={len(facts_test)}"
    )

    # Statistics
    total_clusters = sum(len(r.get("clusters", {})) for r in clusters_enriched)
    avg_clusters = total_clusters / len(clusters_enriched) if clusters_enriched else 0
    print(f"\nTotal 3-way clusters: {total_clusters}")
    print(f"Average clusters per triplet: {avg_clusters:.2f}")

    # Define experiments
    experiments = [
        # Min-depth ordering (finalized approach)
        (
            "bipartite_structural_min_ordered",
            "Bipartite structural (min-depth ordered)",
            build_bipartite_structural,
            lambda c, l: order_clusters_by_depth(c, l, "min"),
        ),
        (
            "bipartite_combined_min_ordered",
            "Bipartite combined cov+struct (min-depth ordered)",
            build_bipartite_combined,
            lambda c, l: order_clusters_by_depth(c, l, "min"),
        ),
        # Original ordering (deterministic by cluster ID)
        (
            "bipartite_structural_original",
            "Bipartite structural (original order)",
            build_bipartite_structural,
            keep_original_order,
        ),
        (
            "bipartite_combined_original",
            "Bipartite combined cov+struct (original order)",
            build_bipartite_combined,
            keep_original_order,
        ),
        # Coverage only (should be ~50% - no signal)
        (
            "bipartite_coverage_min_ordered",
            "Bipartite coverage only (min-depth ordered)",
            build_bipartite_coverage,
            lambda c, l: order_clusters_by_depth(c, l, "min"),
        ),
    ]

    os.makedirs(args.model_dir, exist_ok=True)
    results = {}

    for exp_name, description, feature_builder, ordering_fn in experiments:
        print(f"\n{'=' * 80}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {description}")
        print("=" * 80)

        train_rows = copy.deepcopy(facts_train)
        val_rows = copy.deepcopy(facts_val)
        test_rows = copy.deepcopy(facts_test)

        try:
            model, input_dim, metrics, feature_importances = (
                train_and_evaluate_experiment(
                    train_rows,
                    val_rows,
                    test_rows,
                    feature_builder,
                    ordering_fn,
                    rf_cfg,
                    exp_name,
                )
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

        except Exception as e:
            print(f"  ERROR: {e}")
            results[exp_name] = {"error": str(e)}

    # Reference baselines
    ref_full_coverage = 0.8977
    ref_rst_only = 0.6116

    # Compute deltas
    for exp_name, exp_data in results.items():
        if "metrics" in exp_data:
            test_acc = exp_data["metrics"]["test"]["accuracy"]
            exp_data["delta_vs_full_coverage"] = test_acc - ref_full_coverage
            exp_data["delta_vs_rst_only"] = test_acc - ref_rst_only

    output = {
        "setup": {
            "goal": "Test pure RST framing on 3-way clusters (no omission signal)",
            "description": (
                "All clusters have exactly 1 EDU from each bias (left, center, right). "
                "Coverage features are always 0. Tests if structural framing alone predicts bias."
            ),
            "clusters_file": args.clusters,
            "split_dir": args.split_dir,
            "rf": rf_cfg,
            "created": datetime.now().isoformat(),
        },
        "data_stats": {
            "triplets_with_clusters": len(clusters_enriched),
            "total_3way_clusters": total_clusters,
            "avg_clusters_per_triplet": round(avg_clusters, 2),
            "train_triplets": len(facts_train),
            "val_triplets": len(facts_val),
            "test_triplets": len(facts_test),
        },
        "reference_baselines": {
            "full_dataset_bipartite_coverage": {
                "test_acc": ref_full_coverage,
                "note": "Has omission signal (~72% contribution)",
            },
            "rst_only_full_coverage": {
                "test_acc": ref_rst_only,
                "note": "RST features only, still has some coverage variance",
            },
        },
        "experiments": results,
    }

    save_json(args.out, output)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Pure RST Framing on 3-Way Clusters")
    print("=" * 100)
    print(
        f"\n{'Experiment':<45} {'Dim':<8} {'Test Acc':<12} {'vs Full':<12} {'vs RST-only':<12}"
    )
    print("-" * 100)

    for exp_name, exp_data in results.items():
        if "metrics" in exp_data:
            test_acc = exp_data["metrics"]["test"]["accuracy"]
            delta_full = exp_data.get("delta_vs_full_coverage", 0)
            delta_rst = exp_data.get("delta_vs_rst_only", 0)
            dim = exp_data["input_dim"]
            print(
                f"  {exp_name:<43} {dim:<8} {test_acc:<12.4f} {delta_full:<+12.4f} {delta_rst:<+12.4f}"
            )
        else:
            print(f"  {exp_name:<43} ERROR")

    print("-" * 100)
    print(f"Reference: Full dataset (bipartite coverage) = {ref_full_coverage:.2%}")
    print(f"Reference: RST-only (full coverage clusters) = {ref_rst_only:.2%}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Get best result
    valid_results = {k: v for k, v in results.items() if "metrics" in v}
    if valid_results:
        best_exp = max(
            valid_results.items(), key=lambda x: x[1]["metrics"]["test"]["accuracy"]
        )
        best_acc = best_exp[1]["metrics"]["test"]["accuracy"]

        print(f"\nBest experiment: {best_exp[0]}")
        print(f"Test accuracy: {best_acc:.4f} ({best_acc:.2%})")

        if best_acc < 0.55:
            print(
                "\nConclusion: RST structural framing alone provides minimal predictive power"
            )
            print("when all clusters have all 3 biases (no omission signal).")
            print(
                "This confirms that FACT OMISSION is the dominant predictor of media bias."
            )
        elif best_acc < 0.65:
            print(
                "\nConclusion: RST structural framing provides weak but measurable signal"
            )
            print("when omission is removed. Omission remains the dominant predictor.")
        else:
            print(
                "\nConclusion: RST structural framing provides moderate predictive power"
            )
            print("even without omission signal.")

    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out, "experiments": list(results.keys())})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
