"""
Ordering Strategy 1: Order Clusters by Size (largest first)

This script tests whether imposing a consistent semantic ordering on clusters
improves classification performance. Clusters are ordered by total EDU count
(across all articles), so position 0 always represents the most-mentioned fact.

Hypothesis: If position 0 has 39.4% feature importance, making position 0
consistently represent the "most mentioned fact" across all triplets should
help the model learn more generalizable patterns.

Tests 3 DFI construction approaches with this ordering:
1. Cumulative Prominence (avg)
2. Distributional DFI (3D)
3. Bipartite Decomposition (coverage)

Reference baseline (unordered): 87.78% (padded DFI + RF)
Best alternative (unordered): 89.77% (bipartite coverage)
"""

import argparse
import copy
import json
import math
import os
import pickle
import sys
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Add project root for module imports
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "params.yaml").exists()),
    Path(__file__).resolve().parent,
)
sys.path.insert(0, str(PROJECT_ROOT))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = "archive/ordering-str/by-cluster-size/results/ordered_dfi_results.json"
DEFAULT_MODEL_DIR = "archive/ordering-str/by-cluster-size/results/models"

VALID_BIASES = {"left", "center", "right"}
ORDERING_STRATEGY = "by_cluster_size"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test DFI approaches with cluster ordering by size"
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
# ORDERING STRATEGY: By Cluster Size (largest first)
# =============================================================================


def order_clusters_by_size(clusters: Dict, edu_lookup: Dict) -> OrderedDict:
    """
    Order clusters by total EDU count (largest first).

    This ensures position 0 always corresponds to the most-mentioned fact
    across all triplets, making the positional encoding semantically meaningful.
    """
    # Calculate size for each cluster
    cluster_sizes = {}
    for cluster_id, edus in clusters.items():
        cluster_sizes[cluster_id] = len(edus)

    # Sort by size (descending), then by cluster_id for stability
    sorted_cluster_ids = sorted(
        clusters.keys(), key=lambda cid: (-cluster_sizes[cid], str(cid))
    )

    # Build ordered dict
    ordered = OrderedDict()
    for cid in sorted_cluster_ids:
        ordered[cid] = clusters[cid]

    return ordered


# =============================================================================
# Approach 1: Cumulative Prominence (avg)
# =============================================================================


def build_cumulative_avg(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Cumulative prominence using avg(scores) per cluster.
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


# =============================================================================
# Approach 2: Distributional DFI (3D)
# =============================================================================


def build_distributional_3d(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """
    Distributional DFI with 3 metrics per cluster: [max, sum, count]
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

        # SUM prominence
        sum_left = sum(scores_left) if scores_left else 0.0
        sum_center = sum(scores_center) if scores_center else 0.0
        sum_right = sum(scores_right) if scores_right else 0.0

        # COUNT (repetition frequency)
        count_left = len(scores_left)
        count_center = len(scores_center)
        count_right = len(scores_right)

        features_left.extend(
            [
                max_left - max_center,
                sum_left - sum_center,
                count_left - count_center,
            ]
        )

        features_right.extend(
            [
                max_right - max_center,
                sum_right - sum_center,
                count_right - count_center,
            ]
        )

    return features_left, features_right


# =============================================================================
# Approach 3: Bipartite Decomposition (coverage)
# =============================================================================


def greedy_bipartite_match(
    left_edus: List[Dict],
    center_edus: List[Dict],
    right_edus: List[Dict],
) -> Tuple[List[Tuple], List[Dict], List[Dict], List[Dict]]:
    """
    Greedy 1-to-1-to-1 matching of EDUs across the three sides.
    """
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
                    # Use inverse depth difference as similarity proxy
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
    Each matched/unmatched EDU contributes a coverage delta.
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

        # Matched triplets: all present, delta = 0
        for _ in matched:
            features_left.append(0)
            features_right.append(0)

        # Leftover left: left has, center doesn't
        for _ in leftover_l:
            features_left.append(1)

        # Leftover center: center has, sides don't
        for _ in leftover_c:
            features_left.append(-1)
            features_right.append(-1)

        # Leftover right: right has, center doesn't
        for _ in leftover_r:
            features_right.append(1)

    return features_left, features_right


# =============================================================================
# Baseline (unordered) for comparison
# =============================================================================


def build_baseline_combined(
    fact_row: Dict, ordered_clusters: OrderedDict
) -> Tuple[List[float], List[float]]:
    """BASELINE: Coverage + max prominence (unordered would use original cluster order)."""
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in ordered_clusters.items():
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

        features_left.extend([cov_left, str_left])
        features_right.extend([cov_right, str_right])

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

    # Build features with ordered clusters
    for row in train_rows:
        ordered = order_clusters_by_size(
            row.get("clusters", {}), row.get("edu_lookup", {})
        )
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)
    for row in val_rows:
        ordered = order_clusters_by_size(
            row.get("clusters", {}), row.get("edu_lookup", {})
        )
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)
    for row in test_rows:
        ordered = order_clusters_by_size(
            row.get("clusters", {}), row.get("edu_lookup", {})
        )
        row["feat_left"], row["feat_right"] = feature_builder(row, ordered)

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
        "ordering_strategy": ORDERING_STRATEGY,
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
        script_subdir="ordering-str/by-cluster-size",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "ordering_strategy": ORDERING_STRATEGY,
            "rf": rf_cfg,
        },
    )

    print("=" * 80)
    print("Ordering Strategy: BY CLUSTER SIZE (largest first)")
    print("=" * 80)
    print("\nHypothesis: Making position 0 consistently represent the 'most-mentioned'")
    print("fact across all triplets should improve generalization.")
    print()
    print("Reference baselines (unordered):")
    print("  - Padded DFI + RF: 87.78%")
    print("  - Bipartite coverage: 89.77%")
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
        (
            "baseline_combined_ordered",
            "Baseline: coverage + max (ordered)",
            build_baseline_combined,
        ),
        (
            "cumulative_avg_ordered",
            "Approach 1: Cumulative avg (ordered)",
            build_cumulative_avg,
        ),
        (
            "distributional_3d_ordered",
            "Approach 2: Distributional 3D (ordered)",
            build_distributional_3d,
        ),
        (
            "bipartite_coverage_ordered",
            "Approach 3: Bipartite coverage (ordered)",
            build_bipartite_coverage,
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
            "ordering_strategy": ORDERING_STRATEGY,
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

    # Reference baselines
    reference_unordered = 0.8778  # Padded DFI + RF
    reference_bipartite = 0.8977  # Bipartite coverage (unordered)

    # Compute deltas
    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        exp_data["delta_vs_unordered_baseline"] = test_acc - reference_unordered
        exp_data["delta_vs_unordered_bipartite"] = test_acc - reference_bipartite

    output = {
        "setup": {
            "goal": "Test if cluster ordering by size improves classification",
            "ordering_strategy": ORDERING_STRATEGY,
            "description": "Clusters ordered by total EDU count (largest first)",
            "hypothesis": "Position 0 = most-mentioned fact should improve generalization",
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

    # Summary table
    print("\n" + "=" * 100)
    print(f"SUMMARY: Ordering Strategy = {ORDERING_STRATEGY}")
    print("=" * 100)
    print(
        f"\n{'Experiment':<35} {'Dim':<8} {'Test Acc':<12} {'vs Unordered':<15} {'vs Best Unord':<15}"
    )
    print("-" * 100)

    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        delta_baseline = exp_data["delta_vs_unordered_baseline"]
        delta_bipartite = exp_data["delta_vs_unordered_bipartite"]
        dim = exp_data["input_dim"]
        print(
            f"  {exp_name:<33} {dim:<8} {test_acc:<12.4f} {delta_baseline:<+15.4f} {delta_bipartite:<+15.4f}"
        )

    print("-" * 100)
    print(f"Reference: Unordered padded DFI + RF = {reference_unordered:.2%}")
    print(f"Reference: Unordered bipartite coverage = {reference_bipartite:.2%}")

    # Find best experiment
    best_exp = max(results.items(), key=lambda x: x[1]["metrics"]["test"]["accuracy"])
    print(f"\nBest experiment: {best_exp[0]}")
    print(f"  Test accuracy: {best_exp[1]['metrics']['test']['accuracy']:.4f}")
    print(f"  vs Unordered baseline: {best_exp[1]['delta_vs_unordered_baseline']:+.4f}")
    print(
        f"  vs Unordered bipartite: {best_exp[1]['delta_vs_unordered_bipartite']:+.4f}"
    )

    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out, "experiments": list(results.keys())})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
