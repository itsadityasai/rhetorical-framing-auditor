"""
Universal Structure Approach A: Group by RST Depth Bins

Instead of organizing features by Fact ID (which changes per triplet), this approach
organizes features by STRUCTURAL POSITION in the RST tree. This allows the model
to learn universal patterns about how biased outlets position facts at different
structural depths.

Feature Vector Structure:
- Feature 0: Average Δ of facts at Depth 0 (Root/Headline level)
- Feature 1: Average Δ of facts at Depth 1
- Feature 2: Average Δ of facts at Depth 2
- ...
- Feature N-1: Average Δ of facts at Depth N-1
- Feature N: Total number of omitted facts (facts in center but not in side)

This directly tests the core hypothesis: biased outlets systematically position
aligned facts at more prominent (shallow) depths while burying uncomfortable
facts in deeper (satellite) positions.

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
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

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
DEFAULT_OUT_PATH = "archive/universal-str/by-depth-bins/results/depth_bins_results.json"
DEFAULT_MODEL_DIR = "archive/universal-str/by-depth-bins/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Structure: Group facts by RST depth bins"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum depth bin (depths >= this are grouped together)",
    )
    parser.add_argument(
        "--use-normalized",
        action="store_true",
        help="Use normalized depth (depth / max_depth_in_article)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of bins for normalized depth (only used with --use-normalized)",
    )

    # Random Forest hyperparameters
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=15)
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
# Depth Bin Feature Construction
# =============================================================================


def get_max_depth_per_article(edu_lookup: Dict) -> Dict[str, int]:
    """Get maximum depth for each article (by bias)."""
    max_depths = {"left": 0, "center": 0, "right": 0}
    for edu_id, meta in edu_lookup.items():
        bias = meta.get("bias", "")
        depth = meta.get("depth", 0)
        if bias in max_depths:
            max_depths[bias] = max(max_depths[bias], depth)
    # Avoid division by zero
    for bias in max_depths:
        if max_depths[bias] == 0:
            max_depths[bias] = 1
    return max_depths


def build_depth_bin_features_absolute(
    fact_row: Dict, max_depth_bin: int = 20
) -> Tuple[List[float], List[float], Dict]:
    """
    Build features where each index represents a specific RST depth.

    For each depth bin d (0 to max_depth_bin-1):
    - Collect all clusters that have EDUs at depth d
    - For each such cluster, compute prominence delta (side - center)
    - Average these deltas for the depth bin

    Final feature: [avg_delta_depth0, avg_delta_depth1, ..., omission_count]

    Returns:
        (left_features, right_features, debug_info)
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    # Initialize depth bins for deltas
    left_deltas_by_depth = defaultdict(list)
    right_deltas_by_depth = defaultdict(list)

    # Track omissions (facts in center but not in side)
    left_omissions = 0
    right_omissions = 0

    for cluster_id, edus in clusters.items():
        # Separate EDUs by bias
        left_edus = []
        center_edus = []
        right_edus = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            if bias == "left":
                left_edus.append(meta)
            elif bias == "center":
                center_edus.append(meta)
            elif bias == "right":
                right_edus.append(meta)

        # Calculate center baseline (average prominence across all center EDUs in cluster)
        center_prominence = 0.0
        if center_edus:
            center_prominence = sum(get_edu_prominence(e) for e in center_edus) / len(
                center_edus
            )

        # Check for omissions
        has_center = len(center_edus) > 0
        has_left = len(left_edus) > 0
        has_right = len(right_edus) > 0

        if has_center and not has_left:
            left_omissions += 1
        if has_center and not has_right:
            right_omissions += 1

        # For each left EDU, compute delta and assign to its depth bin
        for edu_meta in left_edus:
            depth = min(edu_meta.get("depth", 0), max_depth_bin - 1)
            prominence = get_edu_prominence(edu_meta)
            delta = prominence - center_prominence
            left_deltas_by_depth[depth].append(delta)

        # For each right EDU, compute delta and assign to its depth bin
        for edu_meta in right_edus:
            depth = min(edu_meta.get("depth", 0), max_depth_bin - 1)
            prominence = get_edu_prominence(edu_meta)
            delta = prominence - center_prominence
            right_deltas_by_depth[depth].append(delta)

    # Build feature vectors
    left_features = []
    right_features = []

    for d in range(max_depth_bin):
        # Average delta at this depth (0 if no facts at this depth)
        if left_deltas_by_depth[d]:
            left_features.append(
                sum(left_deltas_by_depth[d]) / len(left_deltas_by_depth[d])
            )
        else:
            left_features.append(0.0)

        if right_deltas_by_depth[d]:
            right_features.append(
                sum(right_deltas_by_depth[d]) / len(right_deltas_by_depth[d])
            )
        else:
            right_features.append(0.0)

    # Add omission count as final feature
    left_features.append(float(left_omissions))
    right_features.append(float(right_omissions))

    debug_info = {
        "left_omissions": left_omissions,
        "right_omissions": right_omissions,
        "left_depths_with_data": [
            d for d in range(max_depth_bin) if left_deltas_by_depth[d]
        ],
        "right_depths_with_data": [
            d for d in range(max_depth_bin) if right_deltas_by_depth[d]
        ],
    }

    return left_features, right_features, debug_info


def build_depth_bin_features_normalized(
    fact_row: Dict, num_bins: int = 10
) -> Tuple[List[float], List[float], Dict]:
    """
    Build features using normalized depth bins.

    Normalized depth = depth / max_depth_in_article
    Bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]

    This handles varying article lengths/tree heights.
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    # Get max depths for normalization
    max_depths = get_max_depth_per_article(edu_lookup)

    # Initialize bins
    left_deltas_by_bin = defaultdict(list)
    right_deltas_by_bin = defaultdict(list)

    left_omissions = 0
    right_omissions = 0

    for cluster_id, edus in clusters.items():
        left_edus = []
        center_edus = []
        right_edus = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            if bias == "left":
                left_edus.append(meta)
            elif bias == "center":
                center_edus.append(meta)
            elif bias == "right":
                right_edus.append(meta)

        center_prominence = 0.0
        if center_edus:
            center_prominence = sum(get_edu_prominence(e) for e in center_edus) / len(
                center_edus
            )

        has_center = len(center_edus) > 0
        has_left = len(left_edus) > 0
        has_right = len(right_edus) > 0

        if has_center and not has_left:
            left_omissions += 1
        if has_center and not has_right:
            right_omissions += 1

        for edu_meta in left_edus:
            depth = edu_meta.get("depth", 0)
            max_d = max_depths.get("left", 1)
            norm_depth = depth / max_d
            bin_idx = min(int(norm_depth * num_bins), num_bins - 1)
            prominence = get_edu_prominence(edu_meta)
            delta = prominence - center_prominence
            left_deltas_by_bin[bin_idx].append(delta)

        for edu_meta in right_edus:
            depth = edu_meta.get("depth", 0)
            max_d = max_depths.get("right", 1)
            norm_depth = depth / max_d
            bin_idx = min(int(norm_depth * num_bins), num_bins - 1)
            prominence = get_edu_prominence(edu_meta)
            delta = prominence - center_prominence
            right_deltas_by_bin[bin_idx].append(delta)

    left_features = []
    right_features = []

    for b in range(num_bins):
        if left_deltas_by_bin[b]:
            left_features.append(
                sum(left_deltas_by_bin[b]) / len(left_deltas_by_bin[b])
            )
        else:
            left_features.append(0.0)

        if right_deltas_by_bin[b]:
            right_features.append(
                sum(right_deltas_by_bin[b]) / len(right_deltas_by_bin[b])
            )
        else:
            right_features.append(0.0)

    left_features.append(float(left_omissions))
    right_features.append(float(right_omissions))

    debug_info = {
        "left_omissions": left_omissions,
        "right_omissions": right_omissions,
        "normalization": "depth / max_depth_in_article",
    }

    return left_features, right_features, debug_info


def build_depth_bin_features_with_counts(
    fact_row: Dict, max_depth_bin: int = 20
) -> Tuple[List[float], List[float], Dict]:
    """
    Enhanced depth bin features with both avg delta AND count at each depth.

    Feature vector: [delta_d0, count_d0, delta_d1, count_d1, ..., omissions]

    This captures both WHERE facts are placed AND how many facts at each depth.
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    left_deltas_by_depth = defaultdict(list)
    right_deltas_by_depth = defaultdict(list)

    left_omissions = 0
    right_omissions = 0

    for cluster_id, edus in clusters.items():
        left_edus = []
        center_edus = []
        right_edus = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            if bias == "left":
                left_edus.append(meta)
            elif bias == "center":
                center_edus.append(meta)
            elif bias == "right":
                right_edus.append(meta)

        center_prominence = 0.0
        if center_edus:
            center_prominence = sum(get_edu_prominence(e) for e in center_edus) / len(
                center_edus
            )

        has_center = len(center_edus) > 0
        if has_center and not left_edus:
            left_omissions += 1
        if has_center and not right_edus:
            right_omissions += 1

        for edu_meta in left_edus:
            depth = min(edu_meta.get("depth", 0), max_depth_bin - 1)
            prominence = get_edu_prominence(edu_meta)
            delta = prominence - center_prominence
            left_deltas_by_depth[depth].append(delta)

        for edu_meta in right_edus:
            depth = min(edu_meta.get("depth", 0), max_depth_bin - 1)
            prominence = get_edu_prominence(edu_meta)
            delta = prominence - center_prominence
            right_deltas_by_depth[depth].append(delta)

    left_features = []
    right_features = []

    for d in range(max_depth_bin):
        # Average delta
        if left_deltas_by_depth[d]:
            left_features.append(
                sum(left_deltas_by_depth[d]) / len(left_deltas_by_depth[d])
            )
        else:
            left_features.append(0.0)
        # Count at this depth
        left_features.append(float(len(left_deltas_by_depth[d])))

        if right_deltas_by_depth[d]:
            right_features.append(
                sum(right_deltas_by_depth[d]) / len(right_deltas_by_depth[d])
            )
        else:
            right_features.append(0.0)
        right_features.append(float(len(right_deltas_by_depth[d])))

    left_features.append(float(left_omissions))
    right_features.append(float(right_omissions))

    debug_info = {
        "left_omissions": left_omissions,
        "right_omissions": right_omissions,
        "feature_structure": "delta_d0, count_d0, delta_d1, count_d1, ..., omissions",
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
        "max_depth": args.rf_max_depth if args.rf_max_depth != 0 else None,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "seed": args.seed,
    }

    run_log = init_run_logging(
        script_subdir="universal-str/by-depth-bins",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "max_depth_bin": args.max_depth,
            "use_normalized": args.use_normalized,
            "num_bins": args.num_bins,
            "rf": rf_cfg,
        },
    )

    print("=" * 80)
    print("UNIVERSAL STRUCTURE: GROUP BY RST DEPTH BINS")
    print("=" * 80)
    print("\nKey Insight: Organize features by STRUCTURAL POSITION, not Fact ID.")
    print("Each feature index represents a specific depth in the RST tree.")
    print("\nThis allows the model to learn universal patterns:")
    print("  - 'Facts at depth 0-2 tend to have positive delta in biased articles'")
    print("  - 'Facts at depth 10+ tend to have negative delta (buried)'")
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

    # Define experiments
    experiments = [
        (
            "depth_bins_absolute",
            "Absolute depth bins (0-19)",
            lambda row: build_depth_bin_features_absolute(row, args.max_depth),
        ),
        (
            "depth_bins_normalized",
            f"Normalized depth bins ({args.num_bins} bins)",
            lambda row: build_depth_bin_features_normalized(row, args.num_bins),
        ),
        (
            "depth_bins_with_counts",
            "Depth bins with counts",
            lambda row: build_depth_bin_features_with_counts(row, args.max_depth),
        ),
    ]

    os.makedirs(args.model_dir, exist_ok=True)
    results = {}

    for exp_name, description, feature_builder in experiments:
        print(f"\n{'=' * 80}")
        print(f"Experiment: {exp_name}")
        print(f"Description: {description}")
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

        # All vectors should have same length (fixed by design)
        X_train = np.array(x_train_raw, dtype=float)
        X_val = np.array(x_val_raw, dtype=float)
        X_test = np.array(x_test_raw, dtype=float)

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

        model_path = os.path.join(args.model_dir, f"{exp_name}.pkl")
        save_model(
            model_path,
            model,
            input_dim,
            exp_name,
            {
                "description": description,
                "rf": rf_cfg,
                "max_depth_bin": args.max_depth,
            },
        )

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

    # Reference baselines
    reference_unordered = 0.8778
    reference_bipartite = 0.8977

    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        exp_data["delta_vs_unordered_baseline"] = test_acc - reference_unordered
        exp_data["delta_vs_unordered_bipartite"] = test_acc - reference_bipartite

    output = {
        "setup": {
            "goal": "Test if organizing features by RST depth bins enables universal structure learning",
            "approach": "Each feature index represents a structural depth, not a fact ID",
            "hypothesis": "Model can learn 'facts at depth X have characteristic delta patterns'",
            "max_depth_bin": args.max_depth,
            "num_normalized_bins": args.num_bins,
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
    print("SUMMARY: Universal Structure - Depth Bins")
    print("=" * 100)
    print(
        f"\n{'Experiment':<35} {'Dim':<8} {'Test Acc':<12} {'vs Baseline':<15} {'vs Best':<15}"
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
