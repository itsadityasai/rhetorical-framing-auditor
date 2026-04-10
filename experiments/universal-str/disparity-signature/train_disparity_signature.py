"""
Universal Structure Approach B: Disparity Signature

Instead of organizing features by Fact ID (which changes per triplet), this approach
captures the "shape" or intensity distribution of bias by sorting deltas from
largest to smallest before creating the feature vector.

Feature Vector Structure:
- Sort all prominence deltas (side - center) from largest to smallest
- [Largest_Delta, 2nd_Largest_Delta, 3rd_Largest_Delta, ..., Smallest_Delta]
- Pad shorter vectors with zeros

Key Insight: The DISTRIBUTION of deltas is more informative than which specific
fact has which delta. A biased article might have:
- One heavily promoted fact (+0.5 delta)
- Two moderately promoted facts (+0.2, +0.1 delta)
- Three buried facts (-0.1, -0.2, -0.3 delta)

This "signature" pattern should be consistent across different topics.

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
DEFAULT_OUT_PATH = (
    "experiments/universal-str/disparity-signature/results/disparity_signature_results.json"
)
DEFAULT_MODEL_DIR = "experiments/universal-str/disparity-signature/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Structure: Disparity Signature (sorted deltas)"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--max-features",
        type=int,
        default=50,
        help="Maximum number of features (sorted deltas + padding)",
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
# Disparity Signature Feature Construction
# =============================================================================


def build_disparity_signature_basic(
    fact_row: Dict, max_features: int = 50
) -> Tuple[List[float], List[float], Dict]:
    """
    Build disparity signature: sorted deltas from largest to smallest.

    For each cluster:
    - Compute avg prominence for each side
    - Delta = side_prominence - center_prominence

    Sort all deltas descending, pad/truncate to max_features.
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    left_deltas = []
    right_deltas = []

    for cluster_id, edus in clusters.items():
        left_proms = []
        center_proms = []
        right_proms = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            prom = get_edu_prominence(meta)
            if bias == "left":
                left_proms.append(prom)
            elif bias == "center":
                center_proms.append(prom)
            elif bias == "right":
                right_proms.append(prom)

        # Average prominence per side
        avg_left = sum(left_proms) / len(left_proms) if left_proms else 0.0
        avg_center = sum(center_proms) / len(center_proms) if center_proms else 0.0
        avg_right = sum(right_proms) / len(right_proms) if right_proms else 0.0

        # Compute deltas
        delta_left = avg_left - avg_center
        delta_right = avg_right - avg_center

        # Only include if the side has any EDUs (non-zero when present)
        if left_proms:
            left_deltas.append(delta_left)
        if right_proms:
            right_deltas.append(delta_right)

    # Sort descending (largest positive delta first)
    left_deltas.sort(reverse=True)
    right_deltas.sort(reverse=True)

    # Pad/truncate to max_features
    left_features = (left_deltas + [0.0] * max_features)[:max_features]
    right_features = (right_deltas + [0.0] * max_features)[:max_features]

    debug_info = {
        "num_left_deltas": len(left_deltas),
        "num_right_deltas": len(right_deltas),
    }

    return left_features, right_features, debug_info


def build_disparity_signature_with_omissions(
    fact_row: Dict, max_features: int = 50
) -> Tuple[List[float], List[float], Dict]:
    """
    Disparity signature with special handling for omissions.

    - Omissions (center has fact, side doesn't) get delta = -1.0 (maximum penalty)
    - Exclusive facts (side has, center doesn't) get delta = +1.0 (maximum bonus)
    - Normal facts get computed delta
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    left_deltas = []
    right_deltas = []
    left_omissions = 0
    right_omissions = 0

    for cluster_id, edus in clusters.items():
        left_proms = []
        center_proms = []
        right_proms = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            prom = get_edu_prominence(meta)
            if bias == "left":
                left_proms.append(prom)
            elif bias == "center":
                center_proms.append(prom)
            elif bias == "right":
                right_proms.append(prom)

        has_left = len(left_proms) > 0
        has_center = len(center_proms) > 0
        has_right = len(right_proms) > 0

        # Handle omissions
        if has_center and not has_left:
            left_deltas.append(-1.0)  # Omission penalty
            left_omissions += 1
        elif has_left and not has_center:
            left_deltas.append(1.0)  # Exclusive bonus
        elif has_left and has_center:
            avg_left = sum(left_proms) / len(left_proms)
            avg_center = sum(center_proms) / len(center_proms)
            left_deltas.append(avg_left - avg_center)

        if has_center and not has_right:
            right_deltas.append(-1.0)
            right_omissions += 1
        elif has_right and not has_center:
            right_deltas.append(1.0)
        elif has_right and has_center:
            avg_right = sum(right_proms) / len(right_proms)
            avg_center = sum(center_proms) / len(center_proms)
            right_deltas.append(avg_right - avg_center)

    # Sort descending
    left_deltas.sort(reverse=True)
    right_deltas.sort(reverse=True)

    left_features = (left_deltas + [0.0] * max_features)[:max_features]
    right_features = (right_deltas + [0.0] * max_features)[:max_features]

    debug_info = {
        "num_left_deltas": len(left_deltas),
        "num_right_deltas": len(right_deltas),
        "left_omissions": left_omissions,
        "right_omissions": right_omissions,
    }

    return left_features, right_features, debug_info


def build_disparity_signature_binned(
    fact_row: Dict, num_bins: int = 10
) -> Tuple[List[float], List[float], Dict]:
    """
    Binned disparity signature: divide sorted deltas into percentile bins.

    This normalizes for varying numbers of facts per triplet.
    Feature i = average delta in the i-th percentile bin.
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    left_deltas = []
    right_deltas = []

    for cluster_id, edus in clusters.items():
        left_proms = []
        center_proms = []
        right_proms = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            prom = get_edu_prominence(meta)
            if bias == "left":
                left_proms.append(prom)
            elif bias == "center":
                center_proms.append(prom)
            elif bias == "right":
                right_proms.append(prom)

        has_left = len(left_proms) > 0
        has_center = len(center_proms) > 0
        has_right = len(right_proms) > 0

        if has_center and not has_left:
            left_deltas.append(-1.0)
        elif has_left and not has_center:
            left_deltas.append(1.0)
        elif has_left and has_center:
            avg_left = sum(left_proms) / len(left_proms)
            avg_center = sum(center_proms) / len(center_proms)
            left_deltas.append(avg_left - avg_center)

        if has_center and not has_right:
            right_deltas.append(-1.0)
        elif has_right and not has_center:
            right_deltas.append(1.0)
        elif has_right and has_center:
            avg_right = sum(right_proms) / len(right_proms)
            avg_center = sum(center_proms) / len(center_proms)
            right_deltas.append(avg_right - avg_center)

    # Sort descending
    left_deltas.sort(reverse=True)
    right_deltas.sort(reverse=True)

    # Bin into percentiles
    def bin_deltas(deltas, n_bins):
        if not deltas:
            return [0.0] * n_bins

        features = []
        bin_size = len(deltas) / n_bins
        for i in range(n_bins):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            if end > start:
                bin_vals = deltas[start:end]
                features.append(sum(bin_vals) / len(bin_vals))
            else:
                features.append(0.0)
        return features

    left_features = bin_deltas(left_deltas, num_bins)
    right_features = bin_deltas(right_deltas, num_bins)

    debug_info = {
        "num_left_deltas": len(left_deltas),
        "num_right_deltas": len(right_deltas),
        "num_bins": num_bins,
    }

    return left_features, right_features, debug_info


def build_disparity_signature_stats(
    fact_row: Dict, max_features: int = 50
) -> Tuple[List[float], List[float], Dict]:
    """
    Disparity signature + summary statistics.

    Features: [sorted_deltas..., mean, std, max, min, positive_count, negative_count]
    """
    edu_lookup = fact_row.get("edu_lookup", {})
    clusters = fact_row.get("clusters", {})

    left_deltas = []
    right_deltas = []

    for cluster_id, edus in clusters.items():
        left_proms = []
        center_proms = []
        right_proms = []

        for edu_id in edus:
            meta = edu_lookup.get(edu_id)
            if not meta:
                continue
            bias = meta.get("bias")
            prom = get_edu_prominence(meta)
            if bias == "left":
                left_proms.append(prom)
            elif bias == "center":
                center_proms.append(prom)
            elif bias == "right":
                right_proms.append(prom)

        has_left = len(left_proms) > 0
        has_center = len(center_proms) > 0
        has_right = len(right_proms) > 0

        if has_center and not has_left:
            left_deltas.append(-1.0)
        elif has_left and not has_center:
            left_deltas.append(1.0)
        elif has_left and has_center:
            avg_left = sum(left_proms) / len(left_proms)
            avg_center = sum(center_proms) / len(center_proms)
            left_deltas.append(avg_left - avg_center)

        if has_center and not has_right:
            right_deltas.append(-1.0)
        elif has_right and not has_center:
            right_deltas.append(1.0)
        elif has_right and has_center:
            avg_right = sum(right_proms) / len(right_proms)
            avg_center = sum(center_proms) / len(center_proms)
            right_deltas.append(avg_right - avg_center)

    # Sort descending
    left_deltas_sorted = sorted(left_deltas, reverse=True)
    right_deltas_sorted = sorted(right_deltas, reverse=True)

    # Compute statistics
    def compute_stats(deltas):
        if not deltas:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        arr = np.array(deltas)
        return [
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.max(arr)),
            float(np.min(arr)),
            float(np.sum(arr > 0)),  # positive count
            float(np.sum(arr < 0)),  # negative count
        ]

    left_stats = compute_stats(left_deltas)
    right_stats = compute_stats(right_deltas)

    # Truncate sorted deltas to leave room for stats
    stat_count = 6
    delta_count = max_features - stat_count

    left_features = (left_deltas_sorted + [0.0] * delta_count)[
        :delta_count
    ] + left_stats
    right_features = (right_deltas_sorted + [0.0] * delta_count)[
        :delta_count
    ] + right_stats

    debug_info = {
        "num_left_deltas": len(left_deltas),
        "num_right_deltas": len(right_deltas),
        "stats_included": ["mean", "std", "max", "min", "pos_count", "neg_count"],
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
        script_subdir="universal-str/disparity-signature",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "max_features": args.max_features,
            "rf": rf_cfg,
        },
    )

    print("=" * 80)
    print("UNIVERSAL STRUCTURE: DISPARITY SIGNATURE")
    print("=" * 80)
    print(
        "\nKey Insight: The DISTRIBUTION of deltas matters more than specific fact deltas."
    )
    print("By sorting deltas largest-to-smallest, we capture the 'shape' of bias.")
    print()
    print("Example signature patterns:")
    print("  - Heavy promotion: [+0.5, +0.3, +0.1, 0.0, -0.1, ...]")
    print("  - Balanced: [+0.1, +0.05, 0.0, -0.05, -0.1, ...]")
    print("  - Heavy burial: [+0.1, 0.0, -0.1, -0.3, -0.5, ...]")
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
            "disparity_basic",
            "Basic sorted deltas",
            lambda row: build_disparity_signature_basic(row, args.max_features),
        ),
        (
            "disparity_with_omissions",
            "Sorted deltas with omission penalties",
            lambda row: build_disparity_signature_with_omissions(
                row, args.max_features
            ),
        ),
        (
            "disparity_binned",
            "Binned percentile signature (10 bins)",
            lambda row: build_disparity_signature_binned(row, 10),
        ),
        (
            "disparity_with_stats",
            "Sorted deltas + summary statistics",
            lambda row: build_disparity_signature_stats(row, args.max_features),
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
                "max_features": args.max_features,
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
            "goal": "Test if sorting deltas by magnitude enables universal pattern learning",
            "approach": "Sort prominence deltas descending to create 'disparity signature'",
            "hypothesis": "The shape/distribution of deltas is topic-independent",
            "max_features": args.max_features,
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
    print("SUMMARY: Universal Structure - Disparity Signature")
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
