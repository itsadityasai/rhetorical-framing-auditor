"""
Strengthen Structural Signals (Option B)

This script implements Option B: exploring whether alternative structural features
can improve bias classification beyond the coverage-only baseline.

Three approaches are explored:

1. Aggregate Structural Features:
   - Mean/variance of depth across EDUs per side
   - Nuclearity distribution (% nucleus vs satellite)
   - Position-in-document features

2. Richer RST Relations:
   - RST relation type encoding
   - Requires loading original RST parse files

3. Alternative Prominence Formulas:
   - Exponential decay: W = exp(-β * depth)
   - Inverse satellite penalty: W = 1 / (1 + s)
   - Combined variants
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import math

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
DEFAULT_OUT_PATH = "experiments/strengthen-str/results/strengthen_structural_results.json"
DEFAULT_MODEL_DIR = "experiments/strengthen-str/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explore strengthened structural features for bias classification"
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
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
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
# APPROACH 1: Aggregate Structural Features
# =============================================================================


def compute_aggregate_features_per_side(
    edus: List[str], edu_lookup: Dict, side: str
) -> Dict:
    """
    Compute aggregate structural features for a single side in a cluster.

    Returns dict with:
    - depth_mean, depth_var, depth_min, depth_max
    - nucleus_ratio (fraction of EDUs that are nuclei)
    - satellite_edges_mean, satellite_edges_max
    - count (number of EDUs)
    """
    side_edus = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            side_edus.append(meta)

    if not side_edus:
        return {
            "depth_mean": 0.0,
            "depth_var": 0.0,
            "depth_min": 0.0,
            "depth_max": 0.0,
            "nucleus_ratio": 0.0,
            "sat_edges_mean": 0.0,
            "sat_edges_max": 0.0,
            "count": 0,
            "present": 0,
        }

    depths = [m.get("depth", 0) for m in side_edus]
    sat_edges = [m.get("satellite_edges_to_root", 0) for m in side_edus]
    roles = [m.get("role", "S") for m in side_edus]

    nucleus_count = sum(1 for r in roles if r == "N")

    return {
        "depth_mean": float(np.mean(depths)),
        "depth_var": float(np.var(depths)) if len(depths) > 1 else 0.0,
        "depth_min": float(min(depths)),
        "depth_max": float(max(depths)),
        "nucleus_ratio": nucleus_count / len(side_edus),
        "sat_edges_mean": float(np.mean(sat_edges)),
        "sat_edges_max": float(max(sat_edges)),
        "count": len(side_edus),
        "present": 1,
    }


def build_aggregate_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """
    Build aggregate structural feature vectors for a triplet.

    For each cluster, compute aggregate features for each side,
    then compute deltas (side - center).
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        left_agg = compute_aggregate_features_per_side(edus, edu_lookup, "left")
        center_agg = compute_aggregate_features_per_side(edus, edu_lookup, "center")
        right_agg = compute_aggregate_features_per_side(edus, edu_lookup, "right")

        # Delta features for left-vs-center
        left_delta = [
            left_agg["depth_mean"] - center_agg["depth_mean"],
            left_agg["depth_var"] - center_agg["depth_var"],
            left_agg["nucleus_ratio"] - center_agg["nucleus_ratio"],
            left_agg["sat_edges_mean"] - center_agg["sat_edges_mean"],
            left_agg["present"] - center_agg["present"],  # coverage signal
        ]

        # Delta features for right-vs-center
        right_delta = [
            right_agg["depth_mean"] - center_agg["depth_mean"],
            right_agg["depth_var"] - center_agg["depth_var"],
            right_agg["nucleus_ratio"] - center_agg["nucleus_ratio"],
            right_agg["sat_edges_mean"] - center_agg["sat_edges_mean"],
            right_agg["present"] - center_agg["present"],  # coverage signal
        ]

        features_left.extend(left_delta)
        features_right.extend(right_delta)

    return features_left, features_right


def build_aggregate_features_no_coverage(
    fact_row: Dict,
) -> Tuple[List[float], List[float]]:
    """
    Build aggregate features WITHOUT coverage signal (only structural).
    Only include clusters where both sides are present.
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        left_agg = compute_aggregate_features_per_side(edus, edu_lookup, "left")
        center_agg = compute_aggregate_features_per_side(edus, edu_lookup, "center")
        right_agg = compute_aggregate_features_per_side(edus, edu_lookup, "right")

        # Only include if all three sides present (removes coverage signal)
        if left_agg["present"] and center_agg["present"] and right_agg["present"]:
            # Delta features WITHOUT coverage
            left_delta = [
                left_agg["depth_mean"] - center_agg["depth_mean"],
                left_agg["depth_var"] - center_agg["depth_var"],
                left_agg["nucleus_ratio"] - center_agg["nucleus_ratio"],
                left_agg["sat_edges_mean"] - center_agg["sat_edges_mean"],
            ]

            right_delta = [
                right_agg["depth_mean"] - center_agg["depth_mean"],
                right_agg["depth_var"] - center_agg["depth_var"],
                right_agg["nucleus_ratio"] - center_agg["nucleus_ratio"],
                right_agg["sat_edges_mean"] - center_agg["sat_edges_mean"],
            ]

            features_left.extend(left_delta)
            features_right.extend(right_delta)

    return features_left, features_right


# =============================================================================
# APPROACH 2: Richer RST Relation Features (using nuclearity distribution)
# =============================================================================


def build_nuclearity_distribution_features(
    fact_row: Dict,
) -> Tuple[List[float], List[float]]:
    """
    Build features based on nuclearity role distribution.

    For each cluster, encode the nuclearity composition:
    - nucleus_ratio for each side
    - Deltas against center
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        side_stats = {}
        for side in ["left", "center", "right"]:
            side_edus = [
                edu_lookup.get(eid)
                for eid in edus
                if edu_lookup.get(eid, {}).get("bias") == side
            ]
            side_edus = [e for e in side_edus if e is not None]

            if side_edus:
                nucleus_count = sum(1 for e in side_edus if e.get("role") == "N")
                side_stats[side] = {
                    "nucleus_ratio": nucleus_count / len(side_edus),
                    "present": 1,
                    "count": len(side_edus),
                }
            else:
                side_stats[side] = {"nucleus_ratio": 0.0, "present": 0, "count": 0}

        # Coverage + nuclearity deltas
        features_left.extend(
            [
                side_stats["left"]["present"] - side_stats["center"]["present"],
                side_stats["left"]["nucleus_ratio"]
                - side_stats["center"]["nucleus_ratio"],
            ]
        )
        features_right.extend(
            [
                side_stats["right"]["present"] - side_stats["center"]["present"],
                side_stats["right"]["nucleus_ratio"]
                - side_stats["center"]["nucleus_ratio"],
            ]
        )

    return features_left, features_right


# =============================================================================
# APPROACH 3: Alternative Prominence Formulas
# =============================================================================


def W_exponential(depth: int, sat_count: int, beta: float = 0.3) -> float:
    """Exponential decay based on depth only."""
    return math.exp(-beta * depth)


def W_inverse_satellite(depth: int, sat_count: int, gamma: float = 0.5) -> float:
    """Inverse penalty based on satellite edges."""
    return 1.0 / (1.0 + gamma * sat_count)


def W_combined_exp(
    depth: int, sat_count: int, beta: float = 0.3, gamma: float = 0.5
) -> float:
    """Combined: exponential depth decay with inverse satellite penalty."""
    return math.exp(-beta * depth) / (1.0 + gamma * sat_count)


def W_log_depth(depth: int, sat_count: int) -> float:
    """Logarithmic depth penalty."""
    return 1.0 / (1.0 + math.log1p(depth))


def W_original(
    depth: int, sat_count: int, alpha: float = 0.8, gamma: float = 0.5
) -> float:
    """Original formula for comparison."""
    return (alpha ** (depth + 1)) * (gamma**sat_count)


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


def build_alternative_prominence_features(
    fact_row: Dict, W_func
) -> Tuple[List[float], List[float]]:
    """Build DFI-style features using an alternative prominence formula."""
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

    # Build X, y
    x_train_raw, y_train = build_xy(train_rows, "feat_left", "feat_right")
    x_val_raw, y_val = build_xy(val_rows, "feat_left", "feat_right")
    x_test_raw, y_test = build_xy(test_rows, "feat_left", "feat_right")

    # Handle empty features
    max_len = max((len(v) for v in x_train_raw), default=0)
    if max_len == 0:
        return (
            None,
            0,
            {
                "train": {"samples": len(y_train), "accuracy": 0.5, "macro_f1": 0.333},
                "val": {"samples": len(y_val), "accuracy": 0.5, "macro_f1": 0.333},
                "test": {"samples": len(y_test), "accuracy": 0.5, "macro_f1": 0.333},
                "error": "Empty features after filtering",
            },
        )

    x_train = pad_or_truncate(x_train_raw, max_len)
    x_val = pad_or_truncate(x_val_raw, max_len)
    x_test = pad_or_truncate(x_test_raw, max_len)

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

    return model, max_len, metrics


def save_model(path: str, model, max_len: int, experiment_name: str, svm_cfg: Dict):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    payload = {
        "model": model,
        "max_len": int(max_len),
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
        script_subdir="strengthen-str",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "svm": svm_cfg,
        },
    )

    print("=" * 70)
    print("Option B: Strengthen Structural Signals")
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

    # Define experiments
    experiments = [
        # Approach 1: Aggregate features
        (
            "agg_with_coverage",
            "Aggregate features WITH coverage signal",
            build_aggregate_features,
        ),
        (
            "agg_no_coverage",
            "Aggregate features WITHOUT coverage (size-3 only)",
            build_aggregate_features_no_coverage,
        ),
        # Approach 2: Nuclearity distribution
        (
            "nuclearity_dist",
            "Nuclearity distribution features",
            build_nuclearity_distribution_features,
        ),
        # Approach 3: Alternative prominence formulas
        (
            "W_exponential",
            "Exponential decay: W = exp(-0.3 * depth)",
            lambda row: build_alternative_prominence_features(
                row, lambda d, s: W_exponential(d, s, beta=0.3)
            ),
        ),
        (
            "W_inverse_sat",
            "Inverse satellite: W = 1 / (1 + 0.5 * s)",
            lambda row: build_alternative_prominence_features(
                row, lambda d, s: W_inverse_satellite(d, s, gamma=0.5)
            ),
        ),
        (
            "W_combined",
            "Combined: exp(-0.3*d) / (1 + 0.5*s)",
            lambda row: build_alternative_prominence_features(
                row, lambda d, s: W_combined_exp(d, s, beta=0.3, gamma=0.5)
            ),
        ),
        (
            "W_log_depth",
            "Log depth: W = 1 / (1 + log(1+d))",
            lambda row: build_alternative_prominence_features(row, W_log_depth),
        ),
        (
            "W_original",
            "Original formula (baseline): α^(d+1) * γ^s",
            lambda row: build_alternative_prominence_features(
                row, lambda d, s: W_original(d, s, alpha=0.8, gamma=0.5)
            ),
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

        model, max_len, metrics = train_and_evaluate_experiment(
            train_rows, val_rows, test_rows, feature_builder, svm_cfg, exp_name
        )

        if model is not None:
            model_path = os.path.join(args.model_dir, f"{exp_name}.pkl")
            save_model(model_path, model, max_len, exp_name, svm_cfg)
        else:
            model_path = None

        results[exp_name] = {
            "description": description,
            "input_dim": max_len,
            "metrics": metrics,
            "model_path": model_path,
        }

        print(f"Input dim: {max_len}")
        print(
            f"Val:  acc={metrics['val']['accuracy']:.4f}, f1={metrics['val']['macro_f1']:.4f}"
        )
        print(
            f"Test: acc={metrics['test']['accuracy']:.4f}, f1={metrics['test']['macro_f1']:.4f}"
        )

    # Reference baselines
    baselines = {
        "coverage_only": {"test_acc": 0.7528, "test_f1": 0.7504},
        "structural_baseline": {"test_acc": 0.6705, "test_f1": 0.6702},
        "size3_baseline": {"test_acc": 0.5347, "test_f1": 0.5309},
    }

    # Compute deltas
    for exp_name, exp_data in results.items():
        exp_data["delta_vs_coverage"] = {
            "test_acc": exp_data["metrics"]["test"]["accuracy"]
            - baselines["coverage_only"]["test_acc"],
            "test_f1": exp_data["metrics"]["test"]["macro_f1"]
            - baselines["coverage_only"]["test_f1"],
        }
        exp_data["delta_vs_structural"] = {
            "test_acc": exp_data["metrics"]["test"]["accuracy"]
            - baselines["structural_baseline"]["test_acc"],
            "test_f1": exp_data["metrics"]["test"]["macro_f1"]
            - baselines["structural_baseline"]["test_f1"],
        }

    output = {
        "setup": {
            "goal": "Explore strengthened structural features (Option B)",
            "facts": args.facts,
            "split_dir": args.split_dir,
            "svm": svm_cfg,
            "created": datetime.now().isoformat(),
        },
        "reference_baselines": baselines,
        "experiments": results,
    }

    save_json(args.out, output)

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY: Option B Results")
    print("=" * 90)
    print(
        f"\n{'Experiment':<25} {'Test Acc':<12} {'Test F1':<12} {'Δ vs Coverage':<15} {'Δ vs Structural':<15}"
    )
    print("-" * 90)

    for exp_name, exp_data in results.items():
        test_acc = exp_data["metrics"]["test"]["accuracy"]
        test_f1 = exp_data["metrics"]["test"]["macro_f1"]
        delta_cov = exp_data["delta_vs_coverage"]["test_acc"]
        delta_str = exp_data["delta_vs_structural"]["test_acc"]
        print(
            f"{exp_name:<25} {test_acc:<12.4f} {test_f1:<12.4f} {delta_cov:<+15.4f} {delta_str:<+15.4f}"
        )

    print("-" * 90)
    print(
        f"{'coverage_only (ref)':<25} {0.7528:<12.4f} {0.7504:<12.4f} {'--':<15} {'+0.0823':<15}"
    )
    print(
        f"{'structural_baseline (ref)':<25} {0.6705:<12.4f} {0.6702:<12.4f} {'-0.0823':<15} {'--':<15}"
    )

    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out, "experiments": list(results.keys())})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
