"""
Coverage-based SVM for bias classification.

This script implements Option A: a pure coverage/omission-based model that
encodes only the presence/absence of facts per bias side, without any
structural (RST) weighting.

The hypothesis being tested: Fact omission patterns are stronger predictors
of media bias than rhetorical prominence placement.

Coverage features per cluster:
- Binary: [has_left, has_center, has_right] -> 3 features per cluster
- Or simplified: [left_present - center_present, right_present - center_present] -> 2 deltas per cluster

This mirrors the DFI structure but replaces W(d,s) with binary 0/1.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
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
DEFAULT_OUT_PATH = "experiments/omission-based/results/coverage_svm_results.json"
DEFAULT_MODEL_PATH = "experiments/omission-based/results/coverage_svm_model.pkl"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train coverage-only SVM for bias classification (no structural weighting)"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH, help="Facts JSON path")
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help="Directory with train/val/test splits",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT_PATH, help="Output JSON path for results"
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH, help="Path to save trained model"
    )

    # Feature encoding options
    parser.add_argument(
        "--feature-mode",
        choices=["delta", "binary3", "count"],
        default="delta",
        help=(
            "Feature encoding mode: "
            "'delta' = [left-center, right-center] per cluster (2 features); "
            "'binary3' = [has_left, has_center, has_right] per cluster (3 features); "
            "'count' = [count_left, count_center, count_right] per cluster (3 features)"
        ),
    )

    # SVM hyperparameters
    parser.add_argument(
        "--svm-kernel", default="rbf", choices=["rbf", "linear", "poly"]
    )
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


def get_cluster_coverage(edus: List[str], edu_lookup: Dict) -> Dict[str, int]:
    """
    For a cluster, count how many EDUs belong to each bias side.
    Returns: {"left": count, "center": count, "right": count}
    """
    counts = {"left": 0, "center": 0, "right": 0}
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta is None:
            continue
        bias = meta.get("bias")
        if bias in VALID_BIASES:
            counts[bias] += 1
    return counts


def build_coverage_features(
    fact_row: Dict, feature_mode: str
) -> Tuple[List[float], List[float]]:
    """
    Build coverage-based feature vectors for left-vs-center and right-vs-center.

    Returns:
        (features_left, features_right): Feature vectors for the two binary examples
    """
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        coverage = get_cluster_coverage(edus, edu_lookup)

        if feature_mode == "delta":
            # Binary delta: (has_left - has_center), (has_right - has_center)
            has_left = 1 if coverage["left"] > 0 else 0
            has_center = 1 if coverage["center"] > 0 else 0
            has_right = 1 if coverage["right"] > 0 else 0

            # For left example: delta = has_left - has_center
            features_left.append(has_left - has_center)
            # For right example: delta = has_right - has_center
            features_right.append(has_right - has_center)

        elif feature_mode == "binary3":
            # Full binary encoding per cluster
            has_left = 1 if coverage["left"] > 0 else 0
            has_center = 1 if coverage["center"] > 0 else 0
            has_right = 1 if coverage["right"] > 0 else 0

            # Same features for both, labels differentiate
            features_left.extend([has_left, has_center, has_right])
            features_right.extend([has_left, has_center, has_right])

        elif feature_mode == "count":
            # Raw counts per cluster
            features_left.extend(
                [coverage["left"], coverage["center"], coverage["right"]]
            )
            features_right.extend(
                [coverage["left"], coverage["center"], coverage["right"]]
            )

    return features_left, features_right


def build_coverage_rows(facts_rows: List[dict], feature_mode: str) -> List[dict]:
    """Build coverage feature rows from facts."""
    out = []
    for row in facts_rows:
        feat_left, feat_right = build_coverage_features(row, feature_mode)
        out.append(
            {
                "triplet_idx": row.get("triplet_idx"),
                "coverage_left": feat_left,
                "coverage_right": feat_right,
                "num_clusters": len(row.get("clusters", {})),
            }
        )
    return out


def build_xy(
    rows: List[dict], key_left: str = "coverage_left", key_right: str = "coverage_right"
):
    """Build X, y arrays from coverage rows."""
    x, y = [], []
    for row in rows:
        x.append(list(row[key_left]))
        y.append(0)  # left-vs-center label
        x.append(list(row[key_right]))
        y.append(1)  # right-vs-center label
    return x, np.array(y)


def pad_or_truncate(raw_x: List[List[float]], target_len: int) -> np.ndarray:
    """Pad shorter vectors with zeros, truncate longer ones."""
    arr = np.zeros((len(raw_x), target_len), dtype=float)
    for i, vec in enumerate(raw_x):
        lim = min(len(vec), target_len)
        if lim > 0:
            arr[i, :lim] = np.array(vec[:lim], dtype=float)
    return arr


def evaluate(model, rows: List[dict], max_len: int, feature_mode: str) -> Dict:
    """Evaluate model on a set of rows."""
    x_raw, y = build_xy(rows)
    x = pad_or_truncate(x_raw, max_len)
    pred = model.predict(x)

    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def train_and_evaluate(
    train_rows: List[dict],
    val_rows: List[dict],
    test_rows: List[dict],
    svm_cfg: Dict,
    feature_mode: str,
) -> Tuple[object, int, Dict]:
    """Train SVM and evaluate on all splits."""
    if not train_rows:
        raise RuntimeError("No training rows available")

    x_train_raw, y_train = build_xy(train_rows)
    max_len = max((len(v) for v in x_train_raw), default=0)

    if max_len == 0:
        raise RuntimeError("Training feature vectors are empty")

    x_train = pad_or_truncate(x_train_raw, max_len)

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
        "train": evaluate(model, train_rows, max_len, feature_mode),
        "val": evaluate(model, val_rows, max_len, feature_mode),
        "test": evaluate(model, test_rows, max_len, feature_mode),
    }

    return model, max_len, metrics


def save_model(path: str, model, max_len: int, feature_mode: str, svm_cfg: Dict):
    """Save trained model with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model,
        "max_len": int(max_len),
        "feature_mode": feature_mode,
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
        script_subdir="omission-based",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_path": args.model_path,
            "feature_mode": args.feature_mode,
            "svm": svm_cfg,
        },
    )

    print("=" * 60)
    print("Coverage-based SVM Training")
    print("=" * 60)
    print(f"Feature mode: {args.feature_mode}")
    print(f"SVM config: {svm_cfg}")
    print()

    # Load data
    print("Loading facts and split partitions...")
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
        f"Facts rows: train={len(facts_train)}, val={len(facts_val)}, test={len(facts_test)}"
    )

    # Build coverage features
    print(f"\nBuilding coverage features (mode={args.feature_mode})...")
    train_rows = build_coverage_rows(facts_train, args.feature_mode)
    val_rows = build_coverage_rows(facts_val, args.feature_mode)
    test_rows = build_coverage_rows(facts_test, args.feature_mode)

    # Train and evaluate
    print("Training SVM...")
    model, max_len, metrics = train_and_evaluate(
        train_rows, val_rows, test_rows, svm_cfg, args.feature_mode
    )

    print(f"Input dimension (max_len): {max_len}")

    # Save model
    save_model(args.model_path, model, max_len, args.feature_mode, svm_cfg)
    print(f"Model saved to: {args.model_path}")

    # Prepare output
    output = {
        "setup": {
            "goal": "Coverage-only bias classification (no structural weighting)",
            "hypothesis": "Fact omission patterns predict bias better than RST prominence",
            "facts": args.facts,
            "split_dir": args.split_dir,
            "feature_mode": args.feature_mode,
            "svm": svm_cfg,
            "created": datetime.now().isoformat(),
        },
        "data_counts": {
            "triplet_rows": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
            },
            "binary_samples": {
                "train": len(train_rows) * 2,
                "val": len(val_rows) * 2,
                "test": len(test_rows) * 2,
            },
        },
        "model": {
            "input_dim": max_len,
            "model_path": args.model_path,
        },
        "metrics": metrics,
        "comparison_to_structural": {
            "note": "Compare these results to data/ablation/structural_ablation_recluster_gpu.json",
            "structural_baseline_test_acc": 0.6705,
            "structural_without_both_test_acc": 0.7528,
        },
    }

    save_json(args.out, output)

    # Print results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\n{'Split':<10} {'Samples':<10} {'Accuracy':<12} {'Macro-F1':<12}")
    print("-" * 44)
    for split in ["train", "val", "test"]:
        m = metrics[split]
        print(
            f"{split:<10} {m['samples']:<10} {m['accuracy']:<12.4f} {m['macro_f1']:<12.4f}"
        )

    print("\n" + "-" * 60)
    print("Comparison to Structural DFI Models:")
    print("-" * 60)
    print(f"{'Model':<30} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 54)
    print(f"{'Structural baseline (α=0.8,γ=0.5)':<30} {0.6705:<12.4f} {0.6702:<12.4f}")
    print(f"{'Structural without_both (α=1,γ=1)':<30} {0.7528:<12.4f} {0.7504:<12.4f}")
    print(
        f"{'Coverage-only (this model)':<30} {metrics['test']['accuracy']:<12.4f} {metrics['test']['macro_f1']:<12.4f}"
    )

    delta_vs_structural = metrics["test"]["accuracy"] - 0.6705
    delta_vs_without_both = metrics["test"]["accuracy"] - 0.7528
    print(f"\nDelta vs structural baseline: {delta_vs_structural:+.4f}")
    print(f"Delta vs without_both:        {delta_vs_without_both:+.4f}")

    print(f"\nResults saved to: {args.out}")

    # Log results
    log_run_results(
        run_log,
        {
            "out": args.out,
            "model_path": args.model_path,
            "feature_mode": args.feature_mode,
            "input_dim": max_len,
            "metrics": metrics,
            "delta_vs_structural_baseline": delta_vs_structural,
            "delta_vs_without_both": delta_vs_without_both,
        },
    )
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
