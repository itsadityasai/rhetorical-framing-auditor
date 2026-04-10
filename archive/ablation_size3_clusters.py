import argparse
import json
import os
import pickle
from typing import Dict, List, Set, Tuple

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from modules.DFIGenerator import DFIGenerator
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = "data/ablation/structural_ablation_size3_recluster_gpu.json"
DEFAULT_MODEL_DIR = "data/ablation/models_size3_recluster_gpu"

DEFAULT_CLUSTER_FILTER = "exact3_all3"
DEFAULT_MIN_CLUSTERS = 1


default_svm_cfg = params.get("ablation", {}).get("svm", params.get("svm", {}).get("model", {}))
DEFAULT_SVM_KERNEL = default_svm_cfg.get("kernel", "rbf")
DEFAULT_SVM_C = default_svm_cfg.get("C", 10)
DEFAULT_SVM_GAMMA = default_svm_cfg.get("gamma", 0.1)
DEFAULT_SVM_DEGREE = default_svm_cfg.get("degree", 3)

DEFAULT_BASELINE_ALPHA = params.get("dfi", {}).get("alpha", 0.8)
DEFAULT_BASELINE_GAMMA = params.get("dfi", {}).get("gamma", 0.5)


VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run structural DFI ablations using only tri-side clusters (coverage signal removed)"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH, help="Facts JSON built from fresh clusters")
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help="Directory with train/val/test DFI split rows (used only for triplet_idx partition)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Ablation summary JSON output path")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory to store trained SVM models")

    parser.add_argument(
        "--cluster-filter",
        choices=["exact3_all3", "all3_anylen"],
        default=DEFAULT_CLUSTER_FILTER,
        help=(
            "Cluster retention rule: exact3_all3 keeps clusters with exactly 3 EDUs and one per bias; "
            "all3_anylen keeps clusters that contain all three biases regardless of EDU count"
        ),
    )
    parser.add_argument(
        "--min-clusters-per-triplet",
        type=int,
        default=DEFAULT_MIN_CLUSTERS,
        help="Drop triplets that have fewer than this many retained clusters after filtering",
    )

    parser.add_argument("--baseline-alpha", type=float, default=DEFAULT_BASELINE_ALPHA)
    parser.add_argument("--baseline-gamma", type=float, default=DEFAULT_BASELINE_GAMMA)

    parser.add_argument("--svm-kernel", default=DEFAULT_SVM_KERNEL)
    parser.add_argument("--svm-c", type=float, default=DEFAULT_SVM_C)
    parser.add_argument("--svm-gamma", type=float, default=DEFAULT_SVM_GAMMA)
    parser.add_argument("--svm-degree", type=int, default=DEFAULT_SVM_DEGREE)
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


def _extract_biases(edus: List[str], edu_lookup: Dict) -> List[str]:
    biases = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta is None:
            return []
        bias = meta.get("bias")
        if bias not in VALID_BIASES:
            return []
        biases.append(bias)
    return biases


def keep_cluster(edus: List[str], edu_lookup: Dict, cluster_filter: str) -> bool:
    biases = _extract_biases(edus, edu_lookup)
    if not biases:
        return False

    bias_set = set(biases)

    if cluster_filter == "exact3_all3":
        return len(edus) == 3 and bias_set == VALID_BIASES

    if cluster_filter == "all3_anylen":
        return bias_set == VALID_BIASES

    raise ValueError(f"Unsupported cluster_filter: {cluster_filter}")


def filter_fact_rows(
    facts_rows: List[dict],
    cluster_filter: str,
    min_clusters_per_triplet: int,
):
    stats = {
        "input_triplets": len(facts_rows),
        "kept_triplets": 0,
        "dropped_triplets": 0,
        "clusters_before": 0,
        "clusters_after": 0,
    }

    filtered_rows = []

    for row in facts_rows:
        clusters = row.get("clusters", {})
        edu_lookup = row.get("edu_lookup", {})

        stats["clusters_before"] += len(clusters)

        kept_clusters = {}
        for cluster_id, edus in clusters.items():
            if keep_cluster(edus, edu_lookup, cluster_filter):
                kept_clusters[cluster_id] = edus

        stats["clusters_after"] += len(kept_clusters)

        if len(kept_clusters) < min_clusters_per_triplet:
            stats["dropped_triplets"] += 1
            continue

        filtered_rows.append(
            {
                "triplet_idx": row.get("triplet_idx"),
                "clusters": kept_clusters,
                "edu_lookup": edu_lookup,
            }
        )

    stats["kept_triplets"] = len(filtered_rows)
    return filtered_rows, stats


def build_dfi_rows(filtered_rows: List[dict], alpha: float, gamma: float) -> List[dict]:
    out = []
    for row in filtered_rows:
        dfi = DFIGenerator(alpha=alpha, gamma=gamma, clusters=row["clusters"], edu_lookup=row["edu_lookup"])
        cluster_ps = dfi.get_ps()
        dfi_left, dfi_right = dfi.get_DFIs(cluster_ps)

        out.append(
            {
                "triplet_idx": row.get("triplet_idx"),
                "dfi_left": dfi_left,
                "dfi_right": dfi_right,
                "num_clusters": len(cluster_ps),
            }
        )
    return out


def build_xy_raw(rows: List[dict]):
    x, y = [], []
    for row in rows:
        x.append(list(row["dfi_left"]))
        y.append(0)
        x.append(list(row["dfi_right"]))
        y.append(1)
    return x, np.array(y)


def pad_or_truncate(raw_x, target_len):
    arr = np.zeros((len(raw_x), target_len), dtype=float)
    for i, vec in enumerate(raw_x):
        lim = min(len(vec), target_len)
        if lim > 0:
            arr[i, :lim] = np.array(vec[:lim], dtype=float)
    return arr


def evaluate(model, rows: List[dict], max_len: int) -> Dict:
    x_raw, y = build_xy_raw(rows)
    x = pad_or_truncate(x_raw, max_len)
    pred = model.predict(x)
    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def train_and_evaluate(train_rows: List[dict], val_rows: List[dict], test_rows: List[dict], svm_cfg: Dict):
    if not train_rows:
        raise RuntimeError("No training rows available after cluster filtering")

    x_train_raw, y_train = build_xy_raw(train_rows)
    max_len = max((len(v) for v in x_train_raw), default=0)
    if max_len == 0:
        raise RuntimeError("Training DFI vectors are empty after filtering; cannot train SVM")

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

    return model, max_len, {
        "train": evaluate(model, train_rows, max_len),
        "val": evaluate(model, val_rows, max_len),
        "test": evaluate(model, test_rows, max_len),
    }


def save_model(path: str, model, max_len: int, experiment_name: str, alpha: float, gamma: float, svm_cfg: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model,
        "max_len": int(max_len),
        "experiment": experiment_name,
        "alpha": float(alpha),
        "gamma": float(gamma),
        "svm": svm_cfg,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    args = parse_args()

    if args.min_clusters_per_triplet < 1:
        raise ValueError("--min-clusters-per-triplet must be >= 1")

    svm_cfg = {
        "kernel": args.svm_kernel,
        "C": args.svm_c,
        "gamma": args.svm_gamma,
        "degree": args.svm_degree,
    }

    experiments = [
        ("baseline", args.baseline_alpha, args.baseline_gamma, "Full structure: depth and satellite effects."),
        ("without_s", args.baseline_alpha, 1.0, "Without s: remove satellite/nuclearity effect (gamma=1)."),
        ("without_d", 1.0, args.baseline_gamma, "Without d: remove depth effect (alpha=1)."),
        ("without_both", 1.0, 1.0, "Without d and s: no structural weighting."),
    ]

    run_log = init_run_logging(
        script_subdir="ablation",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "cluster_filter": args.cluster_filter,
            "min_clusters_per_triplet": args.min_clusters_per_triplet,
            "baseline_alpha": args.baseline_alpha,
            "baseline_gamma": args.baseline_gamma,
            "svm": svm_cfg,
            "experiments": [name for name, _, _, _ in experiments],
        },
    )

    print("Loading facts and split partitions...")
    facts = load_json(args.facts)
    train_split = load_json(os.path.join(args.split_dir, "train.json"))
    val_split = load_json(os.path.join(args.split_dir, "val.json"))
    test_split = load_json(os.path.join(args.split_dir, "test.json"))

    train_ids = get_split_triplet_idx(train_split)
    val_ids = get_split_triplet_idx(val_split)
    test_ids = get_split_triplet_idx(test_split)

    facts_train, facts_val, facts_test = split_facts_by_existing_splits(facts, train_ids, val_ids, test_ids)

    print(
        "Triplets mapped from facts -> "
        f"train: {len(facts_train)}, val: {len(facts_val)}, test: {len(facts_test)}"
    )

    filt_train, stats_train = filter_fact_rows(facts_train, args.cluster_filter, args.min_clusters_per_triplet)
    filt_val, stats_val = filter_fact_rows(facts_val, args.cluster_filter, args.min_clusters_per_triplet)
    filt_test, stats_test = filter_fact_rows(facts_test, args.cluster_filter, args.min_clusters_per_triplet)

    print(
        "Triplets after cluster filter -> "
        f"train: {len(filt_train)}, val: {len(filt_val)}, test: {len(filt_test)}"
    )

    os.makedirs(args.model_dir, exist_ok=True)
    results_by_experiment = {}

    for name, alpha, gamma, description in experiments:
        print(f"\nBuilding DFI rows for {name} (alpha={alpha}, gamma={gamma})...")
        train_rows = build_dfi_rows(filt_train, alpha=alpha, gamma=gamma)
        val_rows = build_dfi_rows(filt_val, alpha=alpha, gamma=gamma)
        test_rows = build_dfi_rows(filt_test, alpha=alpha, gamma=gamma)

        print(f"Training/evaluating SVM for {name}...")
        model, max_len, metrics = train_and_evaluate(train_rows, val_rows, test_rows, svm_cfg)

        model_path = os.path.join(args.model_dir, f"{name}.pkl")
        save_model(model_path, model, max_len, name, alpha, gamma, svm_cfg)

        results_by_experiment[name] = {
            "description": description,
            "alpha": float(alpha),
            "gamma": float(gamma),
            "feature_mode": "raw_padded_dfi_size3_clusters",
            "input_dim": int(max_len),
            "triplet_rows": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
            },
            "metrics": metrics,
            "model_path": model_path,
        }

    baseline = results_by_experiment["baseline"]
    delta_vs_baseline = {}
    for name in ["without_s", "without_d", "without_both"]:
        exp = results_by_experiment[name]
        delta_vs_baseline[name] = {
            "val_accuracy": exp["metrics"]["val"]["accuracy"] - baseline["metrics"]["val"]["accuracy"],
            "val_macro_f1": exp["metrics"]["val"]["macro_f1"] - baseline["metrics"]["val"]["macro_f1"],
            "test_accuracy": exp["metrics"]["test"]["accuracy"] - baseline["metrics"]["test"]["accuracy"],
            "test_macro_f1": exp["metrics"]["test"]["macro_f1"] - baseline["metrics"]["test"]["macro_f1"],
        }

    output = {
        "setup": {
            "goal": "Structural ablation with coverage signal removed via tri-side cluster filtering",
            "facts": args.facts,
            "split_dir": args.split_dir,
            "cluster_filter": args.cluster_filter,
            "min_clusters_per_triplet": int(args.min_clusters_per_triplet),
            "svm": svm_cfg,
            "baseline": {
                "alpha": float(args.baseline_alpha),
                "gamma": float(args.baseline_gamma),
            },
        },
        "triplet_counts_before_filter": {
            "train": len(facts_train),
            "val": len(facts_val),
            "test": len(facts_test),
        },
        "triplet_counts_after_filter": {
            "train": len(filt_train),
            "val": len(filt_val),
            "test": len(filt_test),
        },
        "filter_stats": {
            "train": stats_train,
            "val": stats_val,
            "test": stats_test,
        },
        "experiments": results_by_experiment,
        "delta_vs_baseline": delta_vs_baseline,
    }

    save_json(args.out, output)

    print("\n=== Size-3 Cluster Structural Ablation Summary ===")
    for name in ["baseline", "without_s", "without_d", "without_both"]:
        exp = results_by_experiment[name]
        print(
            f"{name:12s} | "
            f"val acc={exp['metrics']['val']['accuracy']:.4f}, val f1={exp['metrics']['val']['macro_f1']:.4f} | "
            f"test acc={exp['metrics']['test']['accuracy']:.4f}, test f1={exp['metrics']['test']['macro_f1']:.4f} | "
            f"model={exp['model_path']}"
        )

    print(f"Saved ablation report to {args.out}")

    log_run_results(
        run_log,
        {
            "out": args.out,
            "model_dir": args.model_dir,
            "cluster_filter": args.cluster_filter,
            "triplet_counts_after_filter": {
                "train": len(filt_train),
                "val": len(filt_val),
                "test": len(filt_test),
            },
            "baseline": results_by_experiment["baseline"]["metrics"],
            "delta_vs_baseline": delta_vs_baseline,
        },
    )
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
