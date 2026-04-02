import json
import os
from typing import Dict, List, Set, Tuple
import yaml

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from modules.DFIGenerator import DFIGenerator
from modules.run_logger import init_run_logging, log_run_results, close_run_logging


# =========================
# CONFIG
# =========================
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

FACTS_PATH = params["paths"]["files"]["facts"]
TRAIN_PATH = params["paths"]["files"]["dfi_train"]
VAL_PATH = params["paths"]["files"]["dfi_val"]
TEST_PATH = params["paths"]["files"]["dfi_test"]

OUT_PATH = params["paths"]["files"]["ablation_nuclearity"]

# Same SVM hyperparams for fair comparison
SVM_KERNEL = params["ablation"]["svm"]["kernel"]
SVM_C = params["ablation"]["svm"]["C"]
SVM_GAMMA = params["ablation"]["svm"]["gamma"]
SVM_DEGREE = params["ablation"]["svm"]["degree"]

# DFI params
ALPHA = params["ablation"]["alpha"]
BASELINE_GAMMA = params["ablation"]["baseline_gamma"]
ABLATION_GAMMA = params["ablation"]["ablation_gamma"]  # removes satellite/nuclearity weighting

RUN_LOG = init_run_logging(
    script_subdir="ablation",
    hyperparams={
        "facts_path": FACTS_PATH,
        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path": TEST_PATH,
        "alpha": ALPHA,
        "baseline_gamma": BASELINE_GAMMA,
        "ablation_gamma": ABLATION_GAMMA,
        "svm": {
            "kernel": SVM_KERNEL,
            "C": SVM_C,
            "gamma": SVM_GAMMA,
            "degree": SVM_DEGREE,
        },
        "out_path": OUT_PATH,
    },
)


# =========================
# IO HELPERS
# =========================
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# =========================
# DATA PREP
# =========================
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


def build_dfi_rows_from_facts(facts_rows: List[dict], alpha: float, gamma: float) -> List[dict]:
    out = []
    for row in facts_rows:
        clusters = row.get("clusters", {})
        edu_lookup = row.get("edu_lookup", {})

        dfi = DFIGenerator(alpha=alpha, gamma=gamma, clusters=clusters, edu_lookup=edu_lookup)
        cluster_ps = dfi.get_ps()
        dfi_left, dfi_right = dfi.get_DFIs(cluster_ps)

        out.append(
            {
                "triplet_idx": row.get("triplet_idx"),
                "dfi_left": dfi_left,
                "dfi_right": dfi_right,
            }
        )
    return out


# =========================
# MODEL HELPERS
# =========================
def build_xy_raw(rows: List[dict]):
    X, y = [], []
    for row in rows:
        X.append(list(row["dfi_left"]))
        y.append(0)
        X.append(list(row["dfi_right"]))
        y.append(1)
    return X, np.array(y)


def pad_or_truncate(raw_X, target_len):
    X = np.zeros((len(raw_X), target_len), dtype=float)
    for i, vec in enumerate(raw_X):
        limit = min(len(vec), target_len)
        if limit > 0:
            X[i, :limit] = np.array(vec[:limit], dtype=float)
    return X


def train_eval(train_rows: List[dict], eval_rows: List[dict]) -> Dict:
    X_train_raw, y_train = build_xy_raw(train_rows)
    X_eval_raw, y_eval = build_xy_raw(eval_rows)

    max_len = max((len(v) for v in X_train_raw), default=0)
    X_train = pad_or_truncate(X_train_raw, max_len)
    X_eval = pad_or_truncate(X_eval_raw, max_len)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA, degree=SVM_DEGREE),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred)
    macro_f1 = f1_score(y_eval, y_pred, average="macro")
    cm = confusion_matrix(y_eval, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm,
        "samples": int(len(y_eval)),
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Loading facts and leakage-safe splits...")
    facts = load_json(FACTS_PATH)
    train_split = load_json(TRAIN_PATH)
    val_split = load_json(VAL_PATH)
    test_split = load_json(TEST_PATH)

    train_ids = get_split_triplet_idx(train_split)
    val_ids = get_split_triplet_idx(val_split)
    test_ids = get_split_triplet_idx(test_split)

    facts_train, facts_val, facts_test = split_facts_by_existing_splits(
        facts, train_ids, val_ids, test_ids
    )

    print(
        f"Triplets mapped from facts -> train: {len(facts_train)}, val: {len(facts_val)}, test: {len(facts_test)}"
    )

    print("Building baseline DFI rows (with nuclearity weighting)...")
    base_train = build_dfi_rows_from_facts(facts_train, alpha=ALPHA, gamma=BASELINE_GAMMA)
    base_val = build_dfi_rows_from_facts(facts_val, alpha=ALPHA, gamma=BASELINE_GAMMA)
    base_test = build_dfi_rows_from_facts(facts_test, alpha=ALPHA, gamma=BASELINE_GAMMA)

    print("Building ablation DFI rows (nuclearity removed: gamma=1.0)...")
    abl_train = build_dfi_rows_from_facts(facts_train, alpha=ALPHA, gamma=ABLATION_GAMMA)
    abl_val = build_dfi_rows_from_facts(facts_val, alpha=ALPHA, gamma=ABLATION_GAMMA)
    abl_test = build_dfi_rows_from_facts(facts_test, alpha=ALPHA, gamma=ABLATION_GAMMA)

    print("Training/evaluating baseline...")
    baseline_val = train_eval(base_train, base_val)
    baseline_test = train_eval(base_train, base_test)

    print("Training/evaluating ablation...")
    ablation_val = train_eval(abl_train, abl_val)
    ablation_test = train_eval(abl_train, abl_test)

    delta_val = ablation_val["macro_f1"] - baseline_val["macro_f1"]
    delta_test = ablation_test["macro_f1"] - baseline_test["macro_f1"]

    results = {
        "setup": {
            "description": "Nuclearity ablation: set gamma=1.0 so satellite count has no effect; prominence becomes depth-only.",
            "alpha": ALPHA,
            "baseline_gamma": BASELINE_GAMMA,
            "ablation_gamma": ABLATION_GAMMA,
            "svm": {
                "kernel": SVM_KERNEL,
                "C": SVM_C,
                "gamma": SVM_GAMMA,
                "degree": SVM_DEGREE,
            },
        },
        "triplet_counts": {
            "train": len(facts_train),
            "val": len(facts_val),
            "test": len(facts_test),
        },
        "baseline": {
            "val": baseline_val,
            "test": baseline_test,
        },
        "ablation_depth_only": {
            "val": ablation_val,
            "test": ablation_test,
        },
        "delta_ablation_minus_baseline": {
            "val_macro_f1": float(delta_val),
            "test_macro_f1": float(delta_test),
        },
    }

    save_json(OUT_PATH, results)

    print("\n=== Nuclearity Ablation Summary ===")
    print(
        f"Baseline  macro-F1 | val: {baseline_val['macro_f1']:.4f} | test: {baseline_test['macro_f1']:.4f}"
    )
    print(
        f"Ablation  macro-F1 | val: {ablation_val['macro_f1']:.4f} | test: {ablation_test['macro_f1']:.4f}"
    )
    print(
        f"Delta (abl - base) | val: {delta_val:+.4f} | test: {delta_test:+.4f}"
    )
    print(f"Saved detailed results to {OUT_PATH}")

    log_run_results(
        RUN_LOG,
        {
            "out_path": OUT_PATH,
            "baseline": results["baseline"],
            "ablation_depth_only": results["ablation_depth_only"],
            "delta_ablation_minus_baseline": results["delta_ablation_minus_baseline"],
            "triplet_counts": results["triplet_counts"],
        },
    )
    close_run_logging(RUN_LOG, status="success")
