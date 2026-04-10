#!/usr/bin/env python3
"""Explain Random Forest predictions at EDU level for bipartite coverage features.

This script:
1) Loads the trained `alt3_bipartite_cov` Random Forest model.
2) Rebuilds test-set feature vectors with the same bipartite coverage logic.
3) Samples N binary test vectors (left-vs-center / right-vs-center).
4) Predicts bias side and computes local tree-path (Saabas-style) contributions.
5) Maps top contributing features back to exact EDU(s) and writes explainable outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from typing import Dict, List, Set, Tuple

import numpy as np


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_MODEL_PATH = "dfi-alternatives/results/models/alt3_bipartite_cov.pkl"
DEFAULT_OUT_JSON = "explanable-prediction/results/explain_predictions_results.json"
DEFAULT_OUT_MD = "explanable-prediction/results/explain_predictions_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain RF bias predictions with exact EDU evidence"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_split_triplet_idx(split_rows: List[dict]) -> Set[int]:
    return {row["triplet_idx"] for row in split_rows if "triplet_idx" in row}


def W_log_depth(depth: int) -> float:
    return 1.0 / (1.0 + math.log1p(depth))


def get_edu_prominence(edu_meta: Dict) -> float:
    if not edu_meta:
        return 0.0
    depth = edu_meta.get("depth", 0)
    return W_log_depth(depth)


def get_side_edus(edus: List[str], edu_lookup: Dict, side: str) -> List[Dict]:
    result = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            item = dict(meta)
            item["edu_id"] = eid
            result.append(item)
    return result


def compute_edu_similarity(edu1_meta: Dict, edu2_meta: Dict) -> float:
    if not edu1_meta or not edu2_meta:
        return 0.0

    depth1 = edu1_meta.get("depth", 0)
    depth2 = edu2_meta.get("depth", 0)
    depth_sim = 1.0 / (1.0 + abs(depth1 - depth2))

    prom1 = get_edu_prominence(edu1_meta)
    prom2 = get_edu_prominence(edu2_meta)
    prom_sim = 1.0 - abs(prom1 - prom2)

    return (depth_sim + prom_sim) / 2.0


def greedy_bipartite_match(
    left_edus: List[Dict], center_edus: List[Dict], right_edus: List[Dict]
) -> Tuple[List[Tuple], List[Dict], List[Dict], List[Dict]]:
    left_pool = list(left_edus)
    center_pool = list(center_edus)
    right_pool = list(right_edus)

    matched_triplets = []

    while left_pool and center_pool and right_pool:
        best_score = -1.0
        best_triplet = None
        best_indices = None

        for i, l_edu in enumerate(left_pool):
            for j, c_edu in enumerate(center_pool):
                for k, r_edu in enumerate(right_pool):
                    sim_lc = compute_edu_similarity(l_edu, c_edu)
                    sim_lr = compute_edu_similarity(l_edu, r_edu)
                    sim_cr = compute_edu_similarity(c_edu, r_edu)
                    score = (sim_lc + sim_lr + sim_cr) / 3.0

                    if score > best_score:
                        best_score = score
                        best_triplet = (l_edu, c_edu, r_edu)
                        best_indices = (i, j, k)

        if best_triplet is not None and best_indices is not None:
            matched_triplets.append(best_triplet)
            left_pool.pop(best_indices[0])
            center_pool.pop(best_indices[1])
            right_pool.pop(best_indices[2])

    return matched_triplets, left_pool, center_pool, right_pool


def edu_stub(edu: Dict) -> Dict:
    return {
        "edu_id": edu.get("edu_id"),
        "bias": edu.get("bias"),
        "depth": edu.get("depth", 0),
        "text": edu.get("text", ""),
    }


def build_bipartite_coverage_with_trace(
    fact_row: Dict,
) -> Tuple[Tuple[List[float], List[Dict]], Tuple[List[float], List[Dict]]]:
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left: List[float] = []
    features_right: List[float] = []
    trace_left: List[Dict] = []
    trace_right: List[Dict] = []

    for cluster_id, edus in clusters.items():
        left_edus = get_side_edus(edus, edu_lookup, "left")
        center_edus = get_side_edus(edus, edu_lookup, "center")
        right_edus = get_side_edus(edus, edu_lookup, "right")

        matched, leftover_l, leftover_c, leftover_r = greedy_bipartite_match(
            left_edus, center_edus, right_edus
        )

        for l_edu, c_edu, r_edu in matched:
            features_left.append(0.0)
            trace_left.append(
                {
                    "cluster_id": str(cluster_id),
                    "event_type": "matched_triplet",
                    "feature_value": 0.0,
                    "left_edu": edu_stub(l_edu),
                    "center_edu": edu_stub(c_edu),
                    "right_edu": edu_stub(r_edu),
                }
            )

            features_right.append(0.0)
            trace_right.append(
                {
                    "cluster_id": str(cluster_id),
                    "event_type": "matched_triplet",
                    "feature_value": 0.0,
                    "left_edu": edu_stub(l_edu),
                    "center_edu": edu_stub(c_edu),
                    "right_edu": edu_stub(r_edu),
                }
            )

        for l_edu in leftover_l:
            features_left.append(1.0)
            trace_left.append(
                {
                    "cluster_id": str(cluster_id),
                    "event_type": "left_only_mention",
                    "feature_value": 1.0,
                    "left_edu": edu_stub(l_edu),
                }
            )

        for c_edu in leftover_c:
            features_left.append(-1.0)
            trace_left.append(
                {
                    "cluster_id": str(cluster_id),
                    "event_type": "center_only_mention",
                    "feature_value": -1.0,
                    "center_edu": edu_stub(c_edu),
                }
            )

            features_right.append(-1.0)
            trace_right.append(
                {
                    "cluster_id": str(cluster_id),
                    "event_type": "center_only_mention",
                    "feature_value": -1.0,
                    "center_edu": edu_stub(c_edu),
                }
            )

        for r_edu in leftover_r:
            features_right.append(1.0)
            trace_right.append(
                {
                    "cluster_id": str(cluster_id),
                    "event_type": "right_only_mention",
                    "feature_value": 1.0,
                    "right_edu": edu_stub(r_edu),
                }
            )

    return (features_left, trace_left), (features_right, trace_right)


def pad_or_truncate(vec: List[float], target_len: int) -> np.ndarray:
    arr = np.zeros((target_len,), dtype=float)
    lim = min(len(vec), target_len)
    if lim > 0:
        arr[:lim] = np.array(vec[:lim], dtype=float)
    return arr


def explain_with_tree_paths(model, x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Saabas-style local contributions for P(class=1)."""
    n_feat = int(model.n_features_in_)
    contrib = np.zeros((n_feat,), dtype=float)
    base = 0.0

    for est in model.estimators_:
        tree = est.tree_

        def prob1(node_id: int) -> float:
            value = tree.value[node_id][0]
            total = float(np.sum(value))
            if total <= 0.0:
                return 0.0
            return float(value[1] / total)

        node = 0
        base += prob1(node)

        while tree.feature[node] >= 0:
            feat_idx = int(tree.feature[node])
            threshold = float(tree.threshold[node])
            left = int(tree.children_left[node])
            right = int(tree.children_right[node])

            p_node = prob1(node)
            child = left if x[feat_idx] <= threshold else right
            p_child = prob1(child)

            contrib[feat_idx] += p_child - p_node
            node = child

    n_trees = float(len(model.estimators_))
    if n_trees == 0:
        return 0.0, contrib
    return base / n_trees, contrib / n_trees


def build_test_samples(facts_rows: List[dict], test_ids: Set[int]) -> List[Dict]:
    samples: List[Dict] = []

    for row in facts_rows:
        triplet_idx = row.get("triplet_idx")
        if triplet_idx not in test_ids:
            continue

        (feat_left, trace_left), (feat_right, trace_right) = (
            build_bipartite_coverage_with_trace(row)
        )

        samples.append(
            {
                "sample_id": f"triplet_{triplet_idx}_left",
                "triplet_idx": int(triplet_idx),
                "sample_side": "left",
                "true_label": 0,
                "vector": feat_left,
                "trace": trace_left,
                "triplet": row.get("triplet", {}),
            }
        )

        samples.append(
            {
                "sample_id": f"triplet_{triplet_idx}_right",
                "triplet_idx": int(triplet_idx),
                "sample_side": "right",
                "true_label": 1,
                "vector": feat_right,
                "trace": trace_right,
                "triplet": row.get("triplet", {}),
            }
        )

    samples.sort(key=lambda s: (s["triplet_idx"], s["sample_side"]))
    return samples


def label_name(label: int) -> str:
    return "left-vs-center" if int(label) == 0 else "right-vs-center"


def event_to_text(event: Dict, sample_side: str) -> str:
    et = event.get("event_type", "unknown")
    if et == "center_only_mention":
        center = event.get("center_edu", {})
        return (
            f"Center mentions this fact but {sample_side} side does not: "
            f"{center.get('edu_id')} :: {center.get('text', '')}"
        )
    if et == "left_only_mention":
        left = event.get("left_edu", {})
        return f"Left-only mention: {left.get('edu_id')} :: {left.get('text', '')}"
    if et == "right_only_mention":
        right = event.get("right_edu", {})
        return f"Right-only mention: {right.get('edu_id')} :: {right.get('text', '')}"
    if et == "matched_triplet":
        left = event.get("left_edu", {})
        center = event.get("center_edu", {})
        right = event.get("right_edu", {})
        return (
            "Matched triplet (no coverage asymmetry): "
            f"L={left.get('edu_id')}, C={center.get('edu_id')}, R={right.get('edu_id')}"
        )
    return str(event)


def choose_top_evidence(
    x: np.ndarray,
    trace: List[Dict],
    support_for_pred: np.ndarray,
    global_importance: np.ndarray,
    sample_side: str,
    top_k: int,
) -> List[Dict]:
    candidates: List[Dict] = []
    max_trace_idx = len(trace) - 1

    for i, value in enumerate(x.tolist()):
        if abs(value) < 1e-12:
            continue
        if i > max_trace_idx:
            continue
        ev = trace[i]
        candidates.append(
            {
                "feature_index": i,
                "feature_value": float(value),
                "support_for_predicted_class": float(support_for_pred[i]),
                "global_feature_importance": float(global_importance[i]),
                "event": ev,
                "event_text": event_to_text(ev, sample_side),
            }
        )

    if not candidates:
        return []

    positive = [c for c in candidates if c["support_for_predicted_class"] > 0]
    if positive:
        positive.sort(
            key=lambda c: (
                c["support_for_predicted_class"],
                c["global_feature_importance"],
            ),
            reverse=True,
        )
        return positive[:top_k]

    candidates.sort(
        key=lambda c: (
            abs(c["support_for_predicted_class"]),
            c["global_feature_importance"],
        ),
        reverse=True,
    )
    return candidates[:top_k]


def write_markdown_report(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines: List[str] = []
    lines.append("# Explainable Prediction Report")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Model: `{payload['setup']['model_path']}`")
    lines.append(f"- Input dimension: {payload['setup']['input_dim']}")
    lines.append(
        f"- Test binary vectors available: {payload['setup']['num_test_vectors_total']}"
    )
    lines.append(f"- Sampled vectors: {payload['setup']['num_samples']}")
    lines.append("")
    lines.append("## Sampled Accuracy")
    lines.append(
        f"- Correct: {payload['summary']['correct']} / {payload['summary']['total']}"
    )
    lines.append(f"- Accuracy: {payload['summary']['accuracy']:.4f}")
    lines.append("")
    lines.append("## Per-sample Explanations")
    lines.append("")

    for row in payload["samples"]:
        lines.append(f"### {row['sample_id']}")
        lines.append(
            f"- True: `{row['true_label_name']}` | Pred: `{row['pred_label_name']}` | Correct: `{row['is_correct']}`"
        )
        lines.append(
            f"- P(left-vs-center)={row['proba_left_vs_center']:.4f}, P(right-vs-center)={row['proba_right_vs_center']:.4f}"
        )
        lines.append("- Top EDU-level evidence:")

        top = row.get("top_evidence", [])
        if not top:
            lines.append("  - (no non-zero active features found)")
        else:
            for ev in top:
                lines.append(
                    "  - "
                    + f"f{ev['feature_index']} val={ev['feature_value']:+.1f} "
                    + f"support={ev['support_for_predicted_class']:+.4f} :: {ev['event_text']}"
                )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    facts = load_json(args.facts)
    test_rows = load_json(os.path.join(args.split_dir, "test.json"))
    test_ids = get_split_triplet_idx(test_rows)

    with open(args.model, "rb") as f:
        model_payload = pickle.load(f)

    model = model_payload["model"]
    input_dim = int(model_payload["input_dim"])
    global_importance = np.array(model.feature_importances_, dtype=float)

    all_samples = build_test_samples(facts, test_ids)
    if not all_samples:
        raise RuntimeError("No test samples were built. Check facts/split paths.")

    n = min(args.num_samples, len(all_samples))
    rng = random.Random(args.seed)
    chosen = rng.sample(all_samples, n)

    explained_rows = []
    correct = 0

    for sample in chosen:
        raw_vec = sample["vector"]
        x = pad_or_truncate(raw_vec, input_dim)
        proba = model.predict_proba(x.reshape(1, -1))[0]
        pred = int(model.predict(x.reshape(1, -1))[0])
        true = int(sample["true_label"])

        if pred == true:
            correct += 1

        _, contrib_class1 = explain_with_tree_paths(model, x)
        support_for_pred = contrib_class1 if pred == 1 else -contrib_class1

        top_evidence = choose_top_evidence(
            x=x,
            trace=sample["trace"],
            support_for_pred=support_for_pred,
            global_importance=global_importance,
            sample_side=sample["sample_side"],
            top_k=args.top_k,
        )

        explained_rows.append(
            {
                "sample_id": sample["sample_id"],
                "triplet_idx": sample["triplet_idx"],
                "sample_side": sample["sample_side"],
                "triplet": sample["triplet"],
                "true_label": true,
                "true_label_name": label_name(true),
                "pred_label": pred,
                "pred_label_name": label_name(pred),
                "is_correct": bool(pred == true),
                "proba_left_vs_center": float(proba[0]),
                "proba_right_vs_center": float(proba[1]),
                "active_feature_count": int(np.count_nonzero(x)),
                "raw_feature_length": int(len(raw_vec)),
                "top_evidence": top_evidence,
            }
        )

    explained_rows.sort(key=lambda r: r["sample_id"])

    payload = {
        "setup": {
            "facts_path": args.facts,
            "split_dir": args.split_dir,
            "model_path": args.model,
            "experiment": model_payload.get("experiment"),
            "input_dim": input_dim,
            "seed": args.seed,
            "num_samples": n,
            "top_k": args.top_k,
            "num_test_vectors_total": len(all_samples),
            "method": {
                "feature_builder": "alt3_bipartite_coverage",
                "local_explanation": "tree-path (Saabas-style) contributions",
                "evidence_mapping": "feature index -> matched/unmatched EDU event",
            },
        },
        "summary": {
            "total": n,
            "correct": int(correct),
            "accuracy": float(correct / n) if n > 0 else 0.0,
        },
        "samples": explained_rows,
    }

    save_json(args.out_json, payload)
    write_markdown_report(args.out_md, payload)

    print("Explainable prediction run complete")
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved report: {args.out_md}")
    print(
        f"Sampled accuracy: {payload['summary']['correct']}/{payload['summary']['total']}"
        f" = {payload['summary']['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
