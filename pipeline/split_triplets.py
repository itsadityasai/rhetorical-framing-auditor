import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def article_id(path_str):
    return Path(path_str).stem


def load_triplets(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_triplet_docs(triplets):
    docs_per_triplet = []
    for row in triplets:
        docs = {
            article_id(row["left"]),
            article_id(row["center"]),
            article_id(row["right"]),
        }
        docs_per_triplet.append(docs)
    return docs_per_triplet


def build_components(docs_per_triplet):
    doc_to_triplets = defaultdict(list)
    for i, docs in enumerate(docs_per_triplet):
        for d in docs:
            doc_to_triplets[d].append(i)

    unseen = set(range(len(docs_per_triplet)))
    components = []

    while unseen:
        start = unseen.pop()
        queue = deque([start])
        comp = [start]

        while queue:
            cur = queue.popleft()
            for d in docs_per_triplet[cur]:
                for nb in doc_to_triplets[d]:
                    if nb in unseen:
                        unseen.remove(nb)
                        queue.append(nb)
                        comp.append(nb)

        components.append(comp)

    return components


def component_metadata(components, docs_per_triplet, missing_docs):
    sizes = []
    has_missing = []

    for comp in components:
        sizes.append(len(comp))
        comp_docs = set()
        for idx in comp:
            comp_docs.update(docs_per_triplet[idx])
        has_missing.append(len(comp_docs & missing_docs) > 0)

    return sizes, has_missing


def solve_exact_ratio(sizes, has_missing, targets):
    num_components = len(sizes)
    num_splits = 3  # train, val, test
    num_vars = num_components * num_splits

    # Variables x[c,s] in {0,1}
    integrality = np.ones(num_vars, dtype=int)
    bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))

    constraints = []

    # Each component must go to exactly one split.
    A_assign = lil_matrix((num_components, num_vars), dtype=float)
    for c in range(num_components):
        for s in range(num_splits):
            A_assign[c, c * num_splits + s] = 1.0
    constraints.append(LinearConstraint(A_assign.tocsc(), np.ones(num_components), np.ones(num_components)))

    # Components containing missing RST docs cannot go to train.
    if any(has_missing):
        A_missing = lil_matrix((sum(has_missing), num_vars), dtype=float)
        row = 0
        for c, miss in enumerate(has_missing):
            if miss:
                A_missing[row, c * num_splits + 0] = 1.0  # train column
                row += 1
        constraints.append(LinearConstraint(A_missing.tocsc(), np.zeros(row), np.zeros(row)))

    # Exact split size constraints.
    for split_idx, target in enumerate(targets):
        A_size = lil_matrix((1, num_vars), dtype=float)
        for c, size in enumerate(sizes):
            A_size[0, c * num_splits + split_idx] = size
        constraints.append(LinearConstraint(A_size.tocsc(), np.array([target], dtype=float), np.array([target], dtype=float)))

    c_obj = np.zeros(num_vars)  # feasibility solve
    res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality, options={"disp": False, "presolve": True})

    if not res.success:
        return None

    x = np.rint(res.x).astype(int)
    return x


def solve_best_effort(sizes, has_missing, targets):
    num_components = len(sizes)
    # Decision vars:
    # x[c,train], x[c,val], x[c,test] for each component (binary)
    # d_val_pos, d_val_neg, d_test_pos, d_test_neg (continuous >=0)
    num_x = num_components * 3
    idx_dvp = num_x + 0
    idx_dvn = num_x + 1
    idx_dtp = num_x + 2
    idx_dtn = num_x + 3
    num_vars = num_x + 4

    integrality = np.zeros(num_vars, dtype=int)
    integrality[:num_x] = 1

    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    ub[num_x:] = np.inf
    bounds = Bounds(lb, ub)

    constraints = []

    # Each component assigned to exactly one split.
    A_assign = lil_matrix((num_components, num_vars), dtype=float)
    for c in range(num_components):
        A_assign[c, c * 3 + 0] = 1.0
        A_assign[c, c * 3 + 1] = 1.0
        A_assign[c, c * 3 + 2] = 1.0
    constraints.append(LinearConstraint(A_assign.tocsc(), np.ones(num_components), np.ones(num_components)))

    # Missing components not allowed in train.
    if any(has_missing):
        A_missing = lil_matrix((sum(has_missing), num_vars), dtype=float)
        row = 0
        for c, miss in enumerate(has_missing):
            if miss:
                A_missing[row, c * 3 + 0] = 1.0
                row += 1
        constraints.append(LinearConstraint(A_missing.tocsc(), np.zeros(row), np.zeros(row)))

    # val_size - target_val = d_val_pos - d_val_neg
    A_val = lil_matrix((1, num_vars), dtype=float)
    for c, size in enumerate(sizes):
        A_val[0, c * 3 + 1] = size
    A_val[0, idx_dvp] = -1.0
    A_val[0, idx_dvn] = 1.0
    constraints.append(LinearConstraint(A_val.tocsc(), np.array([targets[1]], dtype=float), np.array([targets[1]], dtype=float)))

    # test_size - target_test = d_test_pos - d_test_neg
    A_test = lil_matrix((1, num_vars), dtype=float)
    for c, size in enumerate(sizes):
        A_test[0, c * 3 + 2] = size
    A_test[0, idx_dtp] = -1.0
    A_test[0, idx_dtn] = 1.0
    constraints.append(LinearConstraint(A_test.tocsc(), np.array([targets[2]], dtype=float), np.array([targets[2]], dtype=float)))

    # Objective: maximize train size, then keep val/test close to targets.
    # milp minimizes, so use negative weight for train.
    c_obj = np.zeros(num_vars)
    for c, size in enumerate(sizes):
        c_obj[c * 3 + 0] = -1000.0 * size
    c_obj[idx_dvp] = 1.0
    c_obj[idx_dvn] = 1.0
    c_obj[idx_dtp] = 1.0
    c_obj[idx_dtn] = 1.0

    res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality, options={"disp": False, "presolve": True})

    if not res.success:
        raise RuntimeError(f"Best-effort split MILP failed: {res.message}")

    x = np.rint(res.x[:num_x]).astype(int)
    return x


def decode_assignment(x, components):
    split_map = {"train": [], "val": [], "test": []}
    for c, comp in enumerate(components):
        if x[c * 3 + 0] == 1:
            split_map["train"].extend(comp)
        elif x[c * 3 + 1] == 1:
            split_map["val"].extend(comp)
        elif x[c * 3 + 2] == 1:
            split_map["test"].extend(comp)
        else:
            raise RuntimeError(f"Component {c} was not assigned to a split")
    return split_map


def collect_docs_for_indices(indices, docs_per_triplet):
    docs = set()
    for i in indices:
        docs.update(docs_per_triplet[i])
    return docs


def save_split(out_dir, name, rows):
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe triplet split with missing-RST exclusion from train")
    parser.add_argument("--triplets", default="data/capped_triplets.json", help="Input triplets JSON")
    parser.add_argument("--rst-output-dir", default="data/rst_output", help="Directory containing RST tree JSON files")
    parser.add_argument("--out-dir", default="data/triplet_splits", help="Output directory for train/val/test split files")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--strict-ratio", action="store_true", help="Fail if exact ratio is infeasible under hard constraints")
    args = parser.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    triplet_path = Path(args.triplets)
    rst_output_dir = Path(args.rst_output_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    triplets = load_triplets(triplet_path)
    total_triplets = len(triplets)

    docs_per_triplet = collect_triplet_docs(triplets)
    all_docs = set().union(*docs_per_triplet) if docs_per_triplet else set()
    available_rst_docs = {p.stem for p in rst_output_dir.glob("*.json")}
    missing_docs = all_docs - available_rst_docs

    components = build_components(docs_per_triplet)
    sizes, has_missing = component_metadata(components, docs_per_triplet, missing_docs)

    target_train = int(total_triplets * args.train_ratio)
    target_val = int(total_triplets * args.val_ratio)
    target_test = total_triplets - target_train - target_val
    targets = (target_train, target_val, target_test)

    exact_solution = solve_exact_ratio(sizes, has_missing, targets)

    if exact_solution is None and args.strict_ratio:
        raise RuntimeError(
            "Exact split ratio is infeasible under constraints (no article overlap + no missing-RST docs in train). "
            "Run without --strict-ratio to generate the closest feasible split."
        )

    if exact_solution is None:
        assignment = solve_best_effort(sizes, has_missing, targets)
        mode = "best_effort"
    else:
        assignment = exact_solution
        mode = "exact"

    split_indices = decode_assignment(assignment, components)

    train_rows = [triplets[i] for i in split_indices["train"]]
    val_rows = [triplets[i] for i in split_indices["val"]]
    test_rows = [triplets[i] for i in split_indices["test"]]

    # Safety checks
    train_docs = collect_docs_for_indices(split_indices["train"], docs_per_triplet)
    val_docs = collect_docs_for_indices(split_indices["val"], docs_per_triplet)
    test_docs = collect_docs_for_indices(split_indices["test"], docs_per_triplet)

    if (train_docs & val_docs) or (train_docs & test_docs) or (val_docs & test_docs):
        raise RuntimeError("Split leakage detected: an article appears in multiple sets")

    if train_docs & missing_docs:
        raise RuntimeError("Constraint violation: train contains articles with missing RST trees")

    save_split(out_dir, "train", train_rows)
    save_split(out_dir, "val", val_rows)
    save_split(out_dir, "test", test_rows)

    meta = {
        "mode": mode,
        "input_triplets": str(triplet_path),
        "rst_output_dir": str(rst_output_dir),
        "targets": {"train": target_train, "val": target_val, "test": target_test},
        "achieved": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "ratios": {
            "train": round(len(train_rows) / total_triplets, 6) if total_triplets else 0.0,
            "val": round(len(val_rows) / total_triplets, 6) if total_triplets else 0.0,
            "test": round(len(test_rows) / total_triplets, 6) if total_triplets else 0.0,
        },
        "components": {
            "count": len(components),
            "largest": max(sizes) if sizes else 0,
            "singletons": sum(1 for s in sizes if s == 1),
            "with_missing_rst": sum(1 for m in has_missing if m),
        },
        "docs": {
            "unique_in_triplets": len(all_docs),
            "with_rst_available": len(all_docs & available_rst_docs),
            "missing_rst": len(missing_docs),
            "train": len(train_docs),
            "val": len(val_docs),
            "test": len(test_docs),
        },
        "overlap": {
            "train_val": len(train_docs & val_docs),
            "train_test": len(train_docs & test_docs),
            "val_test": len(val_docs & test_docs),
        },
        "train_contains_missing_rst": len(train_docs & missing_docs),
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Split complete")
    print(f"Mode: {mode}")
    print(f"Target triplets train/val/test: {target_train}/{target_val}/{target_test}")
    print(f"Achieved triplets train/val/test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"Missing-RST docs in triplet universe: {len(missing_docs)}")
    print(f"Train docs with missing RST: {len(train_docs & missing_docs)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
