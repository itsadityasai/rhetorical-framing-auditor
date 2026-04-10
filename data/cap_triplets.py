#!/usr/bin/env python3
"""Cap article participation across triplets and save the best retained subset.

Optimization objective:
- Keep as many triplets as possible
- Subject to: each article appears in at most `cap` kept triplets
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def article_id(path_str: str) -> str:
    return Path(path_str).stem


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_counts(triplets):
    counts = Counter()
    for t in triplets:
        counts[article_id(t["left"])] += 1
        counts[article_id(t["center"])] += 1
        counts[article_id(t["right"])] += 1
    return counts


def solve_cap_milp(triplets, cap: int):
    article_to_idx = {}
    incidence_rows = []

    for t in triplets:
        ids = [article_id(t["left"]), article_id(t["center"]), article_id(t["right"])]
        idxs = []
        for aid in ids:
            if aid not in article_to_idx:
                article_to_idx[aid] = len(article_to_idx)
            idxs.append(article_to_idx[aid])
        incidence_rows.append(tuple(idxs))

    m = len(article_to_idx)
    n = len(incidence_rows)

    # A[row=article, col=triplet] = 1 if article appears in triplet.
    A = lil_matrix((m, n), dtype=float)
    for j, (a, b, c) in enumerate(incidence_rows):
        A[a, j] = 1.0
        A[b, j] = 1.0
        A[c, j] = 1.0
    A = A.tocsc()

    # maximize sum(x) <=> minimize -sum(x)
    c = -np.ones(n)
    constraints = LinearConstraint(A, -np.inf * np.ones(m), cap * np.ones(m))
    bounds = Bounds(np.zeros(n), np.ones(n))
    integrality = np.ones(n, dtype=int)

    result = milp(
        c=c,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
        options={"disp": False, "presolve": True},
    )

    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    x = np.rint(result.x).astype(int)
    kept_idx = np.where(x == 1)[0].tolist()

    return kept_idx


def print_stats(input_triplets, output_triplets, cap: int, raw_dir: Path):
    in_counts = build_counts(input_triplets)
    out_counts = build_counts(output_triplets)

    total_raw_articles = len(list(raw_dir.glob("*.json")))

    print("CAP TRIPLET SUMMARY")
    print("=" * 64)
    print(f"Input triplets: {len(input_triplets)}")
    print(f"Output triplets: {len(output_triplets)}")
    print(f"Dropped triplets: {len(input_triplets) - len(output_triplets)}")
    print(f"Cap: {cap}")

    print("\nARTICLE COVERAGE")
    print("-" * 64)
    print(f"Raw corpus articles (all): {total_raw_articles}")
    print(f"Unique articles in input triplets: {len(in_counts)}")
    print(f"Unique articles in capped triplets: {len(out_counts)}")
    if total_raw_articles > 0:
        print(f"Participation rate in capped triplets: {len(out_counts) / total_raw_articles * 100.0:.2f}%")

    avg = (3 * len(output_triplets) / len(out_counts)) if out_counts else 0.0
    max_participation = max(out_counts.values()) if out_counts else 0
    print(f"Average triplets per participating article (capped): {avg:.6f}")
    print(f"Max participation in capped set: {max_participation}")

    freq = Counter(out_counts.values())
    print("\nTABLE (CAPPED AT 5)")
    print("-" * 64)
    print(f"{'Triplets per article':>20}    {'Article count':>14}")
    print("-" * 64)
    for k in range(1, cap + 1):
        print(f"{k:>20}    {freq.get(k, 0):>14}")


def main():
    parser = argparse.ArgumentParser(description="Cap article participation across triplets")
    parser.add_argument(
        "--input",
        default="unique_triplets.json",
        help="Input triplets JSON (default: unique_triplets.json)",
    )
    parser.add_argument(
        "--output",
        default="capped_triplets.json",
        help="Output triplets JSON (default: capped_triplets.json)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=5,
        help="Maximum triplets allowed per article (default: 5)",
    )
    parser.add_argument(
        "--raw-dir",
        default="raw/jsons",
        help="Raw article directory used for total corpus count (default: raw/jsons)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    raw_dir = Path(args.raw_dir)

    triplets = load_json(input_path)
    kept_idx = solve_cap_milp(triplets, cap=args.cap)
    capped_triplets = [triplets[i] for i in kept_idx]

    save_json(output_path, capped_triplets)

    print(f"Saved capped triplets to: {output_path}")
    print_stats(triplets, capped_triplets, args.cap, raw_dir)


if __name__ == "__main__":
    main()
