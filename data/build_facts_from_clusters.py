#!/usr/bin/env python3
"""Build fact rows from a cluster_results-style file."""

import argparse
import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from modules.FactCluster import FactCluster


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build facts from cluster artifact")
    parser.add_argument("--clusters", default="data/valid_cluster_results.json")
    parser.add_argument("--out", default="data/valid_facts_results.json")
    parser.add_argument("--meta", default="data/valid_facts_results_meta.json")
    args = parser.parse_args()

    cluster_rows = load_json(args.clusters)
    total = len(cluster_rows)

    facts = []
    errors = 0
    total_clusters = 0
    start = time.time()

    for i, row in enumerate(cluster_rows):
        try:
            fact_row = FactCluster.build_facts(row)
            facts.append(fact_row)
            total_clusters += len(row.get("clusters", {}))
        except Exception as e:
            errors += 1
            print(f"[ERROR] Row {i}: {e}")

    save_json(args.out, facts)

    meta = {
        "input_clusters": args.clusters,
        "output_facts": args.out,
        "input_rows": total,
        "output_rows": len(facts),
        "total_clusters_processed": total_clusters,
        "errors": errors,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    save_json(args.meta, meta)

    print("FACT BUILD SUMMARY")
    print("=" * 64)
    print(f"Input rows: {total}")
    print(f"Output rows: {len(facts)}")
    print(f"Clusters processed: {total_clusters}")
    print(f"Errors: {errors}")
    print(f"Saved: {args.out}")
    print(f"Saved: {args.meta}")


if __name__ == "__main__":
    main()
