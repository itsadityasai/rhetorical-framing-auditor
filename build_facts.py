import json
import os
import time
from modules.FactCluster import FactCluster
import yaml

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

CLUSTERS_PATH = params["paths"]["files"]["clusters"]
FACTS_PATH = params["paths"]["files"]["facts"]



def load_clusters(path):
    with open(path, "r") as f:
        return json.load(f)


def print_progress(current, total, elapsed):
    if current > 0:
        eta_seconds = (elapsed / current) * (total - current)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_str = f"ETA: {eta_min}m {eta_sec}s"
    else:
        eta_str = "ETA: --"

    print(f"\r{current}/{total} - {eta_str}   ", end="", flush=True)


if __name__ == "__main__":
    cluster_results = load_clusters(CLUSTERS_PATH)
    total = len(cluster_results)

    print(f"Building facts for {total} triplets\n")

    all_facts = []
    errors = 0
    start_time = time.time()

    for i, cluster_result in enumerate(cluster_results):
        elapsed = time.time() - start_time
        print_progress(i + 1, total, elapsed)
        try:
            fact_result = FactCluster.build_facts(cluster_result)
            all_facts.append(fact_result)
        except Exception as e:
            errors += 1
            print(f"\n[ERROR] Triplet {cluster_result.get('triplet_idx', i)}: {e}")

    print()
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s: {len(all_facts)} built, {errors} errors")

    with open(FACTS_PATH, "w") as f:
        json.dump(all_facts, f, indent=2)

    print(f"\nSaved {len(all_facts)} fact results to {FACTS_PATH}")
