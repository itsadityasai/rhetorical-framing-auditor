import argparse
import json
import os
import time
from datetime import datetime

import yaml

from modules.FactCluster import FactCluster


with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

RST_OUTPUT_DIR = params["paths"]["dirs"]["rst_output"]
DATA_DIR = params["paths"]["dirs"]["data"]
MIN_ARTICLES_PER_TRIPLET = params["fact_clustering"]["min_articles_per_triplet"]
REFINE_THRESHOLD = params["fact_clustering"]["refine_threshold"]
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUT = f"data/valid_cluster_results_recluster_centerpair_{RUN_STAMP}.json"
DEFAULT_META = f"data/valid_cluster_results_recluster_centerpair_{RUN_STAMP}_meta.json"


def resolve_article_path(path):
    if os.path.exists(path):
        return path
    candidate = os.path.join(DATA_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path


def load_triplets(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_article(path):
    real_path = resolve_article_path(path)
    with open(real_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "article_id": data["ID"],
        "bias": data["bias_text"],
    }


def load_rst_edus(article_id):
    rst_path = os.path.join(RST_OUTPUT_DIR, f"{article_id}.json")
    if not os.path.exists(rst_path):
        return None
    with open(rst_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("edus", [])


def prepare_article(path):
    article_meta = load_article(path)
    edus = load_rst_edus(article_meta["article_id"])
    if edus is None:
        return None
    return {
        "article_id": article_meta["article_id"],
        "bias": article_meta["bias"],
        "edus": edus,
    }


def process_triplet(triplet):
    articles = []
    for bias_key in ["left", "center", "right"]:
        article = prepare_article(triplet[bias_key])
        if article:
            articles.append(article)

    if len(articles) < MIN_ARTICLES_PER_TRIPLET:
        return None, None

    fc = FactCluster(articles)
    fc.refine_clusters(refine_threshold=REFINE_THRESHOLD)

    clusters = fc.get_clusters()
    edu_lookup = {}

    for _, edu_ids in clusters.items():
        for edu_id in edu_ids:
            if edu_id in fc.edu_lookup:
                edu_lookup[edu_id] = {
                    "text": fc.edu_lookup[edu_id]["text"],
                    "bias": fc.edu_lookup[edu_id]["bias"],
                }

    return clusters, edu_lookup, fc.filter_stats


def print_progress(current, total, elapsed, empties, errors):
    if current > 0:
        eta_seconds = (elapsed / current) * (total - current)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_str = f"ETA: {eta_min}m {eta_sec}s"
    else:
        eta_str = "ETA: --"

    print(f"\r{current}/{total} - {eta_str} - {empties} empty, {errors} errors", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run EDU fact clustering on a triplet file")
    parser.add_argument("--triplets", default="data/valid_triplets.json", help="Input triplet JSON path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output cluster JSON path")
    parser.add_argument("--meta", default=DEFAULT_META, help="Output metadata JSON path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output files if they already exist.",
    )
    args = parser.parse_args()

    for out_path in [args.out, args.meta]:
        if os.path.exists(out_path) and not args.overwrite:
            raise FileExistsError(
                f"Output file already exists: {out_path}. Use a new path or pass --overwrite."
            )

    triplets = load_triplets(args.triplets)
    total = len(triplets)

    print(f"Processing {total} triplets")

    all_results = []
    successful = 0
    errors = 0
    empties = 0
    start_time = time.time()

    total_input_edus = 0
    total_kept_edus = 0
    total_dropped_edus = 0

    for i, triplet in enumerate(triplets):
        elapsed = time.time() - start_time
        print_progress(i + 1, total, elapsed, empties, errors)

        try:
            result = process_triplet(triplet)
            if result is None or result[0] is None:
                empties += 1
                continue

            clusters, edu_lookup, filter_stats = result
            total_input_edus += filter_stats["input_edus"]
            total_kept_edus += filter_stats["kept_edus"]
            total_dropped_edus += filter_stats["dropped_edus"]

            if clusters and len(clusters) > 0:
                all_results.append(
                    {
                        "triplet_idx": i,
                        "triplet": triplet,
                        "clusters": clusters,
                        "edu_lookup": edu_lookup,
                    }
                )
                successful += 1
            else:
                empties += 1
        except Exception as e:
            errors += 1
            print(f"\n[ERROR] Triplet {i}: {e}")

    print()
    total_time = time.time() - start_time

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    meta = {
        "input_triplets": args.triplets,
        "output_clusters": args.out,
        "total_triplets": total,
        "successful_triplets_with_clusters": successful,
        "errors": errors,
        "empty_or_dropped_triplets": empties,
        "total_clusters": sum(len(r.get("clusters", {})) for r in all_results),
        "total_clustered_edus": sum(len(r.get("edu_lookup", {})) for r in all_results),
        "edu_filter_stats": {
            "input_edus": total_input_edus,
            "kept_edus": total_kept_edus,
            "dropped_edus": total_dropped_edus,
            "drop_ratio": (total_dropped_edus / total_input_edus) if total_input_edus else 0.0,
        },
        "elapsed_seconds": round(total_time, 3),
    }

    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved clusters: {args.out}")
    print(f"Saved metadata: {args.meta}")
    print(f"Successful triplets: {successful} / {total}")
    print(f"Total clusters: {meta['total_clusters']}")


if __name__ == "__main__":
    main()
