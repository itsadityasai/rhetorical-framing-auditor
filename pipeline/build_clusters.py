from modules.FactCluster import FactCluster
import json
import os
import time
import yaml
from modules.run_logger import init_run_logging, log_run_results, close_run_logging

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

TRIPLETS_PATH = params["paths"]["files"]["triplets"]
RST_OUTPUT_DIR = params["paths"]["dirs"]["rst_output"]
CLUSTERS_PATH = params["paths"]["files"]["clusters"]

MIN_ARTICLES_PER_TRIPLET = params["fact_clustering"]["min_articles_per_triplet"]
REFINE_THRESHOLD = params["fact_clustering"]["refine_threshold"]

RUN_LOG = init_run_logging(
    script_subdir="build_clusters",
    hyperparams={
        "triplets_path": TRIPLETS_PATH,
        "rst_output_dir": RST_OUTPUT_DIR,
        "clusters_path": CLUSTERS_PATH,
        "min_articles_per_triplet": MIN_ARTICLES_PER_TRIPLET,
        "refine_threshold": REFINE_THRESHOLD,
    },
)


def load_triplets(path):
    with open(path, "r") as f:
        return json.load(f)

def load_article(path): # path to raw article json
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "article_id": data["ID"],
        "bias": data["bias_text"]  # left, center, right
    }

def load_rst_edus(article_id): # rst->edus
    rst_path = os.path.join(RST_OUTPUT_DIR, f"{article_id}.json")
    if not os.path.exists(rst_path):
        return None
    with open(rst_path, "r") as f:
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
        "edus": edus
    }

def process_triplet(triplet):
    articles = []
    for bias_key in ["left", "center", "right"]:
        article = prepare_article(triplet[bias_key])
        if article:
            articles.append(article)
    
    if len(articles) < MIN_ARTICLES_PER_TRIPLET:
        return None, None
    
    # above check not really needed because we all triplets in bias_triplets.json have corresponding files,
    # but just there for good practice
    
    fc = FactCluster(articles)
    fc.refine_clusters(refine_threshold=REFINE_THRESHOLD)
    
    clusters = fc.get_clusters()
    edu_lookup = {}
    for cluster_id, edu_ids in clusters.items():
        for edu_id in edu_ids:
            if edu_id in fc.edu_lookup:
                edu_lookup[edu_id] = {
                    "text": fc.edu_lookup[edu_id]["text"],
                    "bias": fc.edu_lookup[edu_id]["bias"],
                }
    
    return clusters, edu_lookup

def print_progress(current, total, elapsed, empties, errors):
    if current > 0:
        eta_seconds = (elapsed / current) * (total - current)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_str = f"ETA: {eta_min}m {eta_sec}s"
    else:
        eta_str = "ETA: --"
    
    print(f"\r{current}/{total} - {eta_str} - {empties} empty, {errors} errors   ", end="", flush=True)

triplets = load_triplets(TRIPLETS_PATH)

total = len(triplets)
print(f"Processing {total} triplets\n")

all_results = []
successful = 0
errors = 0
empties = 0
start_time = time.time()

for i, triplet in enumerate(triplets):
    elapsed = time.time() - start_time
    print_progress(i + 1, total, elapsed, empties, errors)
    try:
        clusters, edu_lookup = process_triplet(triplet)
        if clusters and len(clusters) > 0:
            all_results.append({
                "triplet_idx": i,
                "triplet": triplet,
                "clusters": clusters,
                "edu_lookup": edu_lookup
            })
            successful += 1
        else:
            empties += 1
    except Exception as e:
        errors += 1
        print(f"\n[ERROR] Triplet {i}: {e}")

print()  
total_time = time.time() - start_time
print(f"\nCompleted in {total_time:.1f}s: {successful} successful, {errors} errors, {empties} empty clusters")

with open(CLUSTERS_PATH, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\nResults saved to {CLUSTERS_PATH}")

log_run_results(
    RUN_LOG,
    {
        "total_triplets": total,
        "successful": successful,
        "errors": errors,
        "empty_clusters": empties,
        "output_path": CLUSTERS_PATH,
        "elapsed_seconds": round(total_time, 3),
    },
)
close_run_logging(RUN_LOG, status="success")



