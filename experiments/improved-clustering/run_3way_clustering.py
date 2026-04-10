"""
Improved 3-Way Clustering Script

Problem: Current clustering produces only ~18% 3-way clusters (all 3 biases present).
Root cause: Agglomerative clustering finds semantic similarity but left/right sources
often cover different aspects - they rarely report the exact same fact unless center
also does.

Solution: Anchor-based approach
1. Use CENTER EDUs as anchors (center is required for DFI calculation anyway)
2. For each center EDU, find the best matching LEFT and RIGHT EDU above threshold
3. Only keep clusters where we successfully find matches from BOTH sides
4. This guarantees 100% 3-way cluster coverage

This trades quantity (fewer total clusters) for quality (all clusters have all 3 biases).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder


# Load params
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = str(
    next(
        (p for p in Path(__file__).resolve().parents if (p / "params.yaml").exists()),
        Path(__file__).resolve().parent,
    )
)
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Model params
SBERT_MODEL_NAME = params["models"]["sbert"]["model_name"]
SBERT_ENCODE_BATCH_SIZE = params["models"]["sbert"]["encode_batch_size"]
SBERT_NORMALIZE = params["models"]["sbert"]["normalize_embeddings"]

CROSS_ENCODER_NAME = params["models"]["cross_encoder"]["model_name"]
CROSS_ENCODER_BATCH_SIZE = params["models"]["cross_encoder"]["predict_batch_size"]

# Clustering params
AGGLOMERATIVE_PARAMS = params["fact_clustering"]["agglomerative"]
PAIR_VALIDATION_THRESHOLD = params["fact_clustering"]["pair_validation"]["threshold"]
DEFAULT_REFINE_THRESHOLD = params["fact_clustering"]["refine_threshold"]

# EDU filter params (reuse existing filters)
EDU_FILTER_PARAMS = params["fact_clustering"].get("edu_filter", {})
EDU_FILTER_ENABLED = EDU_FILTER_PARAMS.get("enabled", True)
EDU_FILTER_MIN_TOKENS = EDU_FILTER_PARAMS.get("min_tokens", 3)
EDU_FILTER_DROP_PUNCT_ONLY = EDU_FILTER_PARAMS.get("drop_punct_only", True)
EDU_FILTER_DROP_URL_LIKE = EDU_FILTER_PARAMS.get("drop_url_like", True)
EDU_FILTER_DROP_SOCIAL_META = EDU_FILTER_PARAMS.get("drop_social_meta", True)
EDU_FILTER_DROP_SHORT_ATTR = EDU_FILTER_PARAMS.get("drop_short_attribution", True)
EDU_FILTER_SHORT_ATTR_MAX_TOKENS = EDU_FILTER_PARAMS.get(
    "short_attribution_max_tokens", 8
)
EDU_FILTER_BOILERPLATE_PHRASES = [
    p.lower()
    for p in EDU_FILTER_PARAMS.get(
        "boilerplate_phrases",
        [
            "story highlights",
            "just watched",
            "must watch",
            "read more",
            "add interest",
            "more :",
            "follow us",
            "newsletter",
        ],
    )
]

# Path params
DATA_DIR = os.path.join(PROJECT_ROOT, params["paths"]["dirs"]["data"])
RST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, params["paths"]["dirs"]["rst_output"])

# Regex patterns for EDU filtering
import re

URL_OR_SOCIAL_PATTERN = re.compile(
    r"https?://|www\.|\.com\b|\.org\b|pic\.twitter|twitter\.com|<\s*a\s+href",
    re.IGNORECASE,
)
PUNCT_ONLY_PATTERN = re.compile(r"[^A-Za-z0-9]+")
SHORT_ATTR_PATTERN = re.compile(
    r"\b(said|says|told|according to|asked|wrote|added)\b", re.IGNORECASE
)


# Default output paths
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_OUT_DIR = os.path.join(SCRIPT_DIR, "output")


def token_count(text: str) -> int:
    return len(text.split())


def is_edu_fact_candidate(text: Optional[str]) -> bool:
    """Check if EDU passes quality filters (same logic as original)."""
    if text is None:
        return False

    text = text.strip()
    if not text:
        return False

    if not EDU_FILTER_ENABLED:
        return True

    tc = token_count(text)

    if tc < EDU_FILTER_MIN_TOKENS:
        return False

    if EDU_FILTER_DROP_PUNCT_ONLY and PUNCT_ONLY_PATTERN.fullmatch(text):
        return False

    lowered = text.lower()

    if EDU_FILTER_DROP_URL_LIKE and URL_OR_SOCIAL_PATTERN.search(lowered):
        return False

    if EDU_FILTER_DROP_SOCIAL_META and any(
        p in lowered for p in EDU_FILTER_BOILERPLATE_PHRASES
    ):
        return False

    if (
        EDU_FILTER_DROP_SHORT_ATTR
        and tc <= EDU_FILTER_SHORT_ATTR_MAX_TOKENS
        and SHORT_ATTR_PATTERN.search(lowered)
    ):
        return False

    return True


def load_triplets(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_article_path(path: str) -> str:
    if os.path.exists(path):
        return path
    candidate = os.path.join(DATA_DIR, path)
    if os.path.exists(candidate):
        return candidate
    # Try project root
    candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    return path


def load_article_meta(path: str) -> dict:
    real_path = resolve_article_path(path)
    with open(real_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "article_id": data["ID"],
        "bias": data["bias_text"],
    }


def load_rst_edus(article_id: str) -> Optional[List[dict]]:
    rst_path = os.path.join(RST_OUTPUT_DIR, f"{article_id}.json")
    if not os.path.exists(rst_path):
        return None
    with open(rst_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("edus", [])


def prepare_edus_for_article(path: str, bias: str) -> Tuple[List[dict], dict]:
    """
    Load and filter EDUs for an article.
    Returns (filtered_edus, stats).
    """
    meta = load_article_meta(path)
    article_id = meta["article_id"]
    edus = load_rst_edus(article_id)

    if edus is None:
        return [], {"input": 0, "kept": 0, "dropped": 0}

    stats = {"input": len(edus), "kept": 0, "dropped": 0}
    filtered = []

    for edu in edus:
        text = (edu.get("text") or "").strip()
        if not is_edu_fact_candidate(text):
            stats["dropped"] += 1
            continue

        filtered.append(
            {
                "id": f"{article_id}_{edu['id']}",
                "text": text,
                "bias": bias,
                "article_id": article_id,
                "local_id": edu["id"],
            }
        )
        stats["kept"] += 1

    return filtered, stats


class ThreeWayClusterer:
    """
    Anchor-based 3-way clustering.

    Strategy:
    1. Load and encode all EDUs from the triplet
    2. For each CENTER EDU (anchor):
       - Find best matching LEFT EDU (above threshold)
       - Find best matching RIGHT EDU (above threshold)
       - If both found, create a 3-way cluster
    3. Use cross-encoder for validation (more accurate than cosine alone)
    """

    def __init__(
        self,
        sbert_model: SentenceTransformer,
        cross_encoder: CrossEncoder,
        cosine_threshold: float = 0.65,  # Initial filter with SBERT
        cross_encoder_threshold: float = 0.5,  # Validation threshold (lowered for more recall)
        allow_multiple_matches: bool = True,  # Allow EDU to appear in multiple clusters
    ):
        self.sbert = sbert_model
        self.cross_encoder = cross_encoder
        self.cosine_threshold = cosine_threshold
        self.cross_encoder_threshold = cross_encoder_threshold
        self.allow_multiple_matches = allow_multiple_matches

    def cluster_triplet(
        self, left_edus: List[dict], center_edus: List[dict], right_edus: List[dict]
    ) -> Tuple[Dict[int, List[str]], Dict[str, dict], dict]:
        """
        Perform anchor-based 3-way clustering.

        Returns:
            clusters: {cluster_id: [edu_id1, edu_id2, edu_id3]}
            edu_lookup: {edu_id: {text, bias}}
            stats: clustering statistics
        """
        stats = {
            "center_edus": len(center_edus),
            "left_edus": len(left_edus),
            "right_edus": len(right_edus),
            "clusters_formed": 0,
            "center_without_left_match": 0,
            "center_without_right_match": 0,
            "center_without_both": 0,
        }

        if not center_edus or not left_edus or not right_edus:
            return {}, {}, stats

        # Build edu lookup
        edu_lookup = {}
        for edu in left_edus + center_edus + right_edus:
            edu_lookup[edu["id"]] = {
                "text": edu["text"],
                "bias": edu["bias"],
            }

        # Encode all EDUs
        all_edus = left_edus + center_edus + right_edus
        texts = [e["text"] for e in all_edus]

        embeddings = self.sbert.encode(
            texts,
            batch_size=SBERT_ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=SBERT_NORMALIZE,
        )

        # Map embeddings back to EDUs
        emb_map = {}
        for edu, emb in zip(all_edus, embeddings):
            emb_map[edu["id"]] = emb

        # Separate embeddings by bias
        left_ids = [e["id"] for e in left_edus]
        center_ids = [e["id"] for e in center_edus]
        right_ids = [e["id"] for e in right_edus]

        left_embs = np.array([emb_map[eid] for eid in left_ids])
        center_embs = np.array([emb_map[eid] for eid in center_ids])
        right_embs = np.array([emb_map[eid] for eid in right_ids])

        # Compute cosine similarities: center vs left, center vs right
        # center_embs: (C, D), left_embs: (L, D), right_embs: (R, D)
        # sim_left: (C, L), sim_right: (C, R)
        sim_left = np.dot(center_embs, left_embs.T)
        sim_right = np.dot(center_embs, right_embs.T)

        # Track used EDUs (if not allowing multiple matches)
        used_left = set()
        used_right = set()

        clusters = {}
        cluster_id = 0

        # For each center EDU, find best matches
        for c_idx, c_id in enumerate(center_ids):
            c_text = edu_lookup[c_id]["text"]

            # Find candidate left matches above cosine threshold
            left_scores = sim_left[c_idx]
            left_candidates = [
                (left_ids[i], left_scores[i])
                for i in range(len(left_ids))
                if left_scores[i] >= self.cosine_threshold
                and (self.allow_multiple_matches or left_ids[i] not in used_left)
            ]
            left_candidates.sort(key=lambda x: -x[1])  # Sort by score desc

            # Find candidate right matches above cosine threshold
            right_scores = sim_right[c_idx]
            right_candidates = [
                (right_ids[i], right_scores[i])
                for i in range(len(right_ids))
                if right_scores[i] >= self.cosine_threshold
                and (self.allow_multiple_matches or right_ids[i] not in used_right)
            ]
            right_candidates.sort(key=lambda x: -x[1])  # Sort by score desc

            if not left_candidates:
                stats["center_without_left_match"] += 1
                if not right_candidates:
                    stats["center_without_both"] += 1
                continue

            if not right_candidates:
                stats["center_without_right_match"] += 1
                continue

            # Validate with cross-encoder
            best_left = None
            for l_id, _ in left_candidates:
                l_text = edu_lookup[l_id]["text"]
                score = self.cross_encoder.predict([(c_text, l_text)])[0]
                if score >= self.cross_encoder_threshold:
                    best_left = l_id
                    break

            if best_left is None:
                stats["center_without_left_match"] += 1
                continue

            best_right = None
            for r_id, _ in right_candidates:
                r_text = edu_lookup[r_id]["text"]
                score = self.cross_encoder.predict([(c_text, r_text)])[0]
                if score >= self.cross_encoder_threshold:
                    best_right = r_id
                    break

            if best_right is None:
                stats["center_without_right_match"] += 1
                continue

            # Found a 3-way cluster!
            clusters[cluster_id] = [best_left, c_id, best_right]
            cluster_id += 1
            stats["clusters_formed"] += 1

            if not self.allow_multiple_matches:
                used_left.add(best_left)
                used_right.add(best_right)

        return clusters, edu_lookup, stats


def process_triplet(
    triplet: dict,
    clusterer: ThreeWayClusterer,
) -> Tuple[Optional[Dict], Optional[Dict], dict]:
    """Process a single triplet and return 3-way clusters."""

    total_stats = {
        "edu_filter": {"input": 0, "kept": 0, "dropped": 0},
        "clustering": {},
    }

    # Load EDUs for each bias
    edus_by_bias = {}
    for bias_key in ["left", "center", "right"]:
        if bias_key not in triplet or not triplet[bias_key]:
            return None, None, total_stats

        edus, filter_stats = prepare_edus_for_article(triplet[bias_key], bias_key)
        edus_by_bias[bias_key] = edus

        total_stats["edu_filter"]["input"] += filter_stats["input"]
        total_stats["edu_filter"]["kept"] += filter_stats["kept"]
        total_stats["edu_filter"]["dropped"] += filter_stats["dropped"]

    # Need EDUs from all 3 sides
    if not all(edus_by_bias.get(b) for b in ["left", "center", "right"]):
        return None, None, total_stats

    # Run 3-way clustering
    clusters, edu_lookup, cluster_stats = clusterer.cluster_triplet(
        edus_by_bias["left"],
        edus_by_bias["center"],
        edus_by_bias["right"],
    )

    total_stats["clustering"] = cluster_stats

    if not clusters:
        return None, None, total_stats

    return clusters, edu_lookup, total_stats


def print_progress(
    current: int, total: int, elapsed: float, successes: int, empties: int, errors: int
):
    if current > 0:
        eta_seconds = (elapsed / current) * (total - current)
        eta_min, eta_sec = divmod(int(eta_seconds), 60)
        eta_str = f"ETA: {eta_min}m {eta_sec}s"
    else:
        eta_str = "ETA: --"

    print(
        f"\r{current}/{total} - {eta_str} - "
        f"{successes} with clusters, {empties} empty, {errors} errors",
        end="",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run improved 3-way clustering on triplets (anchor-based approach)"
    )
    parser.add_argument(
        "--triplets",
        default=os.path.join(DATA_DIR, "valid_triplets.json"),
        help="Input triplet JSON path",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(DEFAULT_OUT_DIR, f"3way_clusters_{RUN_STAMP}.json"),
        help="Output cluster JSON path",
    )
    parser.add_argument(
        "--meta",
        default=os.path.join(DEFAULT_OUT_DIR, f"3way_clusters_{RUN_STAMP}_meta.json"),
        help="Output metadata JSON path",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.65,
        help="Cosine similarity threshold for initial candidate selection (default: 0.65)",
    )
    parser.add_argument(
        "--cross-encoder-threshold",
        type=float,
        default=0.5,
        help="Cross-encoder threshold for validation (default: 0.5, lower than original 0.7 for more recall)",
    )
    parser.add_argument(
        "--allow-multiple-matches",
        action="store_true",
        default=True,
        help="Allow an EDU to appear in multiple clusters",
    )
    parser.add_argument(
        "--no-multiple-matches",
        action="store_true",
        help="Prevent an EDU from appearing in multiple clusters",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output files if they exist"
    )

    args = parser.parse_args()

    # Handle multiple matches flag
    allow_multiple = args.allow_multiple_matches and not args.no_multiple_matches

    # Create output directory
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta) or ".", exist_ok=True)

    # Check output files
    for out_path in [args.out, args.meta]:
        if os.path.exists(out_path) and not args.overwrite:
            print(f"ERROR: Output file exists: {out_path}")
            print("Use --overwrite to overwrite existing files")
            sys.exit(1)

    print("=" * 60)
    print("Improved 3-Way Clustering (Anchor-Based)")
    print("=" * 60)
    print(f"Triplets: {args.triplets}")
    print(f"Output: {args.out}")
    print(f"Cosine threshold: {args.cosine_threshold}")
    print(f"Cross-encoder threshold: {args.cross_encoder_threshold}")
    print(f"Allow multiple matches: {allow_multiple}")
    print("=" * 60)

    # Load models
    print("\nLoading models...")
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    # Create clusterer
    clusterer = ThreeWayClusterer(
        sbert_model=sbert_model,
        cross_encoder=cross_encoder,
        cosine_threshold=args.cosine_threshold,
        cross_encoder_threshold=args.cross_encoder_threshold,
        allow_multiple_matches=allow_multiple,
    )

    # Load triplets
    print(f"\nLoading triplets from {args.triplets}...")
    triplets = load_triplets(args.triplets)
    total = len(triplets)
    print(f"Loaded {total} triplets")

    # Process triplets
    print("\nProcessing triplets...")
    all_results = []
    successes = 0
    empties = 0
    errors = 0
    start_time = time.time()

    # Aggregate stats
    total_edu_input = 0
    total_edu_kept = 0
    total_edu_dropped = 0
    total_clusters = 0
    cluster_stats_agg = defaultdict(int)

    for i, triplet in enumerate(triplets):
        elapsed = time.time() - start_time
        print_progress(i, total, elapsed, successes, empties, errors)

        try:
            clusters, edu_lookup, stats = process_triplet(triplet, clusterer)

            # Aggregate filter stats
            total_edu_input += stats["edu_filter"]["input"]
            total_edu_kept += stats["edu_filter"]["kept"]
            total_edu_dropped += stats["edu_filter"]["dropped"]

            # Aggregate clustering stats
            for k, v in stats.get("clustering", {}).items():
                cluster_stats_agg[k] += v

            if clusters and len(clusters) > 0:
                all_results.append(
                    {
                        "triplet_idx": i,
                        "triplet": triplet,
                        "clusters": clusters,
                        "edu_lookup": edu_lookup,
                    }
                )
                successes += 1
                total_clusters += len(clusters)
            else:
                empties += 1

        except Exception as e:
            errors += 1
            print(f"\n[ERROR] Triplet {i}: {e}")

    # Final progress
    elapsed = time.time() - start_time
    print_progress(total, total, elapsed, successes, empties, errors)
    print()  # Newline

    # Save results
    print(f"\nSaving results to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Build metadata
    meta = {
        "script": "run_3way_clustering.py",
        "approach": "anchor-based 3-way clustering",
        "description": (
            "Uses center EDUs as anchors. For each center EDU, finds best matching "
            "left and right EDUs above threshold. Only keeps clusters where all 3 "
            "biases are present. Guarantees 100% 3-way cluster coverage."
        ),
        "params": {
            "cosine_threshold": args.cosine_threshold,
            "cross_encoder_threshold": args.cross_encoder_threshold,
            "allow_multiple_matches": allow_multiple,
            "sbert_model": SBERT_MODEL_NAME,
            "cross_encoder_model": CROSS_ENCODER_NAME,
        },
        "input": {
            "triplets_file": args.triplets,
            "total_triplets": total,
        },
        "output": {
            "clusters_file": args.out,
            "triplets_with_clusters": successes,
            "triplets_empty": empties,
            "triplets_error": errors,
            "total_clusters": total_clusters,
        },
        "edu_filter_stats": {
            "input_edus": total_edu_input,
            "kept_edus": total_edu_kept,
            "dropped_edus": total_edu_dropped,
            "drop_ratio": total_edu_dropped / total_edu_input if total_edu_input else 0,
        },
        "clustering_stats": dict(cluster_stats_agg),
        "coverage": {
            "3way_clusters": total_clusters,
            "3way_ratio": 1.0,  # By design, all clusters are 3-way
            "note": "All clusters guaranteed to have all 3 biases (left, center, right)",
        },
        "elapsed_seconds": round(elapsed, 3),
    }

    print(f"Saving metadata to {args.meta}...")
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Triplets processed: {total}")
    print(f"Triplets with 3-way clusters: {successes} ({100 * successes / total:.1f}%)")
    print(f"Triplets without clusters: {empties}")
    print(f"Errors: {errors}")
    print(f"Total 3-way clusters: {total_clusters}")
    print(f"3-way cluster ratio: 100.0% (by design)")
    print(f"Elapsed time: {elapsed:.1f}s")
    print("=" * 60)
    print(f"\nOutput: {args.out}")
    print(f"Metadata: {args.meta}")


if __name__ == "__main__":
    main()
