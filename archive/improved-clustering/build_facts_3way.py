"""
Build facts from 3-way clusters (enriches with RST features).

This script takes the output from run_3way_clustering.py and adds RST features
(depth, role, satellite_edges_to_root) to create the final facts file that
can be used with the existing DFI pipeline.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# Paths
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

DATA_DIR = os.path.join(PROJECT_ROOT, params["paths"]["dirs"]["data"])
RST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, params["paths"]["dirs"]["rst_output"])


def roles(nuclearity):
    """Get roles from nuclearity string."""
    if nuclearity == "NS":
        return "N", "S"
    if nuclearity == "SN":
        return "S", "N"
    if nuclearity == "NN":
        return "N", "N"
    return None, None


def satellite_counts(edus, relations):
    """Compute satellite edge count and local role for each EDU."""
    parent_of = {}
    role_of = {}

    for rel in relations:
        parent = rel.get("parent")
        left = rel.get("left")
        right = rel.get("right")
        left_role, right_role = roles(rel.get("nuclearity"))

        if left is not None:
            parent_of[left] = parent
            role_of[left] = left_role
        if right is not None:
            parent_of[right] = parent
            role_of[right] = right_role

    sat_edges = {}
    local_role = {}

    for edu in edus:
        edu_id = edu["id"]
        local_role[edu_id] = role_of.get(edu_id)

        count = 0
        cur = edu_id
        seen = set()
        while cur in parent_of and cur not in seen:
            seen.add(cur)
            if role_of.get(cur) == "S":
                count += 1
            cur = parent_of[cur]

        sat_edges[edu_id] = count

    return sat_edges, local_role


def article_id_from_path(path):
    """Extract article ID from file path."""
    return os.path.basename(path).replace(".json", "")


def load_rst_lookup_for_triplet(triplet):
    """Load RST features for all articles in a triplet."""
    lookup = {}

    for bias_key in ["left", "center", "right"]:
        triplet_path = triplet.get(bias_key)
        if not triplet_path:
            continue

        article_id = article_id_from_path(triplet_path)
        rst_path = os.path.join(RST_OUTPUT_DIR, f"{article_id}.json")

        if not os.path.exists(rst_path):
            continue

        with open(rst_path, "r", encoding="utf-8") as f:
            rst = json.load(f)

        edus = rst.get("edus", [])
        relations = rst.get("relations", [])
        sat_edges, local_role = satellite_counts(edus, relations)

        for edu in edus:
            full_edu_id = f"{article_id}_{edu['id']}"
            lookup[full_edu_id] = {
                "text": edu["text"],
                "bias": bias_key,
                "depth": edu.get("depth"),
                "role": local_role.get(edu["id"]),
                "satellite_edges_to_root": sat_edges.get(edu["id"], 0),
            }

    return lookup


def build_facts_from_clusters(cluster_result):
    """Build enriched facts from a cluster result."""
    triplet = cluster_result["triplet"]
    clusters = cluster_result["clusters"]
    edu_lookup = cluster_result["edu_lookup"]

    # Load RST features
    rst_lookup = load_rst_lookup_for_triplet(triplet)

    # Enrich edu_lookup with RST features
    enriched_lookup = {}
    for edu_id, edu_info in edu_lookup.items():
        rst_info = rst_lookup.get(edu_id, {})
        enriched_lookup[edu_id] = {
            "text": rst_info.get("text", edu_info.get("text")),
            "bias": rst_info.get("bias", edu_info.get("bias")),
            "depth": rst_info.get("depth", 0),
            "role": rst_info.get("role"),
            "satellite_edges_to_root": rst_info.get("satellite_edges_to_root", 0),
        }

    # Build facts list
    facts = []
    for cluster_id, edu_ids in clusters.items():
        fact_edus = [
            enriched_lookup[edu_id] for edu_id in edu_ids if edu_id in enriched_lookup
        ]
        facts.append(
            {
                "cluster_id": cluster_id,
                "edus": fact_edus,
            }
        )

    return {
        "triplet_idx": cluster_result.get("triplet_idx"),
        "triplet": triplet,
        "clusters": clusters,
        "edu_lookup": {
            edu_id: enriched_lookup[edu_id]
            for edu_ids in clusters.values()
            for edu_id in edu_ids
            if edu_id in enriched_lookup
        },
        "facts": facts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build facts from 3-way clusters (adds RST features)"
    )
    parser.add_argument(
        "--clusters",
        required=True,
        help="Input clusters JSON from run_3way_clustering.py",
    )
    parser.add_argument("--out", required=True, help="Output facts JSON path")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if exists"
    )

    args = parser.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        print(f"ERROR: Output exists: {args.out}")
        print("Use --overwrite to overwrite")
        sys.exit(1)

    # Load clusters
    print(f"Loading clusters from {args.clusters}...")
    with open(args.clusters, "r", encoding="utf-8") as f:
        cluster_results = json.load(f)

    print(f"Loaded {len(cluster_results)} triplet results")

    # Build facts
    print("Building facts with RST features...")
    facts_results = []
    for i, cluster_result in enumerate(cluster_results):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(cluster_results)}")
        facts_results.append(build_facts_from_clusters(cluster_result))

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Saving facts to {args.out}...")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(facts_results, f, indent=2)

    # Stats
    total_clusters = sum(len(r.get("clusters", {})) for r in facts_results)
    total_facts = sum(len(r.get("facts", [])) for r in facts_results)

    print("\nSummary:")
    print(f"  Triplets: {len(facts_results)}")
    print(f"  Total clusters: {total_clusters}")
    print(f"  Total facts: {total_facts}")
    print(f"  Output: {args.out}")


if __name__ == "__main__":
    main()
