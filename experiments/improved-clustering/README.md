# Improved 3-Way Clustering

This folder contains an improved clustering approach that guarantees **100% 3-way cluster coverage** (all clusters have left, center, and right EDUs).

## Problem with Current Approach

The original clustering (`run_fact_clustering.py`) produces only ~18% 3-way clusters:
- Left + Center only: 43.6%
- Center + Right only: 37.6%
- All 3 sides: 18.8%
- Left + Right only: 0% (by design, requires center for DFI)

This happens because agglomerative clustering finds semantic similarity, but left and right sources often cover different aspects of a story.

## Solution: Anchor-Based 3-Way Clustering

Instead of blindly clustering all EDUs, we use **center EDUs as anchors**:

1. For each CENTER EDU:
   - Find best matching LEFT EDU (above threshold)
   - Find best matching RIGHT EDU (above threshold)
   - Only create cluster if BOTH matches found
2. Result: Every cluster has exactly one EDU from each bias

### Trade-offs

- **Fewer total clusters** (strict matching requirement)
- **100% 3-way coverage** (all clusters usable for comparative analysis)
- **No omission signal** (can't detect when one side omits a fact)

## Usage

### Step 1: Run 3-Way Clustering

```bash
python run_3way_clustering.py \
    --triplets ../data/valid_triplets.json \
    --out output/3way_clusters.json \
    --meta output/3way_clusters_meta.json \
    --cosine-threshold 0.65 \
    --cross-encoder-threshold 0.5
```

Parameters:
- `--cosine-threshold`: Initial filter for candidates (default: 0.65)
- `--cross-encoder-threshold`: Validation threshold (default: 0.5, lower than original 0.7 for more recall)
- `--allow-multiple-matches`: Allow EDU in multiple clusters (default: true)
- `--no-multiple-matches`: Prevent EDU reuse across clusters

### Step 2: Build Facts with RST Features

```bash
python build_facts_3way.py \
    --clusters output/3way_clusters.json \
    --out output/3way_facts.json
```

### Step 3: Use with Existing Pipeline

The output `3way_facts.json` is compatible with the existing DFI pipeline. You can:

1. Generate train/val/test splits
2. Run DFI-based classification
3. Run structural ablations

## Output Files

- `output/3way_clusters_{timestamp}.json` - Raw clusters
- `output/3way_clusters_{timestamp}_meta.json` - Metadata and statistics
- `output/3way_facts.json` - Facts enriched with RST features

## Comparison with Original

| Metric | Original | 3-Way |
|--------|----------|-------|
| 3-way cluster ratio | ~18% | 100% |
| Total clusters | ~16,000 | TBD |
| Can detect omission | Yes | No |
| RST framing analysis | Yes | Yes |

## Notes

- This approach is useful for **pure RST framing analysis** where you want to compare how all 3 biases position the same fact
- If you need to detect **omission bias** (what facts are missing), use the original clustering
- Lower thresholds produce more clusters but potentially lower quality matches
