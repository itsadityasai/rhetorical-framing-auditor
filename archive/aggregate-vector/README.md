# Aggregate Vector Experiment

## Goal

Address the fundamental dimension mismatch issue in the original DFI approach: DFI vectors from different triplets have different facts/clusters, so the 1st entry in one DFI doesn't relate to the 1st entry in another DFI.

## Hypothesis

By extracting fixed-length aggregate statistics from each DFI, we can create semantically consistent feature vectors that are comparable across all triplets.

## Approach

1. **Normalize each DFI vector** (subtract mean, divide by std) to have 0 mean and unit std
2. **Extract aggregate statistical features** that are consistent across all triplets:
   - Mean, Std, Min, Max of DFI deltas
   - Skewness (asymmetry of distribution)
   - % positive deltas (side has higher prominence than center)
   - % zero deltas (coverage overlap)
   - % negative deltas (center has higher prominence)
   - Number of clusters (log-scaled)

3. **Train SVM on these fixed-length vectors** instead of variable-length padded DFI

## Results

| Experiment | Input Dim | Test Acc | vs Coverage-Padded | vs LogDepth-Padded |
|------------|-----------|----------|--------------------|--------------------|
| agg_coverage | 9 | 55.68% | -19.60% | -20.74% |
| agg_coverage_norm | 9 | 57.39% | -17.89% | -19.03% |
| agg_log_depth | 9 | **61.36%** | -13.92% | -15.06% |
| agg_log_depth_norm | 9 | 56.82% | -18.46% | -19.60% |
| agg_combined | 18 | 60.80% | -14.48% | -15.62% |
| agg_extended | 14 | 57.67% | -17.61% | -18.75% |

**Reference baselines (padded DFI approach):**
- Coverage-only padded: 75.28%
- Log-depth padded: 76.42%

## Key Findings

1. **Aggregate features perform significantly worse** than padded DFI vectors (~55-61% vs 75-76%)

2. **The "dimension mismatch" may not be the real problem** - the padded approach works despite theoretically mixing unrelated dimensions

3. **Possible explanations:**
   - The raw DFI values encode **cluster-specific patterns** that are lost when aggregating
   - Specific fact omission patterns (which facts are missing) matter more than aggregate statistics
   - The SVM + padding approach may be learning **sparse, position-specific** signals that aggregation destroys

4. **Log-depth features slightly outperform coverage** in the aggregate setting (61.36% vs 55.68%), reversing the pattern from padded DFI

## Interpretation

This negative result is informative: it suggests that **what matters is not summary statistics of DFI vectors, but the specific pattern of which facts are omitted/emphasized**.

The padded approach, despite its theoretical issues, preserves cluster-level identity through consistent ordering (same triplet → same cluster order). The padding zeros then encode "absence" of later clusters, which is itself informative.

## Files

- `train_aggregate_svm.py` - Training script with 6 experiments
- `results/aggregate_svm_results.json` - Full results
- `results/models/*.pkl` - Trained models

## Usage

```bash
python aggregate-vector/train_aggregate_svm.py
```
