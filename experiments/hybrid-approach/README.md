# Option C: Hybrid Approach

**Goal**: Explicitly separate coverage and structural signals to quantify their independent contributions to bias classification.

## Key Findings

### 1. Feature Group Ablation Results

| Experiment | Test Accuracy | Test F1 | Interpretation |
|------------|---------------|---------|----------------|
| Coverage-only | 75.28% | 75.04% | Binary omission patterns |
| Structural (all clusters) | 76.42% | 76.41% | Log-depth prominence (best from Option B) |
| Structural (size-3 only) | 51.42% | 48.70% | Pure structure, no coverage signal |
| **Combined** | **77.27%** | **77.27%** | Coverage + structural interleaved |

**Key Insight**: Combined features achieve the best performance (+1.99% over coverage-only), suggesting structural information provides complementary signal beyond coverage.

### 2. Two-Stage Classifier Analysis

- Coverage model test errors: 87/352 (24.7%)
- Structure could correct: 21/87 (24.1%)
- Structure also wrong: 66/87 (75.9%)

**Combination Strategies**:
| Strategy | Test Accuracy | Improvement |
|----------|---------------|-------------|
| Uncertainty threshold 0.6 | 75.85% | +0.57% |
| Weighted (cov=0.8, str=0.2) | 76.42% | +1.14% |
| Weighted (cov=0.7, str=0.3) | 76.42% | +1.14% |

**Key Insight**: Structural features can correct ~24% of coverage errors, but the majority of errors are shared (both models wrong). Optimal weighting is 70-80% coverage, 20-30% structure.

### 3. Stacking Ensemble

| Model | Test Accuracy | Test F1 |
|-------|---------------|---------|
| Coverage base | 75.28% | 75.04% |
| Structural base | 76.42% | 76.41% |
| **Stacked meta-model** | **76.99%** | - |

**Meta-model Coefficients** (Logistic Regression):
- Coverage prob[0]: -1.21, prob[1]: +1.25
- Structural prob[0]: -1.41, prob[1]: +1.45

**Key Insight**: Structural features have slightly higher magnitude coefficients, suggesting they provide marginally more discriminative power. The stacking ensemble improves +1.71% over coverage and +0.57% over structural alone.

## Conclusions

1. **Combined is best**: The simple feature concatenation (combined) achieves 77.27% - the highest accuracy observed across all experiments.

2. **Structural adds value**: When coverage and structure are combined, performance exceeds either alone, confirming they capture complementary aspects of bias.

3. **Coverage remains dominant**: ~76% of errors are shared between models, and optimal combination weights favor coverage (70-80%).

4. **Size-3 structural fails**: When coverage signal is removed entirely (size-3 clusters only), accuracy drops to near-chance (51.4%), confirming structural placement alone cannot predict bias.

5. **Practical recommendation**: Use combined features (coverage + log-depth structural) for best performance.

## Files

- `train_hybrid_svm.py` - Main experiment script
- `results/hybrid_results.json` - Full results with all metrics
- `results/models/` - Saved model artifacts

## Usage

```bash
python experiments/hybrid-approach/train_hybrid_svm.py

# With custom parameters
python experiments/hybrid-approach/train_hybrid_svm.py \
    --facts data/valid_facts_results_recluster_gpu.json \
    --split-dir data/valid_dfi_splits_recluster_gpu \
    --svm-c 10.0 --svm-gamma 0.1
```

## Comparison to Previous Results

| Experiment | Test Accuracy | Notes |
|------------|---------------|-------|
| Option A: Coverage-only | 75.28% | Baseline |
| Option B: Log-depth structural | 76.42% | Best structural formula |
| Option B: Aggregate vector | 61.36% | Negative result |
| **Option C: Combined** | **77.27%** | **Best overall** |
