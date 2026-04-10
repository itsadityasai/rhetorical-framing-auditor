# Option D: Experimental Design Improvements

This folder contains experiments to improve the robustness of our evaluation methodology.

## Motivation

The original fixed train/val/test split may have high variance due to small test set size (176 triplets = 352 samples). This could lead to unreliable accuracy estimates. Option D addresses this through:

1. **K-Fold Cross-Validation**: Reduces variance from a single split
2. **Bootstrap Confidence Intervals**: Quantifies uncertainty in accuracy estimates
3. **Paired Bootstrap Test**: Tests if combined features significantly outperform coverage
4. **Sample Size Analysis**: Learning curves to assess data sufficiency

## Experiments

### 1. K-Fold Cross-Validation
- Stratified 5-fold CV instead of fixed 80/10/10 split
- Reports mean ± std accuracy across folds
- Reduces variance from lucky/unlucky test sets

### 2. Bootstrap Confidence Intervals
- Train model on 80% of data
- Resample test predictions 1000 times with replacement
- Compute 95% percentile confidence intervals
- Report CI width as uncertainty measure

### 3. Paired Bootstrap Test
- Tests H0: combined accuracy ≤ coverage accuracy
- Paired resampling preserves sample correlations
- Reports one-sided p-value
- Significant if 95% CI for difference excludes 0

### 4. Sample Size Analysis
- Learning curves: accuracy vs training set size
- Helps assess if we have enough data
- Informs whether collecting more data would help

## Usage

```bash
# Run with default settings (5-fold CV, 1000 bootstrap)
python experimental-design/train_crossval.py

# Customize settings
python experimental-design/train_crossval.py \
    --k-folds 10 \
    --n-bootstrap 2000 \
    --seed 123
```

## Key Outputs

- `results/crossval_results.json`: All experiment results
- Cross-validation mean ± std for each feature set
- 95% confidence intervals for accuracy
- Statistical significance test for combined vs coverage
- Learning curves for sample size analysis

## Interpretation Guide

1. **If CV std is high** (>0.03): Results may not be stable; need more data
2. **If CI is wide** (>0.05): Uncertainty is high; report CI not point estimate
3. **If paired test is significant**: Combined features provide genuine improvement
4. **If learning curve is flat**: More training data likely won't help
5. **If learning curve is rising**: Could benefit from more data
