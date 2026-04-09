# Omission-Based Bias Classification (Option A)

This folder contains the implementation of **Option A**: a pure coverage/omission-based model that tests whether fact omission patterns alone can predict media bias.

## Hypothesis

> Fact omission patterns are stronger predictors of media bias than rhetorical prominence (RST structural placement).

## Key Finding

**The coverage-only model achieves identical performance to `without_both`:**

| Model | Test Accuracy | Test Macro-F1 |
|-------|---------------|---------------|
| Structural baseline (α=0.8, γ=0.5) | 67.05% | 67.02% |
| Structural without_s | 70.17% | 70.08% |
| Structural without_d | 65.91% | 65.91% |
| Structural without_both | 75.28% | 75.04% |
| **Coverage-only (delta mode)** | **75.28%** | **75.04%** |

This confirms that:
1. **The `without_both` model was learning purely from omission patterns**
2. **RST structural information (depth, nuclearity) provides no additional predictive value** when omission is available
3. **Fact coverage asymmetry is the dominant signal for bias classification**

## Background

The structural ablation experiments showed:
- `without_both` (α=1, γ=1) achieved **75.28% test accuracy** — the highest among all conditions
- This suggests the model learns primarily from omission patterns (when a fact is missing → score = 0)
- When omission was controlled away (size-3 clusters), `without_both` collapsed to **50% (chance)**

## Implementation

### Feature Encoding Modes

The script `train_coverage_svm.py` supports three feature modes:

#### 1. `delta` (default) - **RECOMMENDED**
Per cluster, encode `[has_left - has_center]` for left example, `[has_right - has_center]` for right example
- 1 feature per cluster per example
- Mirrors the DFI delta structure but with binary presence instead of W(d,s)
- **Result: 75.28% test accuracy** ✓

#### 2. `binary3` - **NOT USEFUL**
Per cluster, encode `[has_left, has_center, has_right]`
- 3 features per cluster
- **Result: 50% (chance)** - Both examples get identical features, model cannot distinguish
- This mode doesn't work because left and right examples share the same coverage vector

#### 3. `count` 
Per cluster, encode `[count_left, count_center, count_right]`
- 3 features per cluster
- EDU counts instead of binary presence
- Similar issue to binary3

### Running the Script

```bash
# From repository root
cd /path/to/rhetorical-framing-auditor

# Default run (delta features) - RECOMMENDED
python omission-based/train_coverage_svm.py

# With custom SVM hyperparameters
python omission-based/train_coverage_svm.py --svm-kernel linear --svm-c 1.0
```

### Output Files

- `results/coverage_svm_results.json`: Full metrics and comparison (delta mode)
- `results/coverage_svm_model.pkl`: Trained model with metadata

## Interpretation

### Why Delta Mode Works

The delta encoding `[has_X - has_center]` captures the **asymmetric omission pattern**:
- If left omits a fact that center covers: delta = 0 - 1 = -1
- If left covers a fact that center omits: delta = 1 - 0 = +1
- If both cover or both omit: delta = 0

This is mathematically equivalent to what `without_both` (α=1, γ=1) computes when using W=1 for present facts and W=0 for absent facts.

### Why Binary3 Mode Fails

With `[has_left, has_center, has_right]`:
- Left example: features = [1, 1, 0] (for a cluster where left and center cover but right omits)
- Right example: features = [1, 1, 0] (identical!)

Both examples get the same features — only the labels differ. The model has no discriminative information.

### Implications for the Project

1. **The original RST-based hypothesis is not supported**: Structural placement (depth, nuclearity) does not predict bias when coverage information is available.

2. **Coverage/omission IS predictive**: The fact that certain topics are covered or omitted differently across bias orientations is a strong signal.

3. **This is still a valid contribution**: Showing that "what facts are covered" matters more than "how facts are rhetorically positioned" is an interesting finding about media bias.

## Logs

Run logs are stored in `logs/omission-based/`.
