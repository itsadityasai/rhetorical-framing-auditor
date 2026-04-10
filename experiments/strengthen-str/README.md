# Strengthen Structural Signals (Option B)

This folder contains the implementation of **Option B**: exploring whether alternative structural features can improve bias classification beyond the coverage-only baseline.

## Hypothesis

> If the original prominence formula W(d,s) = α^(d+1) * γ^s fails to capture structural signals, perhaps alternative encodings can do better.

## Key Finding

**The logarithmic depth formula achieves the best performance:**

| Model | Test Accuracy | Δ vs Coverage | Δ vs Structural |
|-------|---------------|---------------|-----------------|
| **W = 1/(1+ln(1+d))** | **76.42%** | **+1.14%** | **+9.37%** |
| Coverage-only | 75.28% | baseline | +8.23% |
| W = 1/(1+0.5s) | 75.00% | -0.28% | +7.95% |
| Nuclearity dist. | 75.00% | -0.28% | +7.95% |
| Original structural | 67.05% | -8.23% | baseline |
| Aggregate (no coverage) | 53.12% | -22.16% | -13.93% |

## Approaches Explored

### Approach 1: Aggregate Structural Features

Instead of max-prominence per side, aggregate multiple statistics:
- Mean/variance of depth
- Nuclearity ratio (% nucleus vs satellite)
- Satellite edge statistics

**Results:**
- With coverage: 68.47% (coverage signal helps)
- Without coverage (size-3 only): 53.12% (near chance - structural features alone fail)

### Approach 2: Nuclearity Distribution Features

Focus on nucleus/satellite role distribution per cluster:
- Binary coverage + nucleus ratio per side

**Results:**
- 75.00% test accuracy - matches coverage-only

### Approach 3: Alternative Prominence Formulas

| Formula | Description | Test Acc |
|---------|-------------|----------|
| `exp(-0.3*d)` | Exponential depth decay | 67.05% |
| `1/(1+0.5*s)` | Inverse satellite penalty | 75.00% |
| `exp(-0.3*d)/(1+0.5*s)` | Combined | 65.06% |
| `1/(1+ln(1+d))` | **Logarithmic depth** | **76.42%** |
| `α^(d+1) * γ^s` | Original formula | 67.05% |

## Interpretation

1. **Logarithmic depth compression helps**: The only formula that beats coverage-only (+1.14%) uses log depth, suggesting that compressing depth differences captures useful signal.

2. **Inverse satellite ≈ Coverage**: The 1/(1+s) formula essentially encodes coverage-correlated information (satellite count is 0 when side is absent).

3. **Combining depth and satellite hurts**: The combined formula performs *worse* than either component alone, indicating destructive interaction.

4. **Pure structural features fail**: When coverage is removed (aggregate no coverage), accuracy drops to 53% (near chance).

## Running the Script

```bash
# From repository root
python experiments/strengthen-str/train_strengthen_structural.py

# With custom output paths
python experiments/strengthen-str/train_strengthen_structural.py \
    --out experiments/strengthen-str/results/custom_results.json \
    --model-dir experiments/strengthen-str/results/custom_models
```

## Output Files

- `results/strengthen_structural_results.json`: Full metrics for all experiments
- `results/models/*.pkl`: Trained models for each experiment

## Conclusion

The logarithmic depth formula provides marginal improvement (+1.14%) over coverage-only, but the dominant signal remains fact coverage/omission patterns. Pure structural features without coverage information cannot predict bias.
