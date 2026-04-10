# Option E: Alternative Models

This folder contains experiments comparing different model families for bias classification.

## Motivation

SVM (RBF kernel) may not be optimal for this task. Different model families have different inductive biases:
- **Gradient boosting** (XGBoost/LightGBM): Better for sparse, heterogeneous features
- **Random Forest**: Ensemble method with built-in feature importance
- **Logistic Regression**: Interpretable, works well with regularization
- **MLP**: Can learn non-linear feature interactions

## Experiments

### 1. Model Comparison (Default Hyperparameters)
Test all models on three feature sets:
- Coverage-only
- Structural (log depth)
- Combined (coverage + structural)

Models tested:
- SVM (RBF kernel) - baseline
- Logistic Regression (L2)
- Logistic Regression (L1)
- Random Forest
- MLP Neural Network
- XGBoost (if installed)
- LightGBM (if installed)

### 2. Hyperparameter Tuning
Grid search with 5-fold CV for:
- Logistic Regression: C ∈ {0.01, 0.1, 1.0, 10.0}
- Random Forest: n_estimators, max_depth, min_samples_split
- XGBoost: n_estimators, max_depth, learning_rate

### 3. Feature Importance Analysis
Extract Random Forest feature importances to identify which clusters/features are most predictive.

## Usage

```bash
# Run with default settings
python alternative-models/train_alternative_models.py

# Customize settings
python alternative-models/train_alternative_models.py \
    --seed 123 \
    --cv-folds 10
```

## Installing Optional Dependencies

For gradient boosting models:
```bash
pip install xgboost lightgbm
```

## Key Outputs

- `results/alternative_models_results.json`: All experiment results
- Model comparison across feature sets
- Hyperparameter tuning results
- Feature importance rankings

## Interpretation Guide

1. **If tree-based models beat SVM**: Features may have complex interactions
2. **If logistic regression matches SVM**: Task is essentially linear
3. **If L1 outperforms L2**: Many features are irrelevant (sparse solution better)
4. **Feature importance**: Shows which cluster positions matter most
