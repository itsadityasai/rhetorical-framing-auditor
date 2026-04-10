"""
Alternative Models (Option E): Compare different classifiers beyond SVM.

This script implements Option E to explore whether different model families
perform better on the bias classification task.

Models tested:
1. XGBoost (Gradient Boosting): Better for sparse, heterogeneous features
2. LightGBM: Efficient gradient boosting with native categorical support
3. Random Forest: Ensemble method with feature importance analysis
4. Logistic Regression: Interpretable baseline with L1/L2 regularization
5. MLP Neural Network: Non-linear feature interactions

Each model is tested on three feature sets:
- Coverage-only
- Structural (log depth)
- Combined (coverage + structural)

Hyperparameter tuning via 5-fold cross-validation is performed for each model.
"""

import argparse
import json
import os
import sys
import math
from datetime import datetime
from typing import Dict, List, Tuple, Callable, Set
import copy
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Try to import gradient boosting libraries
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Skipping XGBoost experiments.")

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Skipping LightGBM experiments.")

# Add parent directory for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.run_logger import close_run_logging, init_run_logging, log_run_results


DEFAULT_FACTS_PATH = "data/valid_facts_results_recluster_gpu.json"
DEFAULT_SPLIT_DIR = "data/valid_dfi_splits_recluster_gpu"
DEFAULT_OUT_PATH = "alternative-models/results/alternative_models_results.json"
DEFAULT_MODEL_DIR = "alternative-models/results/models"

VALID_BIASES = {"left", "center", "right"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Alternative models comparison (Option E)"
    )
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="CV folds for hyperparameter tuning"
    )

    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_split_triplet_idx(split_rows: List[dict]) -> Set[int]:
    return {row["triplet_idx"] for row in split_rows if "triplet_idx" in row}


def split_facts_by_existing_splits(
    facts_rows: List[dict],
    train_ids: Set[int],
    val_ids: Set[int],
    test_ids: Set[int],
) -> Tuple[List[dict], List[dict], List[dict]]:
    train_rows, val_rows, test_rows = [], [], []
    for row in facts_rows:
        idx = row.get("triplet_idx")
        if idx in train_ids:
            train_rows.append(row)
        elif idx in val_ids:
            val_rows.append(row)
        elif idx in test_ids:
            test_rows.append(row)
    return train_rows, val_rows, test_rows


# =============================================================================
# FEATURE BUILDERS (same as previous experiments)
# =============================================================================


def get_cluster_coverage(edus: List[str], edu_lookup: Dict) -> Dict[str, int]:
    """Count EDUs per bias side in a cluster."""
    counts = {"left": 0, "center": 0, "right": 0}
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta is None:
            continue
        bias = meta.get("bias")
        if bias in VALID_BIASES:
            counts[bias] += 1
    return counts


def W_log_depth(depth: int, sat_count: int) -> float:
    """Logarithmic depth penalty (best from Option B)."""
    return 1.0 / (1.0 + math.log1p(depth))


def compute_side_prominence(
    edus: List[str], edu_lookup: Dict, side: str, W_func: Callable
) -> float:
    """Compute max prominence score for a side."""
    scores = []
    for eid in edus:
        meta = edu_lookup.get(eid)
        if meta and meta.get("bias") == side:
            depth = meta.get("depth", 0)
            sat_count = meta.get("satellite_edges_to_root", 0)
            scores.append(W_func(depth, sat_count))
    return max(scores) if scores else 0.0


def build_coverage_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build binary coverage features (delta encoding)."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        coverage = get_cluster_coverage(edus, edu_lookup)

        has_left = 1 if coverage["left"] > 0 else 0
        has_center = 1 if coverage["center"] > 0 else 0
        has_right = 1 if coverage["right"] > 0 else 0

        features_left.append(has_left - has_center)
        features_right.append(has_right - has_center)

    return features_left, features_right


def build_structural_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build structural prominence features using log depth formula."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    deltas_left = []
    deltas_right = []

    for cluster_id, edus in clusters.items():
        W_left = compute_side_prominence(edus, edu_lookup, "left", W_log_depth)
        W_center = compute_side_prominence(edus, edu_lookup, "center", W_log_depth)
        W_right = compute_side_prominence(edus, edu_lookup, "right", W_log_depth)

        deltas_left.append(W_left - W_center)
        deltas_right.append(W_right - W_center)

    return deltas_left, deltas_right


def build_combined_features(fact_row: Dict) -> Tuple[List[float], List[float]]:
    """Build combined features: coverage + structural (log depth)."""
    clusters = fact_row.get("clusters", {})
    edu_lookup = fact_row.get("edu_lookup", {})

    features_left = []
    features_right = []

    for cluster_id, edus in clusters.items():
        coverage = get_cluster_coverage(edus, edu_lookup)

        # Coverage features
        has_left = 1 if coverage["left"] > 0 else 0
        has_center = 1 if coverage["center"] > 0 else 0
        has_right = 1 if coverage["right"] > 0 else 0

        cov_left = has_left - has_center
        cov_right = has_right - has_center

        # Structural features
        W_left = compute_side_prominence(edus, edu_lookup, "left", W_log_depth)
        W_center = compute_side_prominence(edus, edu_lookup, "center", W_log_depth)
        W_right = compute_side_prominence(edus, edu_lookup, "right", W_log_depth)

        str_left = W_left - W_center
        str_right = W_right - W_center

        # Combined: [cov, str] per cluster
        features_left.extend([cov_left, str_left])
        features_right.extend([cov_right, str_right])

    return features_left, features_right


# =============================================================================
# DATA PREPARATION
# =============================================================================


def build_features_for_rows(rows: List[dict], feature_builder: Callable):
    """Apply feature builder to all rows."""
    for row in rows:
        row["feat_left"], row["feat_right"] = feature_builder(row)


def build_xy(rows: List[dict], key_left: str, key_right: str):
    """Build X, y arrays from feature rows."""
    x, y = [], []
    for row in rows:
        x.append(list(row[key_left]))
        y.append(0)  # left-vs-center
        x.append(list(row[key_right]))
        y.append(1)  # right-vs-center
    return x, np.array(y)


def pad_or_truncate(raw_x: List[List[float]], target_len: int) -> np.ndarray:
    """Pad shorter vectors with zeros, truncate longer ones."""
    arr = np.zeros((len(raw_x), target_len), dtype=float)
    for i, vec in enumerate(raw_x):
        lim = min(len(vec), target_len)
        if lim > 0:
            arr[i, :lim] = np.array(vec[:lim], dtype=float)
    return arr


def prepare_splits(
    train_rows: List[dict],
    val_rows: List[dict],
    test_rows: List[dict],
    feature_builder: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Prepare train/val/test splits with given feature builder."""
    train_copy = copy.deepcopy(train_rows)
    val_copy = copy.deepcopy(val_rows)
    test_copy = copy.deepcopy(test_rows)

    build_features_for_rows(train_copy, feature_builder)
    build_features_for_rows(val_copy, feature_builder)
    build_features_for_rows(test_copy, feature_builder)

    x_train_raw, y_train = build_xy(train_copy, "feat_left", "feat_right")
    x_val_raw, y_val = build_xy(val_copy, "feat_left", "feat_right")
    x_test_raw, y_test = build_xy(test_copy, "feat_left", "feat_right")

    max_len = max((len(v) for v in x_train_raw), default=0)

    X_train = pad_or_truncate(x_train_raw, max_len)
    X_val = pad_or_truncate(x_val_raw, max_len)
    X_test = pad_or_truncate(x_test_raw, max_len)

    return X_train, y_train, X_val, y_val, X_test, y_test, max_len


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test) -> Dict:
    """Train and evaluate a model, returning metrics."""
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    return {
        "train": {
            "accuracy": float(accuracy_score(y_train, train_pred)),
            "f1": float(f1_score(y_train, train_pred, average="macro")),
        },
        "val": {
            "accuracy": float(accuracy_score(y_val, val_pred)),
            "f1": float(f1_score(y_val, val_pred, average="macro")),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "f1": float(f1_score(y_test, test_pred, average="macro")),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        },
    }


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================


def get_logistic_regression(seed: int):
    """Logistic Regression with L2 regularization."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
            C=1.0,
        ),
    )


def get_logistic_regression_l1(seed: int):
    """Logistic Regression with L1 regularization (sparse)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=seed,
            solver="saga",
            penalty="l1",
            C=1.0,
        ),
    )


def get_random_forest(seed: int, n_estimators: int = 100):
    """Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def get_mlp(seed: int):
    """Multi-layer Perceptron classifier."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,  # L2 regularization
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    )


def get_svm_rbf(seed: int):
    """SVM with RBF kernel (baseline)."""
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10.0,
            gamma=0.1,
            random_state=seed,
        ),
    )


def get_xgboost(seed: int):
    """XGBoost classifier."""
    if not HAS_XGBOOST:
        return None
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="logloss",
    )


def get_lightgbm(seed: int):
    """LightGBM classifier."""
    if not HAS_LIGHTGBM:
        return None
    return lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================


def tune_logistic_regression(X_train, y_train, seed: int, cv_folds: int):
    """Tune logistic regression with grid search."""
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )

    param_grid = {
        "logisticregression__C": [0.01, 0.1, 1.0, 10.0],
        "logisticregression__penalty": ["l2"],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_random_forest(X_train, y_train, seed: int, cv_folds: int):
    """Tune random forest with grid search."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_xgboost(X_train, y_train, seed: int, cv_folds: int):
    """Tune XGBoost with grid search."""
    if not HAS_XGBOOST:
        return None, None, None

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.05, 0.1, 0.2],
    }

    xgb_clf = xgb.XGBClassifier(
        random_state=seed,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    grid = GridSearchCV(
        xgb_clf,
        param_grid,
        cv=cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# =============================================================================
# MAIN EXPERIMENTS
# =============================================================================


def run_model_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_name: str,
    seed: int,
    cv_folds: int,
) -> Dict:
    """Run all models on a given feature set."""
    print(f"\n  Feature set: {feature_name}")
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    results = {}

    # Combine train + val for final training
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    models = [
        ("svm_rbf", "SVM (RBF kernel)", get_svm_rbf(seed)),
        ("logistic_l2", "Logistic Regression (L2)", get_logistic_regression(seed)),
        ("logistic_l1", "Logistic Regression (L1)", get_logistic_regression_l1(seed)),
        ("random_forest", "Random Forest", get_random_forest(seed)),
        ("mlp", "MLP Neural Network", get_mlp(seed)),
    ]

    if HAS_XGBOOST:
        models.append(("xgboost", "XGBoost", get_xgboost(seed)))

    if HAS_LIGHTGBM:
        models.append(("lightgbm", "LightGBM", get_lightgbm(seed)))

    for model_key, model_name, model in models:
        if model is None:
            continue

        print(f"\n    {model_name}...")

        try:
            metrics = evaluate_model(
                model,
                X_trainval,
                y_trainval,  # Train on train+val
                X_val,
                y_val,  # Val metrics (for reference)
                X_test,
                y_test,  # Test metrics
            )

            results[model_key] = {
                "name": model_name,
                "metrics": metrics,
            }

            print(
                f"      Test acc: {metrics['test']['accuracy']:.4f}, F1: {metrics['test']['f1']:.4f}"
            )

        except Exception as e:
            print(f"      ERROR: {str(e)}")
            results[model_key] = {
                "name": model_name,
                "error": str(e),
            }

    return results


def run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_name: str,
    seed: int,
    cv_folds: int,
) -> Dict:
    """Run hyperparameter tuning for key models."""
    print(f"\n  Hyperparameter tuning for {feature_name}...")

    results = {}

    # Logistic Regression tuning
    print("    Tuning Logistic Regression...")
    try:
        best_lr, best_params_lr, best_score_lr = tune_logistic_regression(
            X_train, y_train, seed, cv_folds
        )
        test_pred_lr = best_lr.predict(X_test)
        test_acc_lr = accuracy_score(y_test, test_pred_lr)

        results["logistic_tuned"] = {
            "best_params": {
                k.replace("logisticregression__", ""): v
                for k, v in best_params_lr.items()
            },
            "cv_score": float(best_score_lr),
            "test_accuracy": float(test_acc_lr),
            "test_f1": float(f1_score(y_test, test_pred_lr, average="macro")),
        }
        print(
            f"      Best params: {best_params_lr}, CV: {best_score_lr:.4f}, Test: {test_acc_lr:.4f}"
        )
    except Exception as e:
        print(f"      ERROR: {str(e)}")
        results["logistic_tuned"] = {"error": str(e)}

    # Random Forest tuning
    print("    Tuning Random Forest...")
    try:
        best_rf, best_params_rf, best_score_rf = tune_random_forest(
            X_train, y_train, seed, cv_folds
        )
        test_pred_rf = best_rf.predict(X_test)
        test_acc_rf = accuracy_score(y_test, test_pred_rf)

        results["random_forest_tuned"] = {
            "best_params": best_params_rf,
            "cv_score": float(best_score_rf),
            "test_accuracy": float(test_acc_rf),
            "test_f1": float(f1_score(y_test, test_pred_rf, average="macro")),
        }
        print(
            f"      Best params: {best_params_rf}, CV: {best_score_rf:.4f}, Test: {test_acc_rf:.4f}"
        )
    except Exception as e:
        print(f"      ERROR: {str(e)}")
        results["random_forest_tuned"] = {"error": str(e)}

    # XGBoost tuning
    if HAS_XGBOOST:
        print("    Tuning XGBoost...")
        try:
            best_xgb, best_params_xgb, best_score_xgb = tune_xgboost(
                X_train, y_train, seed, cv_folds
            )
            if best_xgb is not None:
                test_pred_xgb = best_xgb.predict(X_test)
                test_acc_xgb = accuracy_score(y_test, test_pred_xgb)

                results["xgboost_tuned"] = {
                    "best_params": best_params_xgb,
                    "cv_score": float(best_score_xgb),
                    "test_accuracy": float(test_acc_xgb),
                    "test_f1": float(f1_score(y_test, test_pred_xgb, average="macro")),
                }
                print(
                    f"      Best params: {best_params_xgb}, CV: {best_score_xgb:.4f}, Test: {test_acc_xgb:.4f}"
                )
        except Exception as e:
            print(f"      ERROR: {str(e)}")
            results["xgboost_tuned"] = {"error": str(e)}

    return results


def run_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_name: str,
    seed: int,
    n_features_to_show: int = 10,
) -> Dict:
    """Extract feature importance from Random Forest."""
    print(f"\n  Feature importance analysis for {feature_name}...")

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_features = []
    for i in range(min(n_features_to_show, len(importances))):
        idx = indices[i]
        top_features.append(
            {
                "feature_idx": int(idx),
                "importance": float(importances[idx]),
            }
        )
        print(f"    Feature {idx}: {importances[idx]:.4f}")

    return {
        "top_features": top_features,
        "total_features": len(importances),
        "sum_top_10_importance": float(sum(f["importance"] for f in top_features)),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()

    run_log = init_run_logging(
        script_subdir="alternative-models",
        hyperparams={
            "facts": args.facts,
            "split_dir": args.split_dir,
            "out": args.out,
            "model_dir": args.model_dir,
            "seed": args.seed,
            "cv_folds": args.cv_folds,
            "has_xgboost": HAS_XGBOOST,
            "has_lightgbm": HAS_LIGHTGBM,
        },
    )

    print("=" * 70)
    print("Option E: Alternative Models Comparison")
    print("=" * 70)
    print(f"XGBoost available: {HAS_XGBOOST}")
    print(f"LightGBM available: {HAS_LIGHTGBM}")

    # Load data
    print("\nLoading facts and splits...")
    facts = load_json(args.facts)
    train_split = load_json(os.path.join(args.split_dir, "train.json"))
    val_split = load_json(os.path.join(args.split_dir, "val.json"))
    test_split = load_json(os.path.join(args.split_dir, "test.json"))

    train_ids = get_split_triplet_idx(train_split)
    val_ids = get_split_triplet_idx(val_split)
    test_ids = get_split_triplet_idx(test_split)

    facts_train, facts_val, facts_test = split_facts_by_existing_splits(
        facts, train_ids, val_ids, test_ids
    )

    print(
        f"Data: train={len(facts_train)}, val={len(facts_val)}, test={len(facts_test)}"
    )

    os.makedirs(args.model_dir, exist_ok=True)

    # Prepare feature sets
    print("\nPreparing feature sets...")

    feature_sets = {
        "coverage": build_coverage_features,
        "structural": build_structural_features,
        "combined": build_combined_features,
    }

    prepared_data = {}
    for name, builder in feature_sets.items():
        X_train, y_train, X_val, y_val, X_test, y_test, dim = prepare_splits(
            facts_train, facts_val, facts_test, builder
        )
        prepared_data[name] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "dim": dim,
        }
        print(f"  {name}: {X_train.shape[0]} train, {X_test.shape[0]} test, {dim} dims")

    all_results = {}

    # =================================================================
    # Experiment 1: Model Comparison (Default Hyperparameters)
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Model Comparison (Default Hyperparameters)")
    print("=" * 70)

    comparison_results = {}
    for feat_name, data in prepared_data.items():
        comparison_results[feat_name] = run_model_comparison(
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["y_test"],
            feat_name,
            args.seed,
            args.cv_folds,
        )

    all_results["model_comparison"] = comparison_results

    # =================================================================
    # Experiment 2: Hyperparameter Tuning (Combined features only)
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Hyperparameter Tuning (Combined Features)")
    print("=" * 70)

    # Use combined features for tuning
    combined_data = prepared_data["combined"]
    X_trainval = np.vstack([combined_data["X_train"], combined_data["X_val"]])
    y_trainval = np.concatenate([combined_data["y_train"], combined_data["y_val"]])

    tuning_results = run_hyperparameter_tuning(
        X_trainval,
        y_trainval,
        combined_data["X_test"],
        combined_data["y_test"],
        "combined",
        args.seed,
        args.cv_folds,
    )

    all_results["hyperparameter_tuning"] = tuning_results

    # =================================================================
    # Experiment 3: Feature Importance Analysis
    # =================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Feature Importance Analysis")
    print("=" * 70)

    importance_results = {}
    for feat_name in ["coverage", "combined"]:
        data = prepared_data[feat_name]
        X_trainval = np.vstack([data["X_train"], data["X_val"]])
        y_trainval = np.concatenate([data["y_train"], data["y_val"]])

        importance_results[feat_name] = run_feature_importance(
            X_trainval, y_trainval, feat_name, args.seed
        )

    all_results["feature_importance"] = importance_results

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Option E Results")
    print("=" * 70)

    print("\n1. Model Comparison (Test Accuracy by Feature Set)")

    # Collect all model names
    all_models = set()
    for feat_results in comparison_results.values():
        all_models.update(feat_results.keys())
    all_models = sorted(all_models)

    header = f"   {'Model':<20}"
    for feat_name in feature_sets.keys():
        header += f" {feat_name:<12}"
    print(header)
    print("   " + "-" * (20 + 13 * len(feature_sets)))

    for model_key in all_models:
        row = f"   {model_key:<20}"
        for feat_name in feature_sets.keys():
            if model_key in comparison_results[feat_name]:
                model_data = comparison_results[feat_name][model_key]
                if "metrics" in model_data:
                    acc = model_data["metrics"]["test"]["accuracy"]
                    row += f" {acc:<12.4f}"
                else:
                    row += f" {'ERROR':<12}"
            else:
                row += f" {'-':<12}"
        print(row)

    # Find best overall
    best_acc = 0
    best_model = ""
    best_feat = ""
    for feat_name, feat_results in comparison_results.items():
        for model_key, model_data in feat_results.items():
            if "metrics" in model_data:
                acc = model_data["metrics"]["test"]["accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_key
                    best_feat = feat_name

    print(
        f"\n   Best: {best_model} on {best_feat} features with {best_acc:.4f} accuracy"
    )

    print("\n2. Hyperparameter Tuning (Combined Features)")
    for model_key, result in tuning_results.items():
        if "test_accuracy" in result:
            print(
                f"   {model_key}: CV={result['cv_score']:.4f}, Test={result['test_accuracy']:.4f}"
            )

    # Reference baselines
    baselines = {
        "svm_coverage_original": 0.7528,
        "svm_combined_original": 0.7727,
    }

    # Save results
    output = {
        "setup": {
            "goal": "Alternative models comparison (Option E)",
            "facts": args.facts,
            "split_dir": args.split_dir,
            "seed": args.seed,
            "cv_folds": args.cv_folds,
            "has_xgboost": HAS_XGBOOST,
            "has_lightgbm": HAS_LIGHTGBM,
            "created": datetime.now().isoformat(),
        },
        "data_info": {
            "train_samples": int(len(prepared_data["coverage"]["y_train"])),
            "val_samples": int(len(prepared_data["coverage"]["y_val"])),
            "test_samples": int(len(prepared_data["coverage"]["y_test"])),
            "feature_dims": {name: data["dim"] for name, data in prepared_data.items()},
        },
        "reference_baselines": baselines,
        "experiments": all_results,
    }

    save_json(args.out, output)
    print(f"\nResults saved to: {args.out}")

    log_run_results(run_log, {"out": args.out})
    close_run_logging(run_log, status="success")


if __name__ == "__main__":
    main()
