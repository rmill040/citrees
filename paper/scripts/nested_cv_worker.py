"""Unified Nested CV Worker for Feature Selection Experiments.

Implements proper nested cross-validation where feature selection
happens INSIDE each CV fold on training data only. No data leakage.

Supports both classification and regression tasks.

Usage:
    URL=http://localhost:8000 TABLE_NAME=Clf AWS_PROFILE=personal uv run python nested_cv_worker.py

Environment variables:
    URL: Server URL to fetch configs from
    TABLE_NAME: DynamoDB table prefix (Clf or Reg)
    AWS_DEFAULT_REGION: AWS region (default: us-east-1)
    N_JOBS: Number of parallel jobs (-1 for all cores)
"""

import json
import os
import time
from copy import deepcopy
from decimal import Decimal
from math import ceil
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import requests
from boruta import BorutaPy
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)
from citrees._selector import ClassifierSelectors, ClassifierSelectorTests, RegressorSelectors, RegressorSelectorTests

# ==============================================================================
# Constants
# ==============================================================================

RANDOM_STATE = 1718
N_SPLITS = 5

DATASETS: dict[str, tuple[np.ndarray, np.ndarray]] = {}

# Downstream models for evaluation
CLF_DOWNSTREAM_MODELS = {
    "lr": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "svm": lambda: SVC(class_weight="balanced", probability=True),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"),
}

REG_DOWNSTREAM_MODELS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "svr": lambda: SVR(),
    "knn": lambda: KNeighborsRegressor(n_neighbors=5, weights="distance"),
}

# Estimator classes for embedding methods
CLF_ESTIMATORS = {
    "cit": ConditionalInferenceTreeClassifier,
    "cif": ConditionalInferenceForestClassifier,
    "rf": RandomForestClassifier,
    "et": ExtraTreesClassifier,
    "xgb": XGBClassifier,
    "lgbm": LGBMClassifier,
}

REG_ESTIMATORS = {
    "cit": ConditionalInferenceTreeRegressor,
    "cif": ConditionalInferenceForestRegressor,
    "rf": RandomForestRegressor,
    "et": ExtraTreesRegressor,
    "xgb": XGBRegressor,
    "lgbm": LGBMRegressor,
}

EMBEDDING_METHODS = {"cit", "cif", "rf", "et", "xgb", "lgbm"}


# ==============================================================================
# Feature Selection Methods
# ==============================================================================


def filter_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    task_type: str,
    n_classes: int,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """Filter method: compute association scores, select top-k."""
    n_features = X_train.shape[1]
    hp = {k: v for k, v in hyperparameters.items() if k in ["random_state"]}

    if task_type == "classification":
        selector = ClassifierSelectors[method]
        scores = np.array([
            selector(x=X_train[:, j], y=y_train.astype(np.int64), n_classes=n_classes, **hp)
            for j in range(n_features)
        ])
    else:
        selector = RegressorSelectors[method]
        # Regression selectors require standardize=True
        scores = np.array([
            selector(x=X_train[:, j], y=y_train.astype(np.float64), standardize=True, **hp)
            for j in range(n_features)
        ])

    return np.argsort(scores)[::-1][:n_select]


def permutation_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    task_type: str,
    n_classes: int,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """Permutation test method: compute p-values, select lowest."""
    n_features = X_train.shape[1]

    # Filter to only relevant hyperparameters
    valid_keys = {"alpha", "n_resamples", "early_stopping", "random_state"}
    hp = {k: v for k, v in hyperparameters.items() if k in valid_keys}

    # Set defaults
    hp.setdefault("alpha", 0.05)
    hp.setdefault("early_stopping", True)
    hp.setdefault("random_state", RANDOM_STATE)

    # Resolve n_resamples
    alpha = hp["alpha"]
    n_resamples = hp.get("n_resamples", "auto")

    if n_resamples == "minimum":
        hp["n_resamples"] = ceil(1 / alpha)
    elif n_resamples == "maximum":
        hp["n_resamples"] = ceil(1 / (4 * alpha * alpha))
    elif n_resamples == "auto" or n_resamples is None:
        z = norm.ppf(1 - alpha)
        hp["n_resamples"] = ceil(z * z * (1 - alpha) / alpha)

    key = method.split("_")[-1]  # ptest_mc -> mc

    if task_type == "classification":
        selector = ClassifierSelectorTests[key]
        pvalues = np.array([
            selector(x=X_train[:, j], y=y_train.astype(np.int64), n_classes=n_classes, **hp)
            for j in range(n_features)
        ])
    else:
        selector = RegressorSelectorTests[key]
        pvalues = np.array([
            selector(x=X_train[:, j], y=y_train.astype(np.float64), standardize=True, **hp)
            for j in range(n_features)
        ])

    return np.argsort(pvalues)[:n_select]


def embedding_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    task_type: str,
    n_classes: int,
    hyperparameters: dict,
    n_select: int,
) -> tuple[np.ndarray, Any]:
    """Embedding method: fit model, use feature importances. Returns (features, fitted_model)."""
    estimators = CLF_ESTIMATORS if task_type == "classification" else REG_ESTIMATORS
    estimator_class = estimators.get(method)
    if estimator_class is None:
        raise ValueError(f"Unknown embedding method: {method}")

    hp = {k: v for k, v in hyperparameters.items() if k not in ["method", "task_type"]}

    # Default hyperparameters
    hp.setdefault("random_state", RANDOM_STATE)
    if "verbose" not in hp:
        hp["verbose"] = 0
    if method in ["xgb", "lgbm"] and "n_jobs" not in hp:
        hp["n_jobs"] = 1

    clf = estimator_class(**hp)
    clf.fit(X_train, y_train)

    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    else:
        raise ValueError(f"Method {method} does not have feature_importances_")

    return np.argsort(scores)[::-1][:n_select], clf


def boruta_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """Boruta feature selection using Random Forest."""
    hp = hyperparameters.copy()
    max_iter = hp.pop("max_iter", 100)
    n_estimators = hp.pop("n_estimators", 100)
    random_state = hp.pop("random_state", RANDOM_STATE)

    if task_type == "classification":
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=random_state)

    boruta = BorutaPy(rf, n_estimators="auto", max_iter=max_iter, random_state=random_state, verbose=0)
    boruta.fit(X_train, y_train)

    ranking = boruta.ranking_
    return np.argsort(ranking)[:n_select]


def sklearn_permutation_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """sklearn permutation importance using RF as base estimator."""
    hp = hyperparameters.copy()
    n_repeats = hp.pop("n_repeats", 10)
    n_estimators = hp.pop("n_estimators", 100)
    random_state = hp.pop("random_state", RANDOM_STATE)

    if task_type == "classification":
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    rf.fit(X_train, y_train)
    result = permutation_importance(rf, X_train, y_train, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    return np.argsort(result.importances_mean)[::-1][:n_select]


def shap_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """SHAP-based feature selection using RF as base model (classification only)."""
    from citrees._importance import shap_importance

    hp = hyperparameters.copy()
    n_estimators = hp.pop("n_estimators", 100)
    random_state = hp.pop("random_state", RANDOM_STATE)
    max_background = hp.pop("max_background", 100)

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = shap_importance(rf, X_train, X_train, max_background)
    return np.argsort(importances)[::-1][:n_select]


def mrmr_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """mRMR feature selection (classification only)."""
    from mrmr import mrmr_classif

    df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
    target = pd.Series(y_train, name="target")
    selected = mrmr_classif(X=df, y=target, K=n_select)

    return np.array([int(f[1:]) for f in selected])


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    task_type: str,
    n_classes: int,
    hyperparameters: dict,
    n_select: int,
) -> tuple[np.ndarray, Any]:
    """Dispatch to appropriate feature selection method. Returns (features, fitted_model or None)."""

    # Filter methods
    clf_filter = ["mc", "mi", "rdc", "mrmr"]
    reg_filter = ["pc", "dc", "rdc"]
    filter_methods = clf_filter if task_type == "classification" else reg_filter

    if method in filter_methods:
        if method == "mrmr":
            return mrmr_selector(X_train, y_train, hyperparameters, n_select), None
        return filter_selector(X_train, y_train, method, task_type, n_classes, hyperparameters, n_select), None

    elif method.startswith("ptest_"):
        return permutation_selector(X_train, y_train, method, task_type, n_classes, hyperparameters, n_select), None

    elif method == "boruta":
        return boruta_selector(X_train, y_train, task_type, hyperparameters, n_select), None
    elif method == "pi":
        return sklearn_permutation_selector(X_train, y_train, task_type, hyperparameters, n_select), None
    elif method == "shap":
        return shap_selector(X_train, y_train, hyperparameters, n_select), None

    elif method in EMBEDDING_METHODS:
        return embedding_selector(X_train, y_train, method, task_type, n_classes, hyperparameters, n_select)

    else:
        raise ValueError(f"Unknown feature selection method: {method}")


# ==============================================================================
# Evaluation Functions
# ==============================================================================


def evaluate_clf_model(model, X_test: np.ndarray, y_test: np.ndarray, n_classes: int) -> dict[str, float]:
    """Evaluate a classification model."""
    y_pred = model.predict(X_test)
    results = {
        "acc": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="micro")),
    }

    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if n_classes == 2:
                y_prob = y_prob[:, 1]
                results["auc"] = float(roc_auc_score(y_test, y_prob))
            else:
                y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
                results["auc"] = float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        results["auc"] = float("nan")

    return results


def evaluate_reg_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Evaluate a regression model."""
    y_pred = model.predict(X_test)
    return {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def evaluate_downstream(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_features: np.ndarray,
    task_type: str,
    n_classes: int,
) -> dict[str, dict[str, float]]:
    """Evaluate selected features with downstream models."""
    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    downstream_models = CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS
    evaluate_fn = evaluate_clf_model if task_type == "classification" else evaluate_reg_model

    results = {}
    for model_name, model_factory in downstream_models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model_factory())])
        pipeline.fit(X_train_sel, y_train)

        if task_type == "classification":
            results[model_name] = evaluate_fn(pipeline, X_test_sel, y_test, n_classes)
        else:
            results[model_name] = evaluate_fn(pipeline, X_test_sel, y_test)

    return results


# ==============================================================================
# Nested CV
# ==============================================================================


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task_type: str,
    hyperparameters: dict,
    n_features_list: list[int],
    n_classes: int,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Run nested CV experiment with proper feature selection."""

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    max_features = max(n_features_list)

    results = {
        "n_features_list": n_features_list,
        "folds": [],
    }

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Feature selection on TRAIN only
        tic = time.perf_counter()
        selected_features, fitted_model = select_features(
            X_train, y_train, method, task_type, n_classes, hyperparameters, max_features
        )
        selection_time = time.perf_counter() - tic

        fold_data = {
            "fold": fold_idx,
            "selected_features": selected_features.tolist(),
            "selection_time": selection_time,
            "downstream_metrics": {},
        }

        # Embedding model metrics (if applicable)
        if method in EMBEDDING_METHODS and fitted_model is not None:
            if task_type == "classification":
                fold_data["embedding_metrics"] = evaluate_clf_model(fitted_model, X_test, y_test, n_classes)
            else:
                fold_data["embedding_metrics"] = evaluate_reg_model(fitted_model, X_test, y_test)

        # Evaluate downstream models for each n_features
        for n_features in n_features_list:
            top_k = selected_features[:n_features]
            fold_data["downstream_metrics"][n_features] = evaluate_downstream(
                X_train, y_train, X_test, y_test, top_k, task_type, n_classes
            )

        results["folds"].append(fold_data)

    # Aggregate across folds
    results["aggregated"] = aggregate_results(results, task_type)

    return results


def aggregate_results(results: dict, task_type: str) -> dict:
    """Aggregate metrics across folds."""
    aggregated = {}

    n_features_list = results["n_features_list"]
    downstream_models = list(CLF_DOWNSTREAM_MODELS.keys()) if task_type == "classification" else list(REG_DOWNSTREAM_MODELS.keys())
    metrics = ["acc", "f1", "auc"] if task_type == "classification" else ["mse", "mae", "r2"]

    for n_features in n_features_list:
        agg = {}
        for model_name in downstream_models:
            for metric in metrics:
                values = [
                    f["downstream_metrics"][n_features][model_name].get(metric, float("nan"))
                    for f in results["folds"]
                ]
                agg[f"{model_name}_{metric}_mean"] = float(np.nanmean(values))
                agg[f"{model_name}_{metric}_std"] = float(np.nanstd(values))
        aggregated[n_features] = agg

    # Embedding metrics aggregation (if present)
    if results["folds"] and "embedding_metrics" in results["folds"][0]:
        emb_agg = {}
        for metric in metrics:
            values = [f["embedding_metrics"].get(metric, float("nan")) for f in results["folds"]]
            emb_agg[f"embedding_{metric}_mean"] = float(np.nanmean(values))
            emb_agg[f"embedding_{metric}_std"] = float(np.nanstd(values))
        aggregated["embedding"] = emb_agg

    return aggregated


# ==============================================================================
# Worker Entry Point
# ==============================================================================


def run(url: str) -> None:
    """Fetch config from server and run nested CV experiment."""
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    table_prefix = os.environ.get("TABLE_NAME", "Clf")

    ddb_table_s = boto3.resource("dynamodb", region_name=region).Table(f"{table_prefix}NestedCV")
    ddb_table_f = boto3.resource("dynamodb", region_name=region).Table(f"{table_prefix}NestedCVFail")

    url = url.rstrip("/")

    response = requests.get(url)
    if not response.ok:
        return

    config = json.loads(response.text)
    if not config:
        return

    config_idx = config.pop("config_idx")
    task_type = config.pop("task_type")
    dataset = config.pop("dataset")
    n_samples = config.pop("n_samples")
    n_features = config.pop("n_features")
    n_classes = config.pop("n_classes", 2)
    method = config.pop("method")

    logger.info(
        f"Config: {config_idx} | Task: {task_type} | Dataset: {dataset} | Method: {method} | "
        f"Samples: {n_samples} | Features: {n_features}"
    )

    try:
        X, y = DATASETS[dataset]

        if n_features >= 100:
            n_features_list = list(range(5, 105, 5))
        else:
            n_features_list = list(range(1, min(n_features, 20) + 1))

        tic = time.perf_counter()
        results = run_nested_cv(
            X=X, y=y,
            method=method,
            task_type=task_type,
            hyperparameters=config,
            n_features_list=n_features_list,
            n_classes=n_classes,
        )
        elapsed = time.perf_counter() - tic

        item = {
            "config_idx": config_idx,
            "task_type": task_type,
            "dataset": dataset,
            "method": method,
            "hyperparameters": config,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "elapsed_seconds": Decimal(str(round(elapsed, 4))),
            "results": json.dumps(results),
        }
        item = json.loads(json.dumps(item), parse_float=Decimal)
        ddb_table_s.put_item(Item=item)
        logger.info(f"Config: {config_idx} | Completed in {elapsed:.2f}s")

    except Exception as e:
        logger.error(f"Config: {config_idx} | Error: {e}")
        import traceback
        item = {
            "config_idx": config_idx,
            "task_type": task_type,
            "dataset": dataset,
            "method": method,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        item = json.loads(json.dumps(item), parse_float=Decimal)
        ddb_table_f.put_item(Item=item)


# ==============================================================================
# Local Testing
# ==============================================================================


def test_local():
    """Test the worker locally without AWS."""
    from sklearn.datasets import make_classification, make_regression

    print("=" * 70)
    print("LOCAL NESTED CV WORKER TEST")
    print("=" * 70)

    # Classification
    print("\n=== CLASSIFICATION ===")
    X_clf, y_clf = make_classification(n_samples=100, n_features=10, n_informative=3, random_state=42)

    for method in ["mc", "rf"]:
        print(f"\n--- {method} ---")
        results = run_nested_cv(
            X=X_clf, y=y_clf, method=method, task_type="classification",
            hyperparameters={"random_state": 42, "verbose": 0},
            n_features_list=[3], n_classes=2, n_splits=2,
        )
        agg = results["aggregated"]
        print(f"  LR acc={agg[3]['lr_acc_mean']:.3f}, SVM acc={agg[3]['svm_acc_mean']:.3f}")
        if "embedding" in agg:
            print(f"  Embedding acc={agg['embedding']['embedding_acc_mean']:.3f}")

    # Regression
    print("\n=== REGRESSION ===")
    X_reg, y_reg = make_regression(n_samples=100, n_features=10, n_informative=3, random_state=42)

    for method in ["pc", "rf"]:
        print(f"\n--- {method} ---")
        results = run_nested_cv(
            X=X_reg, y=y_reg, method=method, task_type="regression",
            hyperparameters={"random_state": 42, "verbose": 0},
            n_features_list=[3], n_classes=0, n_splits=2,
        )
        agg = results["aggregated"]
        print(f"  Ridge R2={agg[3]['ridge_r2_mean']:.3f}, SVR R2={agg[3]['svr_r2_mean']:.3f}")
        if "embedding" in agg:
            print(f"  Embedding R2={agg['embedding']['embedding_r2_mean']:.3f}")

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    if os.environ.get("LOCAL_TEST"):
        test_local()
    else:
        url = os.environ["URL"]
        table_prefix = os.environ.get("TABLE_NAME", "Clf")
        here = Path(__file__).resolve()
        data_dir = here.parents[1] / "data"

        file_prefix = "clf_" if table_prefix.lower().startswith("clf") else "reg_"
        files = [f for f in os.listdir(data_dir) if f.startswith(file_prefix)]

        for f in files:
            dataset = f.replace(file_prefix, "").replace(".snappy.parquet", "").replace(".parquet", "")
            logger.info(f"Loading {dataset}")
            df = pd.read_parquet(data_dir / f)
            y = df.pop("y").values
            y = y.astype(int) if file_prefix == "clf_" else y.astype(float)
            X = StandardScaler().fit_transform(df.values)
            DATASETS[dataset] = (X, y)

        n_jobs = int(os.environ.get("N_JOBS", -1))
        url = url.rstrip("/")

        with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
            response = requests.get(f"{url}/status/")
            if response.ok:
                n_remaining = int(json.loads(response.text)["n_configs_remaining"])
                if n_remaining:
                    parallel(delayed(run)(url=url) for _ in range(n_remaining))
