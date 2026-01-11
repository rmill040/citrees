"""Unified Nested CV Worker for Feature Selection Experiments.

Implements proper nested cross-validation where feature selection
happens INSIDE each CV fold on training data only. No data leakage.

Usage:
    python nested_cv_worker.py

Environment variables:
    URL: Server URL to fetch configs from
    TABLE_NAME: DynamoDB table name for results
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
from joblib import Parallel, delayed
from loguru import logger
from scipy.stats import norm
from boruta import BorutaPy
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceTreeClassifier
from citrees._selector import ClassifierSelectors, ClassifierSelectorTests

# ==============================================================================
# Constants
# ==============================================================================

RANDOM_STATE = 1718
N_SPLITS = 5

DATASETS: dict[str, tuple[np.ndarray, np.ndarray]] = {}

# Downstream models for evaluation
DOWNSTREAM_MODELS = {
    "lr": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
}

# Estimator classes for embedding methods (feature importance-based)
ESTIMATORS = {
    "cit": ConditionalInferenceTreeClassifier,
    "cif": ConditionalInferenceForestClassifier,
    "rf": RandomForestClassifier,
    "et": ExtraTreesClassifier,
    "xgb": XGBClassifier,
    "lgbm": LGBMClassifier,
}


# ==============================================================================
# Feature Selection Methods
# ==============================================================================


def filter_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    n_classes: int,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """Filter method: compute association scores, select top-k."""
    n_features = X_train.shape[1]
    hp = {k: v for k, v in hyperparameters.items() if k in ["random_state"]}

    scores = np.array([
        ClassifierSelectors[method](
            x=X_train[:, j],
            y=y_train.astype(np.int64),
            n_classes=n_classes,
            **hp,
        )
        for j in range(n_features)
    ])
    return np.argsort(scores)[::-1][:n_select]


def permutation_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    n_classes: int,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """Permutation test method: compute p-values, select lowest."""
    n_features = X_train.shape[1]
    hp = deepcopy(hyperparameters)

    # Resolve n_resamples
    alpha = hp.get("alpha", 0.05)
    n_resamples = hp.get("n_resamples", "auto")

    if n_resamples == "minimum":
        hp["n_resamples"] = ceil(1 / alpha)
    elif n_resamples == "maximum":
        hp["n_resamples"] = ceil(1 / (4 * alpha * alpha))
    elif n_resamples == "auto" or n_resamples is None:
        z = norm.ppf(1 - alpha)
        hp["n_resamples"] = ceil(z * z * (1 - alpha) / alpha)

    if "random_state" not in hp:
        hp["random_state"] = RANDOM_STATE

    key = method.split("_")[-1]  # ptest_mc -> mc
    pvalues = np.array([
        ClassifierSelectorTests[key](
            x=X_train[:, j],
            y=y_train.astype(np.int64),
            n_classes=n_classes,
            **hp,
        )
        for j in range(n_features)
    ])
    return np.argsort(pvalues)[:n_select]


def embedding_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    n_classes: int,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """Embedding method: fit model, use feature importances."""
    estimator_class = ESTIMATORS.get(method)
    if estimator_class is None:
        raise ValueError(f"Unknown embedding method: {method}")

    hp = {k: v for k, v in hyperparameters.items() if k not in ["method"]}
    clf = estimator_class(**hp).fit(X_train, y_train)

    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    else:
        raise ValueError(f"Method {method} does not have feature_importances_")

    return np.argsort(scores)[::-1][:n_select]


def boruta_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """Boruta feature selection using Random Forest."""
    hp = hyperparameters.copy()
    max_iter = hp.pop("max_iter", 100)
    n_estimators = hp.pop("n_estimators", 100)
    random_state = hp.pop("random_state", RANDOM_STATE)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    boruta = BorutaPy(
        rf,
        n_estimators="auto",
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )
    boruta.fit(X_train, y_train)

    # Get ranking (1 = confirmed, 2 = tentative, 3+ = rejected)
    # Return features sorted by ranking
    ranking = boruta.ranking_
    return np.argsort(ranking)[:n_select]


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    n_classes: int,
    hyperparameters: dict,
    n_select: int,
) -> np.ndarray:
    """Dispatch to appropriate feature selection method."""
    if method in ["mc", "mi", "rdc"]:
        return filter_selector(X_train, y_train, method, n_classes, hyperparameters, n_select)
    elif method.startswith("ptest_"):
        return permutation_selector(X_train, y_train, method, n_classes, hyperparameters, n_select)
    elif method == "boruta":
        return boruta_selector(X_train, y_train, hyperparameters, n_select)
    elif method in ESTIMATORS:
        return embedding_selector(X_train, y_train, method, n_classes, hyperparameters, n_select)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")


# ==============================================================================
# Downstream Evaluation
# ==============================================================================


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_features: np.ndarray,
    n_classes: int,
    model_name: str = "lr",
) -> dict[str, float]:
    """Evaluate a single fold with selected features."""
    X_train_sel = X_train[:, selected_features]
    X_test_sel = X_test[:, selected_features]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DOWNSTREAM_MODELS[model_name]()),
    ])

    pipeline.fit(X_train_sel, y_train)
    y_pred = pipeline.predict(X_test_sel)

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="micro")),
    }

    # AUC
    try:
        y_prob = pipeline.predict_proba(X_test_sel)
        if n_classes == 2:
            y_prob = y_prob[:, 1]
            results["auc"] = float(roc_auc_score(y_test, y_prob))
        else:
            y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
            results["auc"] = float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted"))
    except Exception:
        results["auc"] = float("nan")

    return results


# ==============================================================================
# Nested CV
# ==============================================================================


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    hyperparameters: dict,
    n_features_list: list[int],
    n_classes: int,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """
    Run nested CV experiment with proper feature selection.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    method : str
        Feature selection method.
    hyperparameters : dict
        Method hyperparameters.
    n_features_list : list[int]
        List of feature counts to evaluate (e.g., [5, 10, 15, ...]).
    n_classes : int
        Number of classes.
    n_splits : int
        Number of CV folds.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Results including per-fold metrics and selected features.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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
        selected_features = select_features(
            X_train, y_train, method, n_classes, hyperparameters, max_features
        )
        selection_time = time.perf_counter() - tic

        fold_data = {
            "fold": fold_idx,
            "selected_features": selected_features.tolist(),
            "selection_time": selection_time,
            "metrics_by_n_features": {},
        }

        # Evaluate for each n_features
        for n_features in n_features_list:
            top_k = selected_features[:n_features]
            metrics = evaluate_fold(X_train, y_train, X_test, y_test, top_k, n_classes)
            fold_data["metrics_by_n_features"][n_features] = metrics

        results["folds"].append(fold_data)

    # Aggregate across folds for each n_features
    results["aggregated"] = {}
    for n_features in n_features_list:
        agg = {}
        for metric in ["accuracy", "f1", "auc"]:
            values = [f["metrics_by_n_features"][n_features][metric] for f in results["folds"]]
            agg[f"{metric}_mean"] = float(np.nanmean(values))
            agg[f"{metric}_std"] = float(np.nanstd(values))
        results["aggregated"][n_features] = agg

    return results


# ==============================================================================
# Worker Entry Point
# ==============================================================================


def run(url: str) -> None:
    """Fetch config from server and run nested CV experiment."""
    ddb_table_s = boto3.resource("dynamodb", region_name="us-east-1").Table(
        os.environ["TABLE_NAME"] + "NestedCV"
    )
    ddb_table_f = boto3.resource("dynamodb", region_name="us-east-1").Table(
        os.environ["TABLE_NAME"] + "NestedCVFail"
    )

    response = requests.get(url)
    if not response.ok:
        return

    config = json.loads(response.text)
    config_idx = config.pop("config_idx")
    dataset = config.pop("dataset")
    n_samples = config.pop("n_samples")
    n_features = config.pop("n_features")
    n_classes = config.pop("n_classes")
    method = config.pop("method")

    logger.info(
        f"Config: {config_idx} | Dataset: {dataset} | Method: {method} | "
        f"Samples: {n_samples} | Features: {n_features}"
    )

    try:
        X, y = DATASETS[dataset]

        # Determine feature counts to evaluate
        if n_features >= 100:
            n_features_list = list(range(5, 105, 5))
        else:
            n_features_list = list(range(1, n_features + 1))

        # Run nested CV
        tic = time.perf_counter()
        results = run_nested_cv(
            X=X,
            y=y,
            method=method,
            hyperparameters=config,
            n_features_list=n_features_list,
            n_classes=n_classes,
        )
        elapsed = time.perf_counter() - tic

        # Store results
        item = {
            "config_idx": config_idx,
            "dataset": dataset,
            "method": method,
            "hyperparameters": config,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "elapsed_seconds": Decimal(str(round(elapsed, 4))),
            "results": json.dumps(results),  # Store as JSON string
        }
        item = json.loads(json.dumps(item), parse_float=Decimal)
        ddb_table_s.put_item(Item=item)

    except Exception as e:
        logger.error(f"Config: {config_idx} | Error: {e}")
        item = {
            "config_idx": config_idx,
            "dataset": dataset,
            "method": method,
            "message": str(e),
        }
        item = json.loads(json.dumps(item), parse_float=Decimal)
        ddb_table_f.put_item(Item=item)


# ==============================================================================
# Local Testing
# ==============================================================================


def test_local():
    """Test the worker locally without AWS."""
    from sklearn.datasets import make_classification

    print("=" * 70)
    print("LOCAL NESTED CV WORKER TEST")
    print("=" * 70)

    X, y = make_classification(
        n_samples=500,
        n_features=50,
        n_informative=10,
        random_state=42,
    )

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(np.unique(y))}")

    # Test filter method
    print("\n--- Testing mc (filter) ---")
    results = run_nested_cv(
        X=X, y=y,
        method="mc",
        hyperparameters={"random_state": 42},
        n_features_list=[5, 10, 20],
        n_classes=2,
    )
    for n_feat, agg in results["aggregated"].items():
        print(f"  {n_feat} features: acc={agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")

    # Test permutation method
    print("\n--- Testing ptest_mc (permutation) ---")
    results = run_nested_cv(
        X=X, y=y,
        method="ptest_mc",
        hyperparameters={"alpha": 0.05, "n_resamples": "auto", "early_stopping": True},
        n_features_list=[5, 10, 20],
        n_classes=2,
    )
    for n_feat, agg in results["aggregated"].items():
        print(f"  {n_feat} features: acc={agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")

    # Test embedding method
    print("\n--- Testing cit (embedding) ---")
    results = run_nested_cv(
        X=X, y=y,
        method="cit",
        hyperparameters={"random_state": 42, "verbose": 0},
        n_features_list=[5, 10, 20],
        n_classes=2,
    )
    for n_feat, agg in results["aggregated"].items():
        print(f"  {n_feat} features: acc={agg['accuracy_mean']:.4f} +/- {agg['accuracy_std']:.4f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    if os.environ.get("LOCAL_TEST"):
        test_local()
    else:
        # Production mode
        url = os.environ["URL"]
        here = Path(__file__).resolve()
        data_dir = here.parents[1] / "data"
        files = [f for f in os.listdir(data_dir) if f.startswith("clf_")]

        # Load datasets
        for f in files:
            dataset = f.replace("clf_", "").replace(".snappy.parquet", "")
            logger.info(f"Loading {dataset}")
            df = pd.read_parquet(data_dir / f)
            y = df.pop("y").astype(int).values
            X = StandardScaler().fit_transform(df.values)
            DATASETS[dataset] = (X, y)

        # Run parallel
        n_jobs = int(os.environ.get("N_JOBS", -1))
        with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
            response = requests.get(f"{url}/status/")
            if response.ok:
                n_remaining = int(json.loads(response.text)["n_configs_remaining"])
                if n_remaining:
                    parallel(delayed(run)(url=url) for _ in range(n_remaining))
