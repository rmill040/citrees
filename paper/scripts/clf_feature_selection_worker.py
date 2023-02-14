"""Classifier experiments - WORKER."""
import json
import os
import requests
from copy import deepcopy
from decimal import Decimal
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

import boto3
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from pydantic import BaseModel
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from citrees import ConditionalInferenceForestClassifier, ConditionalInferenceTreeClassifier
from citrees._selector import ClassifierSelectors, ClassifierSelectorTests


DATASETS = {}
ESTIMATORS = {
    "lr": LogisticRegression,
    "lr_l1": LogisticRegression,
    "lr_l2": LogisticRegression,
    "xgb": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "catboost": CatBoostClassifier,
    "dt": DecisionTreeClassifier,
    "rt": DecisionTreeClassifier,
    "et": ExtraTreesClassifier,
    "rf": RandomForestClassifier,
    "cit": ConditionalInferenceTreeClassifier,
    "cif": ConditionalInferenceForestClassifier,
}


class Result(BaseModel):
    """Data structure to hold single experiment result."""

    method: str
    hyperparameters: Dict[str, Any] = {}
    feature_ranks: List[int]
    dataset: str
    n_samples: int
    n_features: int
    n_classes: int


def sort_features(*, scores: np.ndarray, higher_is_better: bool) -> List[int]:
    """Sort features based on score and return up to top 100 features."""
    ranks = np.argsort(scores).tolist()
    if higher_is_better:
        ranks = ranks[::-1]
    return ranks[:100]


def run(url: str, skip: List[str]) -> None:
    """Run configuration for feature selection."""
    ddb_table = boto3.resource("dynamodb", region_name="us-east-1").Table(os.environ["TABLE_NAME"])
    response = requests.get(url)
    if response.ok:
        config = json.loads(response.text)
        config_idx = config.pop("config_idx")
        dataset = config.pop("dataset")
        n_samples = config.pop("n_samples")
        n_features = config.pop("n_features")
        n_classes = config.pop("n_classes")
        method = config.pop("method")
        if method in skip:
            logger.warning(
                f"Skipping Config Index: {config_idx} | Dataset: {dataset} | # Samples: {n_samples} | # Features: "
                f"{n_features} | # Classes: {n_classes} | Method: {method} | Hyperparameters:\n{config}"
            )
            return

        # Handle infinite recursion errors
        if method == "cif" and dataset == "isolet":
            config["max_depth"] = 1_000

        X, y = DATASETS[dataset]
        if method in ["mc", "mi", "hybrid"]:
            func = _filter_method_selector
        elif method.startswith("ptest_"):
            func = _filter_permutation_method_selector
        else:
            func = _embedding_method_selector

        logger.info(
            f"Config Index: {config_idx} | Dataset: {dataset} | # Samples: {n_samples} | # Features: {n_features} | "
            f"# Classes: {n_classes} | Method: {method} | Hyperparameters:\n{config}"
        )

        try:
            feature_ranks = func(
                method=method,
                hyperparameters=config,
                n_features=n_features,
                n_classes=n_classes,
                X=X,
                y=y,
            )

            # Transform into comma delimited string to store easier in DDB
            feature_ranks = ",".join(feature_ranks.astype(str))

            # Write to DynamoDB
            item = dict(
                config_idx=config_idx,
                method=method,
                hyperparameters=config,
                dataset=dataset,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                feature_ranks=feature_ranks,
            )

            item = json.loads(json.dumps(item), parse_float=Decimal)
            ddb_table.put_item(Item=item)

        except Exception as e:
            message = str(e)
            logger.error(
                f"Config Index: {config_idx} | Dataset: {dataset} | # Samples: {n_samples} | "
                f"# Features: {n_features} | # Classes: {n_classes} | Method: {method} | Hyperparameters:\n{config} | "
                f"Error: {message}"
            )


def _filter_method_selector(
    *,
    method: str,
    hyperparameters: Dict[str, Any],
    n_features: int,
    n_classes: int,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Filter method feature selector."""
    scores = np.zeros(n_features)
    for j in range(n_features):
        scores[j] = ClassifierSelectors[method](x=X[:, j], y=y, n_classes=n_classes, **hyperparameters)

    return sort_features(scores=scores, higher_is_better=True)


def _filter_permutation_method_selector(
    *,
    method: str,
    hyperparameters: Dict[str, Any],
    n_features: int,
    n_classes: int,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Filter permutation method feature selector."""
    _hyperparameters = deepcopy(hyperparameters)
    if hyperparameters["n_resamples"] == "minimum":
        _hyperparameters["n_resamples"] = ceil(1 / hyperparameters["alpha"])
    elif hyperparameters["n_resamples"] == "maximum":
        _hyperparameters["n_resamples"] = ceil(1 / (4 * hyperparameters["alpha"] * hyperparameters["alpha"]))
    else:
        z = norm.ppf(1 - hyperparameters["alpha"])
        _hyperparameters["n_resamples"] = ceil(z * z * (1 - hyperparameters["alpha"]) / hyperparameters["alpha"])

    key = method.split("_")[-1]
    scores = np.zeros(n_features)
    for j in range(n_features):
        scores[j] = ClassifierSelectorTests[key](x=X[:, j], y=y, n_classes=n_classes, **_hyperparameters)

    return sort_features(scores=scores, higher_is_better=False)


def _embedding_method_selector(
    *,
    method: str,
    hyperparameters: Dict[str, Any],
    n_features: int,
    n_classes: int,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Embedding method feature selector."""
    estimator = ESTIMATORS[method]
    clf = estimator(**hyperparameters).fit(X, y)
    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    else:
        scores = abs(clf.coef_.ravel())

    return sort_features(scores=scores, higher_is_better=False)


if __name__ == "__main__":
    skip = os.environ.get("SKIP", [])
    if skip:
        skip = skip.lower().strip().split(",")
    url = os.environ["URL"]

    here = Path(__file__).resolve()
    data_dir = here.parents[1] / "data"
    files = [f for f in os.listdir(data_dir) if f.startswith("clf_")]

    # Populate datasets
    n_files = len(files)
    for j, f in enumerate(files, 1):
        dataset = f.replace("clf_", "").replace(".snappy.parquet", "")
        logger.info(f"Loading dataset {dataset} ({j}/{n_files})")
        X = pd.read_parquet(os.path.join(data_dir, f))
        y = X.pop("y").astype(int).values
        X = X.astype(float).values

        # Standardize features and store dataset in memory
        X = StandardScaler().fit_transform(X)
        DATASETS[dataset] = (X, y)

    # Parallel loop
    with Parallel(n_jobs=1, backend="multiprocessing", verbose=0) as parallel:
        response = requests.get(f"{url}/status/")
        if response.ok:
            payload = json.loads(response.text)
            n_configs_remaining = int(payload["n_configs_remaining"])
            if n_configs_remaining:
                _ = parallel(delayed(run)(url=url, skip=skip) for _ in range(n_configs_remaining))
