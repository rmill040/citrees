"""Regression experiments - WORKER."""
import json
import os
from copy import deepcopy
from decimal import Decimal
from math import ceil
from pathlib import Path
from typing import Any, Dict, List

import boto3
import numpy as np
import pandas as pd
import requests
from catboost import CatBoostRegressor
from joblib import delayed, Parallel
from lightgbm import LGBMRegressor
from loguru import logger
from pydantic import BaseModel
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from citrees import ConditionalInferenceForestRegressor, ConditionalInferenceTreeRegressor
from citrees._selector import RegressorSelectors, RegressorSelectorTests

DATASETS = {}
ESTIMATORS = {
    "lr": LinearRegression,
    "lr_l1": Lasso,
    "lr_l2": Ridge,
    "xgb": XGBRegressor,
    "lightgbm": LGBMRegressor,
    "catboost": CatBoostRegressor,
    "dt": DecisionTreeRegressor,
    "rt": DecisionTreeRegressor,
    "et": ExtraTreesRegressor,
    "rf": RandomForestRegressor,
    "cit": ConditionalInferenceTreeRegressor,
    "cif": ConditionalInferenceForestRegressor,
}


class Result(BaseModel):
    """Data structure to hold single experiment result."""

    method: str
    hyperparameters: Dict[str, Any] = {}
    feature_ranks: List[int]
    dataset: str
    n_samples: int
    n_features: int


def sort_features(*, scores: np.ndarray, higher_is_better: bool) -> List[int]:
    """Sort features based on score and return up to top 100 features."""
    ranks = np.argsort(scores).tolist()
    if higher_is_better:
        ranks = ranks[::-1]
    return ranks[:100]


def run(url: str, skip: List[str]) -> None:
    """Run configuration for feature selection."""
    ddb_table_s = boto3.resource("dynamodb", region_name="us-east-1").Table(os.environ["TABLE_NAME"])
    ddb_table_f = boto3.resource("dynamodb", region_name="us-east-1").Table(os.environ["TABLE_NAME"] + "Fail")

    response = requests.get(url)
    if response.ok:
        config = json.loads(response.text)
        config_idx = config.pop("config_idx")
        dataset = config.pop("dataset")
        n_samples = config.pop("n_samples")
        n_features = config.pop("n_features")
        method = config.pop("method")
        if method in skip:
            logger.warning(
                f"Skipping Config Index: {config_idx} | Dataset: {dataset} | # Samples: {n_samples} | # Features: "
                f"{n_features} | Method: {method} | Hyperparameters:\n{config}"
            )
            return

        if int(os.environ["N_JOBS_INNER"]) == -1 and config.get("n_jobs") == 1:
            config["n_jobs"] = -1

        X, y = DATASETS[dataset]
        if method in ["pc", "dc", "hybrid"]:
            func = _filter_method_selector
        elif method.startswith("ptest_"):
            func = _filter_permutation_method_selector
        else:
            func = _embedding_method_selector

        logger.info(
            f"Config Index: {config_idx} | Dataset: {dataset} | # Samples: {n_samples} | # Features: {n_features} | "
            f"Method: {method} | Hyperparameters:\n{config}"
        )

        try:
            feature_ranks = func(
                method=method,
                hyperparameters=config,
                n_features=n_features,
                X=X,
                y=y,
            )

            # Transform into comma delimited string to store easier in DDB
            feature_ranks = ",".join(list(map(str, feature_ranks)))

            # Write to DynamoDB
            item = {
                "config_idx": config_idx,
                "method": method,
                "hyperparameters": config,
                "dataset": dataset,
                "n_samples": n_samples,
                "n_features": n_features,
                "feature_ranks": feature_ranks,
            }

            item = json.loads(json.dumps(item), parse_float=Decimal)
            ddb_table_s.put_item(Item=item)

        except Exception as e:
            message = str(e)
            logger.error(
                f"Config Index: {config_idx} | Dataset: {dataset} | # Samples: {n_samples} | "
                f"# Features: {n_features} | Method: {method} | Hyperparameters:\n{config} | "
                f"Error: {message}"
            )

            # Write to DynamoDB
            item = {
                "config_idx": config_idx,
                "method": method,
                "hyperparameters": config,
                "dataset": dataset,
                "n_samples": n_samples,
                "n_features": n_features,
                "message": message,
            }

            item = json.loads(json.dumps(item), parse_float=Decimal)
            ddb_table_f.put_item(Item=item)


def _filter_method_selector(
    *,
    method: str,
    hyperparameters: Dict[str, Any],
    n_features: int,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Filter method feature selector."""
    scores = Parallel(n_jobs=int(os.environ["N_JOBS_INNER"]), backend="loky")(
        delayed(RegressorSelectors[method])(x=X[:, j], y=y, **hyperparameters) for j in range(n_features)
    )

    return sort_features(scores=scores, higher_is_better=True)


def _filter_permutation_method_selector(
    *,
    method: str,
    hyperparameters: Dict[str, Any],
    n_features: int,
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
    scores = Parallel(n_jobs=int(os.environ["N_JOBS_INNER"]), backend="loky")(
        delayed(RegressorSelectorTests[key])(x=X[:, j], y=y, **_hyperparameters) for j in range(n_features)
    )

    return sort_features(scores=scores, higher_is_better=False)


def _embedding_method_selector(
    *,
    method: str,
    hyperparameters: Dict[str, Any],
    n_features: int,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Embedding method feature selector."""
    estimator = ESTIMATORS[method]
    reg = estimator(**hyperparameters).fit(X, y)
    if hasattr(reg, "feature_importances_"):
        scores = reg.feature_importances_
    else:
        scores = np.abs(reg.coef_)
        if scores.shape[0] > 1:
            scores = scores.sum(axis=0)
        scores = scores.ravel()

    return sort_features(scores=scores, higher_is_better=True)


if __name__ == "__main__":
    skip = os.environ.get("SKIP", [])
    if skip:
        skip = skip.lower().strip().split(",")
    url = os.environ["URL"]

    here = Path(__file__).resolve()
    data_dir = here.parents[1] / "data"
    files = [f for f in os.listdir(data_dir) if f.startswith("reg_")]

    # Populate datasets
    n_files = len(files)
    for j, f in enumerate(files, 1):
        dataset = f.replace("reg_", "").replace(".snappy.parquet", "")
        logger.info(f"Loading dataset {dataset} ({j}/{n_files})")
        X = pd.read_parquet(os.path.join(data_dir, f))
        y = X.pop("y").astype(int).values
        X = X.astype(float).values

        # Standardize features and store dataset in memory
        X = StandardScaler().fit_transform(X)
        DATASETS[dataset] = (X, y)

    # Parallel loop
    with Parallel(n_jobs=int(os.environ["N_JOBS_OUTER"]), backend="loky", verbose=0) as parallel:
        response = requests.get(f"{url}/status/")
        if response.ok:
            payload = json.loads(response.text)
            n_configs_remaining = int(payload["n_configs_remaining"])
            if n_configs_remaining:
                _ = parallel(delayed(run)(url=url, skip=skip) for _ in range(n_configs_remaining))
