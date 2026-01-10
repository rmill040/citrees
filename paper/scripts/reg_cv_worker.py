"""Regression metrics - WORKER.

Evaluates feature selection quality using multiple downstream regressors:
- SVR (Support Vector Regression)
- Ridge (L2 regularized linear regression)
- kNN (k-Nearest Neighbors Regressor)

Features pre-computed CV folds to ensure fair comparison across models.
"""
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import boto3
import numpy as np
import pandas as pd
import requests
from joblib import delayed, Parallel
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

DATASETS = {}
RANDOM_STATE = 1718
N_SPLITS = 5
ITERATIONS = 3

# Multiple downstream models
DOWNSTREAM_MODELS = {
    "svr": lambda: SVR(),
    "ridge": lambda: Ridge(alpha=1.0),
    "knn": lambda: KNeighborsRegressor(n_neighbors=5, weights="distance"),
    "xgb": lambda: XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0, n_jobs=1),
    "lgbm": lambda: LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
}


class DecimalEncoder(json.JSONEncoder):
    """Handle decimal data."""

    def default(self, obj: Any) -> str:
        """Cast decimal types."""
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


def cv_scores_single_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    precomputed_folds: List[tuple]
) -> Dict[str, float]:
    """Run CV for a single model with precomputed folds."""
    r2s = np.zeros(len(precomputed_folds))
    mses = np.zeros(len(precomputed_folds))
    maes = np.zeros(len(precomputed_folds))

    for fold, (train_idx, test_idx) in enumerate(precomputed_folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Build pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", DOWNSTREAM_MODELS[model_name]())
        ])

        pipeline.fit(X_train, y_train)
        y_hat = pipeline.predict(X_test)

        r2s[fold] = r2_score(y_test, y_hat)
        mses[fold] = mean_squared_error(y_test, y_hat)
        maes[fold] = mean_absolute_error(y_test, y_hat)

    return {
        f"{model_name}_r2_mean": float(np.nanmean(r2s)),
        f"{model_name}_r2_std": float(np.nanstd(r2s)),
        f"{model_name}_mse_mean": float(np.nanmean(mses)),
        f"{model_name}_mse_std": float(np.nanstd(mses)),
        f"{model_name}_mae_mean": float(np.nanmean(maes)),
        f"{model_name}_mae_std": float(np.nanstd(maes)),
    }


def cv_scores_all_models(
    *,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """Run CV for all downstream models."""
    # Precompute folds once for all models (ensures fair comparison)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    precomputed_folds = list(cv.split(X, y))

    results = {}
    for model_name in DOWNSTREAM_MODELS:
        model_results = cv_scores_single_model(
            X=X, y=y,
            model_name=model_name,
            precomputed_folds=precomputed_folds
        )
        results.update(model_results)

    return results


def run(url: str) -> None:
    """Calculate regression metrics for all downstream models."""
    global DATASETS

    ddb_table_s = boto3.resource("dynamodb", region_name="us-east-1").Table(os.environ["TABLE_NAME"] + "Metrics")
    ddb_table_f = boto3.resource("dynamodb", region_name="us-east-1").Table(os.environ["TABLE_NAME"] + "MetricsFail")

    response = requests.get(url)
    if not response.ok:
        return

    config = json.loads(response.text)
    config["config_idx"] = int(config["config_idx"])

    X, y = DATASETS[config["dataset"]]
    feature_ranks = list(map(int, config.pop("feature_ranks").split(",")))

    if int(config["n_features"]) >= 100:
        n_features_to_keep = np.arange(5, 105, 5)
    else:
        n_features_to_keep = np.arange(1, int(config["n_features"]) + 1)

    # Initialize metrics dict with all model columns
    config["metrics"] = {
        "feature_ranks": [],
        "n_features_used": [],
    }
    for model_name in DOWNSTREAM_MODELS:
        for metric in ["r2", "mse", "mae"]:
            for stat in ["mean", "std"]:
                config["metrics"][f"{model_name}_{metric}_{stat}"] = []

    try:
        for j, n_features in enumerate(n_features_to_keep, 1):
            logger.info(
                f"Config: {config['config_idx']} | Features: {n_features} ({j}/{len(n_features_to_keep)}) "
                f"| Dataset: {config['dataset']}"
            )

            X_ = X[:, feature_ranks[:n_features]]
            metrics = cv_scores_all_models(X=X_, y=y)

            config["metrics"]["feature_ranks"].append(",".join(map(str, feature_ranks[:n_features])))
            config["metrics"]["n_features_used"].append(int(n_features))
            for key, value in metrics.items():
                config["metrics"][key].append(value)

        # Write to DynamoDB
        item = json.loads(json.dumps(config), parse_float=Decimal)
        ddb_table_s.put_item(Item=item)

    except Exception as e:
        message = str(e)
        logger.error(f"Config: {config['config_idx']} | Dataset: {config['dataset']} | Error: {message}")

        item = {
            "config_idx": config["config_idx"],
            "method": config["method"],
            "hyperparameters": config.get("hyperparameters", {}),
            "dataset": config["dataset"],
            "n_samples": config["n_samples"],
            "n_features": config["n_features"],
            "message": message,
        }
        item = json.loads(json.dumps(item), parse_float=Decimal)
        ddb_table_f.put_item(Item=item)


if __name__ == "__main__":
    url = os.environ["URL"]
    here = Path(__file__).resolve()
    data_dir = here.parents[1] / "data"
    files = [f for f in os.listdir(data_dir) if f.startswith("reg_")]

    # Load datasets
    n_files = len(files)
    for j, f in enumerate(files, 1):
        dataset = f.replace("reg_", "").replace(".snappy.parquet", "")
        logger.info(f"Loading dataset {dataset} ({j}/{n_files})")
        X = pd.read_parquet(os.path.join(data_dir, f))
        y = X.pop("y").astype(float).values
        X = X.astype(float).values
        DATASETS[dataset] = (X, y)

    # Run parallel jobs
    with Parallel(n_jobs=-1, backend="loky", verbose=0) as parallel:
        response = requests.get(f"{url}/status/")
        if response.ok:
            payload = json.loads(response.text)
            n_configs_remaining = int(payload["n_configs_remaining"])
            if n_configs_remaining:
                _ = parallel(delayed(run)(url=url) for _ in range(n_configs_remaining))
