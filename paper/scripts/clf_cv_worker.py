"""Classifier metrics - WORKER."""
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import boto3
import numpy as np
import pandas as pd
import requests
from joblib import delayed, Parallel
from loguru import logger
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.svm import SVC

DATASETS = {}
RANDOM_STATE = 1718
N_SPLITS = 5
ITERATIONS = 3


class DecimalEncoder(json.JSONEncoder):
    """Handle decimal data."""

    def default(self, obj: Any) -> str:
        """Cast decimal types."""
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


def cv_scores(*, X: np.ndarray, y: np.ndarray, n_classes: int) -> Dict[str, Any]:
    """Run multiple iterations of cross-validation."""
    accs = np.zeros(ITERATIONS)
    aucs = np.zeros(ITERATIONS)
    f1s = np.zeros(ITERATIONS)

    for i in range(ITERATIONS):
        # SGD parameters
        hyperparameters = {
            "class_weight": "balanced",
        }

        # Variables for CV
        tmp_accs = np.zeros(N_SPLITS)
        tmp_aucs = np.zeros(N_SPLITS)
        tmp_f1s = np.zeros(N_SPLITS)
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        # Run CV
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            # Split data
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Define pipeline
            pipeline = Pipeline([("ss", StandardScaler()), ("clf", SVC(**hyperparameters))])

            # Fit pipeline and calculate metrics
            pipeline.fit(X_train, y_train)
            y_hat = pipeline.predict(X_test)
            tmp_accs[fold] = np.mean(y_test == y_hat)
            tmp_f1s[fold] = f1_score(y_test, y_hat, average="micro")

            # Calculate AUC
            if n_classes > 2:
                y_test = label_binarize(y_test, classes=list(np.unique(y)))
            y_df = pipeline.decision_function(X_test)
            tmp_aucs[fold] = roc_auc_score(y_test, y_df)

        # Average results
        accs[i] = tmp_accs.mean()
        aucs[i] = tmp_aucs.mean()
        f1s[i] = tmp_f1s.mean()

    # Return averaged results
    return {
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
    }


def run(url: str) -> None:
    """Calculate classifier metrics."""
    global DATASETS

    ddb_table_s = boto3.resource("dynamodb", region_name="us-east-1").Table(os.environ["TABLE_NAME"] + "Metrics")
    ddb_table_f = boto3.resource("dynamodb", region_name="us-east-1").Table(os.environ["TABLE_NAME"] + "MetricsFail")

    response = requests.get(url)
    if response.ok:
        config = json.loads(response.text)
        config["config_idx"] = int(config["config_idx"])  # just make sure again, cast to int

        # Get dataset and relevant metadata
        X, y = DATASETS[config["dataset"]]
        n_classes = int(config["n_classes"])
        feature_ranks = list(map(int, config["feature_ranks"].split(",")))

        if int(config["n_features"]) >= 100:
            n_features_to_keep = np.arange(5, 105, 5)
        else:
            n_features_to_keep = np.arange(1, int(config["n_features"]) + 1)

        config["metrics"] = {
            "feature_ranks": [],
            "n_features_used": [],
            "accuracy_mean": [],
            "accuracy_std": [],
            "f1_mean": [],
            "f1_std": [],
            "auc_mean": [],
            "auc_std": [],
        }

        try:
            for j, n_features in enumerate(n_features_to_keep, 1):

                logger.info(
                    f"Configuration: {config['config_idx']} | # Features - {n_features}: {j}/{len(n_features_to_keep)} "
                    f"| Dataset: {config['dataset']}"
                )

                X_ = X[:, feature_ranks[:n_features]].reshape(-1, n_features)
                metrics = cv_scores(X=X_, y=y, n_classes=n_classes)

                # Update metadata
                config["metrics"]["feature_ranks"].append(",".join(map(str, feature_ranks[:n_features])))
                config["metrics"]["n_features_used"].append(int(n_features))
                for key, value in metrics.items():
                    config["metrics"][key].append(value)

            # Write to DynamoDB
            item = json.loads(json.dumps(config), parse_float=Decimal)
            ddb_table_s.put_item(Item=item)
        except Exception as e:
            message = str(e)
            logger.error(f"Configuration: {config['config_idx']} | Dataset: {config['dataset']} | Error: {message}")

            # Write to DynamoDB
            item = {
                "config_idx": config["config_idx"],
                "method": config["method"],
                "hyperparameters": config["hyperparameters"],
                "dataset": config["dataset"],
                "n_samples": config["n_samples"],
                "n_features": config["n_features"],
                "n_classes": config["n_classes"],
                "message": message,
            }

            item = json.loads(json.dumps(item), parse_float=Decimal)
            ddb_table_f.put_item(Item=item)


if __name__ == "__main__":
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

        # Store dataset in memory
        DATASETS[dataset] = (X, y)

    # Parallel loop
    with Parallel(n_jobs=-1, backend="loky", verbose=0) as parallel:
        response = requests.get(f"{url}/status/")
        if response.ok:
            payload = json.loads(response.text)
            n_configs_remaining = int(payload["n_configs_remaining"])
            if n_configs_remaining:
                _ = parallel(delayed(run)(url=url) for _ in range(n_configs_remaining))
