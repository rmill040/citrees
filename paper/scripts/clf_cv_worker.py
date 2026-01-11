"""Classifier metrics - WORKER.

Evaluates feature selection quality using multiple downstream classifiers:
- SVM (Support Vector Machine)
- LR (Logistic Regression)
- kNN (k-Nearest Neighbors)

Features pre-computed CV folds to ensure fair comparison across models.
"""

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

DATASETS = {}
RANDOM_STATE = 1718
N_SPLITS = 5
ITERATIONS = 3

# Multiple downstream models
DOWNSTREAM_MODELS = {
    "svm": lambda: SVC(class_weight="balanced", probability=True),
    "lr": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs"),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "xgb": lambda: XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=1,
    ),
    "lgbm": lambda: LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=1,
    ),
}


class DecimalEncoder(json.JSONEncoder):
    """Handle decimal data."""

    def default(self, obj: Any) -> str:
        """Cast decimal types."""
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


def cv_scores_single_model(
    *, X: np.ndarray, y: np.ndarray, n_classes: int, model_name: str, precomputed_folds: list[tuple]
) -> dict[str, float]:
    """Run CV for a single model with precomputed folds."""
    accs = np.zeros(len(precomputed_folds))
    aucs = np.zeros(len(precomputed_folds))
    f1s = np.zeros(len(precomputed_folds))

    for fold, (train_idx, test_idx) in enumerate(precomputed_folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Build pipeline
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", DOWNSTREAM_MODELS[model_name]())]
        )

        pipeline.fit(X_train, y_train)
        y_hat = pipeline.predict(X_test)

        accs[fold] = np.mean(y_test == y_hat)
        f1s[fold] = f1_score(y_test, y_hat, average="micro")

        # AUC calculation
        try:
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)
                if n_classes == 2:
                    y_prob = y_prob[:, 1]
            else:
                y_prob = pipeline.decision_function(X_test)

            if n_classes > 2:
                y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
                aucs[fold] = roc_auc_score(
                    y_test_bin, y_prob, multi_class="ovr", average="weighted"
                )
            else:
                aucs[fold] = roc_auc_score(y_test, y_prob)
        except Exception:
            aucs[fold] = np.nan

    return {
        f"{model_name}_accuracy_mean": float(np.nanmean(accs)),
        f"{model_name}_accuracy_std": float(np.nanstd(accs)),
        f"{model_name}_f1_mean": float(np.nanmean(f1s)),
        f"{model_name}_f1_std": float(np.nanstd(f1s)),
        f"{model_name}_auc_mean": float(np.nanmean(aucs)),
        f"{model_name}_auc_std": float(np.nanstd(aucs)),
    }


def cv_scores_all_models(*, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, Any]:
    """Run CV for all downstream models."""
    # Precompute folds once for all models (ensures fair comparison)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    precomputed_folds = list(cv.split(X, y))

    results = {}
    for model_name in DOWNSTREAM_MODELS:
        model_results = cv_scores_single_model(
            X=X,
            y=y,
            n_classes=n_classes,
            model_name=model_name,
            precomputed_folds=precomputed_folds,
        )
        results.update(model_results)

    return results


def run(url: str) -> None:
    """Calculate classifier metrics for all downstream models."""
    global DATASETS

    ddb_table_s = boto3.resource("dynamodb", region_name="us-east-1").Table(
        os.environ["TABLE_NAME"] + "Metrics"
    )
    ddb_table_f = boto3.resource("dynamodb", region_name="us-east-1").Table(
        os.environ["TABLE_NAME"] + "MetricsFail"
    )

    response = requests.get(url)
    if not response.ok:
        return

    config = json.loads(response.text)
    config["config_idx"] = int(config["config_idx"])

    X, y = DATASETS[config["dataset"]]
    n_classes = int(config["n_classes"])
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
        for metric in ["accuracy", "f1", "auc"]:
            for stat in ["mean", "std"]:
                config["metrics"][f"{model_name}_{metric}_{stat}"] = []

    try:
        for j, n_features in enumerate(n_features_to_keep, 1):
            logger.info(
                f"Config: {config['config_idx']} | Features: {n_features} ({j}/{len(n_features_to_keep)}) "
                f"| Dataset: {config['dataset']}"
            )

            X_ = X[:, feature_ranks[:n_features]]
            metrics = cv_scores_all_models(X=X_, y=y, n_classes=n_classes)

            config["metrics"]["feature_ranks"].append(
                ",".join(map(str, feature_ranks[:n_features]))
            )
            config["metrics"]["n_features_used"].append(int(n_features))
            for key, value in metrics.items():
                config["metrics"][key].append(value)

        # Write to DynamoDB
        item = json.loads(json.dumps(config), parse_float=Decimal)
        ddb_table_s.put_item(Item=item)

    except Exception as e:
        message = str(e)
        logger.error(
            f"Config: {config['config_idx']} | Dataset: {config['dataset']} | Error: {message}"
        )

        item = {
            "config_idx": config["config_idx"],
            "method": config["method"],
            "hyperparameters": config.get("hyperparameters", {}),
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

    # Load datasets
    n_files = len(files)
    for j, f in enumerate(files, 1):
        dataset = f.replace("clf_", "").replace(".snappy.parquet", "")
        logger.info(f"Loading dataset {dataset} ({j}/{n_files})")
        X = pd.read_parquet(os.path.join(data_dir, f))
        y = X.pop("y").astype(int).values
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
