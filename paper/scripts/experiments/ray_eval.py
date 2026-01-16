"""Ray-based distributed downstream evaluation."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from paper.scripts.utils.constants import CLF_DOWNSTREAM_MODELS, CLF_METHODS, REG_DOWNSTREAM_MODELS, REG_METHODS
from paper.scripts.utils.experiment_configs import config_label, expand_method_configs
from paper.scripts.infra.config import load_config

config = load_config()

# Data directory - works both locally and when rsynced to remote
DATA_DIR = (
    Path("/home/ubuntu/citrees/paper/data")
    if Path("/home/ubuntu/citrees").exists()
    else Path(__file__).resolve().parents[2] / "data"
)

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3")
    return _s3_client


def get_datasets(task_type: str) -> list[str]:
    prefix = "clf_" if task_type == "classification" else "reg_"
    datasets = []
    for f in DATA_DIR.glob(f"{prefix}*.parquet"):
        name = f.stem.replace(prefix, "").replace(".snappy", "")
        datasets.append(name)
    return sorted(datasets)


def load_dataset(name: str, task_type: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = "clf_" if task_type == "classification" else "reg_"
    # Try both .parquet and .snappy.parquet extensions
    path = DATA_DIR / f"{prefix}{name}.parquet"
    if not path.exists():
        path = DATA_DIR / f"{prefix}{name}.snappy.parquet"
    df = pd.read_parquet(path)
    y = df.pop("y").values
    if task_type == "classification":
        y = y.astype(np.int64)
    else:
        y = y.astype(np.float64)
    X = df.values.astype(np.float64)
    return X, y


def s3_file_exists(s3_path: str) -> bool:
    parts = s3_path.replace("s3://", "").split("/", 1)
    try:
        get_s3_client().head_object(Bucket=parts[0], Key=parts[1])
        return True
    except Exception:
        return False


def download_from_s3(s3_path: str) -> pd.DataFrame:
    parts = s3_path.replace("s3://", "").split("/", 1)
    response = get_s3_client().get_object(Bucket=parts[0], Key=parts[1])
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


def upload_to_s3(data: list[dict], s3_path: str) -> None:
    df = pd.DataFrame(data)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    parts = s3_path.replace("s3://", "").split("/", 1)
    get_s3_client().put_object(Bucket=parts[0], Key=parts[1], Body=buffer.getvalue())


def get_clf_models(random_state: int) -> dict[str, Any]:
    return {
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "svm": SVC(class_weight="balanced", probability=True, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }


def get_reg_models(random_state: int) -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "svr": SVR(),
        "knn": KNeighborsRegressor(n_neighbors=5, weights="distance"),
    }


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ranking: np.ndarray,
    task_type: str,
    random_state: int,
) -> list[dict[str, Any]]:
    n_features = X_train.shape[1]
    k_values = [5, 10, 25, 50, 100, n_features]
    k_values = sorted(set(k for k in k_values if k <= n_features))

    downstream_models = CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS
    model_factory = get_clf_models if task_type == "classification" else get_reg_models

    results = []
    for k in k_values:
        top_k_features = ranking[:k]
        X_train_k = X_train[:, top_k_features]
        X_test_k = X_test[:, top_k_features]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_k)
        X_test_scaled = scaler.transform(X_test_k)

        models = model_factory(random_state)
        for model_name in downstream_models:
            model = models[model_name]
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if task_type == "classification":
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, average="weighted"),
                }
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test_scaled)
                    if y_proba.shape[1] == 2:
                        metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        metrics["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            else:
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                }

            results.append({
                "k": k,
                "downstream_model": model_name,
                **metrics,
            })

    return results


def run_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    rankings_df: pd.DataFrame,
    task_type: str,
    seed: int,
) -> list[dict[str, Any]]:
    results = []

    for _, row in rankings_df.iterrows():
        fold_idx = row["fold_idx"]
        train_idx = np.array(row["train_indices"])
        test_idx = np.array(row["test_indices"])
        ranking = np.array(row["feature_ranking"])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rs = seed + fold_idx
        fold_results = evaluate_fold(X_train, y_train, X_test, y_test, ranking, task_type, rs)

        for res in fold_results:
            res["fold_idx"] = fold_idx
            results.append(res)

    return results


@ray.remote(resources={"evaluation": 1})
def process_config(config: dict[str, Any], dataset: str, seed: int, task_type: str) -> dict[str, Any]:
    method = config["method"]
    method_id = config_label(config)
    bucket = os.environ["S3_BUCKET"]
    rankings_path = f"s3://{bucket}/rankings/{task_type}/{dataset}/{method_id}_seed{seed}.parquet"
    metrics_path = f"s3://{bucket}/metrics/{task_type}/{dataset}/{method_id}_seed{seed}.parquet"

    if s3_file_exists(metrics_path):
        return {"status": "skipped", "method": method_id, "dataset": dataset, "seed": seed}

    if not s3_file_exists(rankings_path):
        return {"status": "no_rankings", "method": method_id, "dataset": dataset, "seed": seed}

    try:
        rankings_df = download_from_s3(rankings_path)
        X, y = load_dataset(dataset, task_type)
        tic = time.perf_counter()
        results = run_evaluation(X, y, rankings_df, task_type, seed)
        elapsed = time.perf_counter() - tic
        upload_to_s3(results, metrics_path)
        return {"status": "done", "method": method_id, "dataset": dataset, "seed": seed, "elapsed": elapsed}
    except Exception as e:
        return {"status": "failed", "method": method_id, "dataset": dataset, "seed": seed, "error": str(e)}


def main():
    ray.init(address="auto", ignore_reinit_error=True)

    task_type = config.experiment.type
    methods = CLF_METHODS if task_type == "classification" else REG_METHODS
    method_configs = expand_method_configs(methods)
    datasets = get_datasets(task_type)
    n_seeds = config.experiment.n_seeds

    configs = [(m, d, s) for m in method_configs for d in datasets for s in range(n_seeds)]
    logger.info(
        f"Submitting {len(configs)} configs ({len(method_configs)} methods × {len(datasets)} datasets × {n_seeds} seeds)"
    )

    futures = [process_config.remote(m, d, s, task_type) for m, d, s in configs]
    results = ray.get(futures)

    done = sum(1 for r in results if r["status"] == "done")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    no_rankings = sum(1 for r in results if r["status"] == "no_rankings")
    failed = sum(1 for r in results if r["status"] == "failed")

    logger.info(f"Done: {done}, Skipped: {skipped}, No rankings: {no_rankings}, Failed: {failed}")

    if failed > 0:
        for r in results:
            if r["status"] == "failed":
                logger.error(f"Failed: {r['method']}/{r['dataset']}/seed{r['seed']}: {r['error']}")


if __name__ == "__main__":
    main()
