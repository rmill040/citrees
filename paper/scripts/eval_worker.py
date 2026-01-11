"""
Stage 2: Downstream Evaluation Worker

Distributed worker that evaluates downstream models (LR, SVM, kNN) at ALL k values.
Downloads rankings from S3, uploads metrics to S3, tracks via DynamoDB.

Usage:
    URL=http://localhost:8000 S3_BUCKET=citrees-results TABLE_NAME=ClfDownstreamEval \
        AWS_PROFILE=personal uv run python paper/scripts/downstream_eval_worker.py

Local testing:
    LOCAL_TEST=1 S3_BUCKET=citrees-results AWS_PROFILE=personal \
        uv run python paper/scripts/downstream_eval_worker.py
"""

from __future__ import annotations

import io
import os
import socket
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from loguru import logger
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

# Suppress convergence warnings for LR/SVM
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================================
# Configuration
# ============================================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "citrees-results")
TABLE_NAME = os.environ.get("TABLE_NAME", "ClfDownstreamEval")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
N_JOBS = int(os.environ.get("N_JOBS", -1))

# Embedding methods that have their own predictions stored
EMBEDDING_METHODS = {"rf", "et", "xgb", "lgbm", "cit", "cif"}

# ============================================================================
# AWS Clients
# ============================================================================

s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)

# ============================================================================
# Dataset Cache
# ============================================================================

DATASETS: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_datasets(task_type: str) -> None:
    """Load all datasets for the given task type."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    prefix = "clf_" if task_type == "classification" else "reg_"

    for f in data_dir.glob(f"{prefix}*.parquet"):
        name = f.stem.replace(prefix, "").replace(".snappy", "")
        logger.info(f"Loading {name}")
        df = pd.read_parquet(f)
        y = df.pop("y").values
        if task_type == "classification":
            y = y.astype(np.int64)
        else:
            y = y.astype(np.float64)
        X = df.values.astype(np.float64)
        DATASETS[name] = (X, y)


# ============================================================================
# Downstream Models
# ============================================================================

def get_clf_models() -> dict[str, Any]:
    """Get classification downstream models."""
    return {
        "lr": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "svm": SVC(class_weight="balanced", probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }


def get_reg_models() -> dict[str, Any]:
    """Get regression downstream models."""
    return {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "svr": SVR(),
        "knn": KNeighborsRegressor(n_neighbors=5, weights="distance"),
    }


def evaluate_clf_at_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_indices: list[int],
    fold_idx: int,
    k: int,
) -> dict[str, Any]:
    """Evaluate classification models at k features."""
    X_train_k = X_train[:, feature_indices]
    X_test_k = X_test[:, feature_indices]

    result = {"fold_idx": fold_idx, "n_features": k}
    n_classes = len(np.unique(y_train))

    for name, model in get_clf_models().items():
        try:
            model.fit(X_train_k, y_train)
            y_pred = model.predict(X_test_k)

            result[f"{name}_acc"] = accuracy_score(y_test, y_pred)
            result[f"{name}_f1"] = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # AUC for binary or multi-class
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_k)
                if n_classes == 2:
                    result[f"{name}_roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                    result[f"{name}_pr_auc"] = average_precision_score(y_test, y_proba[:, 1])
                else:
                    try:
                        result[f"{name}_roc_auc"] = roc_auc_score(
                            y_test, y_proba, multi_class="ovr", average="macro"
                        )
                        # PR-AUC for multi-class (one-vs-rest)
                        from sklearn.preprocessing import label_binarize
                        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
                        result[f"{name}_pr_auc"] = average_precision_score(
                            y_test_bin, y_proba, average="macro"
                        )
                    except ValueError:
                        result[f"{name}_roc_auc"] = np.nan
                        result[f"{name}_pr_auc"] = np.nan
            else:
                result[f"{name}_roc_auc"] = np.nan
                result[f"{name}_pr_auc"] = np.nan

        except Exception as e:
            logger.warning(f"  {name} failed at k={k}: {e}")
            result[f"{name}_acc"] = np.nan
            result[f"{name}_f1"] = np.nan
            result[f"{name}_roc_auc"] = np.nan
            result[f"{name}_pr_auc"] = np.nan

    return result


def evaluate_reg_at_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_indices: list[int],
    fold_idx: int,
    k: int,
) -> dict[str, Any]:
    """Evaluate regression models at k features."""
    X_train_k = X_train[:, feature_indices]
    X_test_k = X_test[:, feature_indices]

    result = {"fold_idx": fold_idx, "n_features": k}

    for name, model in get_reg_models().items():
        try:
            model.fit(X_train_k, y_train)
            y_pred = model.predict(X_test_k)

            result[f"{name}_mse"] = mean_squared_error(y_test, y_pred)
            result[f"{name}_mae"] = mean_absolute_error(y_test, y_pred)
            result[f"{name}_r2"] = r2_score(y_test, y_pred)

        except Exception as e:
            logger.warning(f"  {name} failed at k={k}: {e}")
            result[f"{name}_mse"] = np.nan
            result[f"{name}_mae"] = np.nan
            result[f"{name}_r2"] = np.nan

    return result


# ============================================================================
# Embedding Metrics
# ============================================================================

def compute_embedding_metrics(
    y_test: np.ndarray,
    test_preds: list,
    task_type: str,
    n_classes: int = 2,
) -> dict[str, float]:
    """Compute metrics from stored embedding predictions."""
    y_pred = np.array(test_preds)

    if task_type == "classification":
        return {
            "embedding_acc": accuracy_score(y_test, y_pred),
            "embedding_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
    else:
        return {
            "embedding_mse": mean_squared_error(y_test, y_pred),
            "embedding_mae": mean_absolute_error(y_test, y_pred),
            "embedding_r2": r2_score(y_test, y_pred),
        }


# ============================================================================
# Main Evaluation
# ============================================================================

def download_rankings(s3_path: str) -> pd.DataFrame:
    """Download rankings parquet from S3."""
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]

    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


def evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    rankings_df: pd.DataFrame,
    method: str,
    task_type: str,
) -> list[dict[str, Any]]:
    """Evaluate downstream models for all folds and all k values."""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    all_results = []
    n_features = X.shape[1]
    n_classes = len(np.unique(y)) if task_type == "classification" else 0

    for _, row in rankings_df.iterrows():
        fold_idx = row["fold_idx"]
        train_indices = np.array(row["train_indices"])
        test_indices = np.array(row["test_indices"])
        feature_ranking = row["feature_ranking"]

        X_train = X_scaled[train_indices]
        X_test = X_scaled[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        # Evaluate at ALL k values in parallel
        if task_type == "classification":
            fold_results = Parallel(n_jobs=N_JOBS)(
                delayed(evaluate_clf_at_k)(
                    X_train, y_train, X_test, y_test,
                    feature_ranking[:k], fold_idx, k
                )
                for k in range(1, len(feature_ranking) + 1)
            )
        else:
            fold_results = Parallel(n_jobs=N_JOBS)(
                delayed(evaluate_reg_at_k)(
                    X_train, y_train, X_test, y_test,
                    feature_ranking[:k], fold_idx, k
                )
                for k in range(1, len(feature_ranking) + 1)
            )

        # Add embedding metrics if applicable
        if method in EMBEDDING_METHODS and "embedding_test_preds" in row and row["embedding_test_preds"]:
            embedding_metrics = compute_embedding_metrics(
                y_test, row["embedding_test_preds"], task_type, n_classes
            )
            # Add to all results for this fold
            for r in fold_results:
                r.update(embedding_metrics)

        all_results.extend(fold_results)

    return all_results


# ============================================================================
# S3 and DynamoDB Helpers
# ============================================================================

def upload_to_s3(data: list[dict], s3_path: str) -> None:
    """Upload results to S3 as parquet."""
    df = pd.DataFrame(data)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]

    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


def s3_file_exists(s3_path: str) -> bool:
    """Check if S3 file exists."""
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def mark_pending(config: dict[str, Any]) -> None:
    """Mark config as pending in DynamoDB."""
    table.put_item(Item={
        "config_idx": config["config_idx"],
        "dataset": config["dataset"],
        "method": config["method"],
        "seed": config["seed"],
        "task_type": config["task_type"],
        "rankings_s3_path": config["rankings_s3_path"],
        "metrics_s3_path": config["metrics_s3_path"],
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "worker_id": socket.gethostname(),
    })


def mark_completed(config: dict[str, Any], elapsed: float, n_evaluations: int) -> None:
    """Mark config as completed in DynamoDB."""
    table.update_item(
        Key={"config_idx": config["config_idx"]},
        UpdateExpression="SET #s = :s, completed_at = :t, elapsed_seconds = :e, n_evaluations = :n",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "completed",
            ":t": datetime.now(timezone.utc).isoformat(),
            ":e": str(elapsed),
            ":n": n_evaluations,
        },
    )


def mark_failed(config: dict[str, Any], error: str) -> None:
    """Mark config as failed in DynamoDB."""
    table.update_item(
        Key={"config_idx": config["config_idx"]},
        UpdateExpression="SET #s = :s, #e = :e",
        ExpressionAttributeNames={"#s": "status", "#e": "error"},
        ExpressionAttributeValues={
            ":s": "failed",
            ":e": error[:1000],
        },
    )


# ============================================================================
# Worker Loop
# ============================================================================

def run_worker(server_url: str) -> None:
    """Main worker loop."""
    hostname = socket.gethostname()
    server_url = server_url.rstrip("/")

    while True:
        # Get next config
        try:
            response = requests.get(f"{server_url}/", params={"host": hostname})
        except Exception as e:
            logger.error(f"Error contacting server: {e}")
            time.sleep(5)
            continue

        if response.status_code == 204:
            logger.info("No more configs. Exiting.")
            break

        config = response.json()
        logger.info(
            f"Config {config['config_idx']}: {config['method']} on {config['dataset']} seed={config['seed']}"
        )

        # Defensive check - skip if already done
        if s3_file_exists(config["metrics_s3_path"]):
            logger.info(f"  Already exists in S3, skipping")
            continue

        # Mark as pending
        mark_pending(config)

        try:
            # Load dataset if not cached
            if config["dataset"] not in DATASETS:
                load_datasets(config["task_type"])

            X, y = DATASETS[config["dataset"]]

            # Download rankings
            logger.info(f"  Downloading rankings from S3...")
            rankings_df = download_rankings(config["rankings_s3_path"])

            # Run evaluation
            logger.info(f"  Evaluating at all k values (1 to {X.shape[1]})...")
            tic = time.perf_counter()
            results = evaluate_config(
                X, y, rankings_df,
                method=config["method"],
                task_type=config["task_type"],
            )
            elapsed = time.perf_counter() - tic

            # Upload to S3
            logger.info(f"  Uploading {len(results)} evaluations to S3...")
            upload_to_s3(results, config["metrics_s3_path"])

            # Mark completed
            mark_completed(config, elapsed, len(results))
            logger.info(f"  Completed in {elapsed:.2f}s ({len(results)} evaluations)")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            mark_failed(config, str(e))


# ============================================================================
# Local Test
# ============================================================================

def test_local() -> None:
    """Test downstream evaluation locally."""
    from sklearn.datasets import make_classification, make_regression

    print("=" * 70)
    print("LOCAL DOWNSTREAM EVALUATION TEST")
    print("=" * 70)

    # Create synthetic rankings data (simulating Stage 1 output)
    print("\n=== Creating synthetic rankings ===")

    # Classification test
    X_clf, y_clf = make_classification(
        n_samples=200, n_features=20, n_informative=5,
        n_redundant=5, random_state=42
    )

    # Simulate rankings from Stage 1
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rankings_data = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_clf, y_clf)):
        # Random ranking for testing
        ranking = list(range(20))
        np.random.shuffle(ranking)
        rankings_data.append({
            "fold_idx": fold_idx,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "feature_ranking": ranking,
            "embedding_test_preds": None,
        })

    rankings_df = pd.DataFrame(rankings_data)

    print("\n=== Classification evaluation ===")
    results = evaluate_config(X_clf, y_clf, rankings_df, "mc", "classification")
    print(f"  Total evaluations: {len(results)}")
    print(f"  K values: 1 to {X_clf.shape[1]}")
    print(f"  Sample result at k=5: {[r for r in results if r['n_features'] == 5][0]}")

    # Regression test
    print("\n=== Regression evaluation ===")
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=20, n_informative=5,
        random_state=42
    )

    from sklearn.model_selection import KFold
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    rankings_data = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_reg, y_reg)):
        ranking = list(range(20))
        np.random.shuffle(ranking)
        rankings_data.append({
            "fold_idx": fold_idx,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "feature_ranking": ranking,
            "embedding_test_preds": None,
        })

    rankings_df = pd.DataFrame(rankings_data)

    results = evaluate_config(X_reg, y_reg, rankings_df, "pc", "regression")
    print(f"  Total evaluations: {len(results)}")
    print(f"  K values: 1 to {X_reg.shape[1]}")
    print(f"  Sample result at k=5: {[r for r in results if r['n_features'] == 5][0]}")

    # Test S3 upload
    print("\n=== S3 upload test ===")
    test_s3_path = f"s3://{S3_BUCKET}/test/test_metrics.parquet"
    try:
        upload_to_s3(results, test_s3_path)
        print(f"  Uploaded to {test_s3_path}")

        if s3_file_exists(test_s3_path):
            print("  Verified: file exists in S3")

            # Clean up
            parts = test_s3_path.replace("s3://", "").split("/", 1)
            s3.delete_object(Bucket=parts[0], Key=parts[1])
            print("  Cleaned up test file")
    except Exception as e:
        print(f"  S3 test failed: {e}")

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if os.environ.get("LOCAL_TEST"):
        test_local()
    else:
        server_url = os.environ["URL"]
        run_worker(server_url)
