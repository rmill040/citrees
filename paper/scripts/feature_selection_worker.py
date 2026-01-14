"""
Stage 1: Feature Selection Worker

Distributed worker that runs feature selection and uploads rankings to S3.
Tracks completion via DynamoDB for resume capability.

Usage:
    URL=http://localhost:8000 S3_BUCKET=citrees-results TABLE_NAME=ClfFeatureSelection \
        AWS_PROFILE=personal uv run python paper/scripts/feature_selection_worker.py

Local testing:
    LOCAL_TEST=1 S3_BUCKET=citrees-results AWS_PROFILE=personal \
        uv run python paper/scripts/feature_selection_worker.py
"""

from __future__ import annotations

import io
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import requests
from loguru import logger
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Configuration
# ============================================================================

S3_BUCKET = os.environ.get("S3_BUCKET", "citrees-results")
TABLE_NAME = os.environ.get("TABLE_NAME", "ClfFeatureSelection")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
N_SPLITS = 5
RANDOM_STATE = 42

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
# Feature Selection Methods
# ============================================================================

def filter_selector(X: np.ndarray, y: np.ndarray, method: str, task_type: str,
                    random_state: int) -> np.ndarray:
    """Run filter-based feature selection, return feature ranking (indices)."""
    from citrees._selector import (
        ClassifierSelectors,
        RegressorSelectors,
    )

    n_features = X.shape[1]
    selectors = ClassifierSelectors if task_type == "classification" else RegressorSelectors

    # Get selector function
    selector_fn = selectors[method]

    # Compute scores for each feature
    scores = np.zeros(n_features)
    rng = np.random.default_rng(random_state)

    if task_type == "classification":
        n_classes = len(np.unique(y))
        for j in range(n_features):
            scores[j] = selector_fn(X[:, j], y, n_classes, random_state=rng.integers(0, 2**31))
    else:
        for j in range(n_features):
            scores[j] = selector_fn(X[:, j], y, standardize=True, random_state=rng.integers(0, 2**31))

    # Rank by score (descending - higher is better)
    ranking = np.argsort(scores)[::-1]
    return ranking


def permutation_selector(X: np.ndarray, y: np.ndarray, method: str, task_type: str,
                         random_state: int, alpha: float = 0.05,
                         n_resamples: int = 1000) -> np.ndarray:
    """Run permutation test feature selection, return feature ranking."""
    from citrees._selector import (
        ClassifierSelectors,
        ClassifierSelectorTests,
        RegressorSelectors,
        RegressorSelectorTests,
    )

    # Map ptest_X to X
    base_method = method.replace("ptest_", "")

    n_features = X.shape[1]
    selectors = ClassifierSelectors if task_type == "classification" else RegressorSelectors
    selector_tests = ClassifierSelectorTests if task_type == "classification" else RegressorSelectorTests

    selector_fn = selectors[base_method]
    test_fn = selector_tests[base_method]

    # Compute scores and p-values for each feature
    scores = np.zeros(n_features)
    pvalues = np.ones(n_features)
    rng = np.random.default_rng(random_state)

    if task_type == "classification":
        n_classes = len(np.unique(y))
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            scores[j] = selector_fn(X[:, j], y, n_classes, random_state=rs)
            pvalues[j] = test_fn(
                X[:, j], y, n_classes,
                alpha=alpha,
                n_resamples=n_resamples,
                early_stopping=None,
                random_state=rs,
            )
    else:
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            scores[j] = selector_fn(X[:, j], y, standardize=True, random_state=rs)
            pvalues[j] = test_fn(
                X[:, j], y,
                standardize=True,
                alpha=alpha,
                n_resamples=n_resamples,
                early_stopping=None,
                random_state=rs,
            )

    # Rank by p-value (ascending - lower is better), then by score (descending)
    # Use lexsort: primary key last, secondary key first
    ranking = np.lexsort((-scores, pvalues))
    return ranking


def embedding_selector(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                       y_test: np.ndarray, method: str, task_type: str,
                       random_state: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Run embedding-based feature selection, return ranking and model predictions."""
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
    )

    try:
        from xgboost import XGBClassifier, XGBRegressor
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        HAS_LGBM = True
    except ImportError:
        HAS_LGBM = False

    from citrees import (
        ConditionalInferenceTreeClassifier,
        ConditionalInferenceTreeRegressor,
        ConditionalInferenceForestClassifier,
        ConditionalInferenceForestRegressor,
    )

    # Model mapping
    if task_type == "classification":
        models = {
            "rf": lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
            "et": lambda: ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
            "cit": lambda: ConditionalInferenceTreeClassifier(random_state=random_state),
            "cif": lambda: ConditionalInferenceForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
        }
        if HAS_XGB:
            models["xgb"] = lambda: XGBClassifier(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=0)
        if HAS_LGBM:
            models["lgbm"] = lambda: LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=-1)
    else:
        models = {
            "rf": lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state),
            "et": lambda: ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=random_state),
            "cit": lambda: ConditionalInferenceTreeRegressor(random_state=random_state),
            "cif": lambda: ConditionalInferenceForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state),
        }
        if HAS_XGB:
            models["xgb"] = lambda: XGBRegressor(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=0)
        if HAS_LGBM:
            models["lgbm"] = lambda: LGBMRegressor(n_estimators=100, n_jobs=-1, random_state=random_state, verbosity=-1)

    # Fit model
    model = models[method]()
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_

    # Rank by importance (descending)
    ranking = np.argsort(importances)[::-1]

    # Get predictions for embedding metrics
    embedding_data = {}
    if task_type == "classification":
        embedding_data["train_preds"] = model.predict(X_train).tolist()
        embedding_data["test_preds"] = model.predict(X_test).tolist()
        if hasattr(model, "predict_proba"):
            embedding_data["train_proba"] = model.predict_proba(X_train).tolist()
            embedding_data["test_proba"] = model.predict_proba(X_test).tolist()
    else:
        embedding_data["train_preds"] = model.predict(X_train).tolist()
        embedding_data["test_preds"] = model.predict(X_test).tolist()

    return ranking, embedding_data


def boruta_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str,
                    random_state: int) -> np.ndarray:
    """Run Boruta feature selection, return feature ranking."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    try:
        from boruta import BorutaPy
    except ImportError:
        logger.error("Boruta not installed. Run: pip install boruta")
        raise

    if task_type == "classification":
        base_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
    else:
        base_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)

    boruta = BorutaPy(base_model, n_estimators="auto", random_state=random_state, verbose=0)
    boruta.fit(X_train, y_train)

    # Boruta gives ranking_ where 1 is best
    # Convert to indices sorted by ranking
    ranking = np.argsort(boruta.ranking_)
    return ranking


def permutation_importance_selector(X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray,
                                    task_type: str, random_state: int) -> np.ndarray:
    """Run permutation importance feature selection, return feature ranking."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.inspection import permutation_importance

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)
        scoring = "r2"

    model.fit(X_train, y_train)

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )

    # Rank by importance (descending)
    ranking = np.argsort(result.importances_mean)[::-1]
    return ranking


def shap_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str,
                  random_state: int) -> np.ndarray:
    """Run SHAP feature selection, return feature ranking."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap")
        raise

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=random_state)

    model.fit(X_train, y_train)

    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)

    # Sample if dataset is large
    if X_train.shape[0] > 1000:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_train.shape[0], 1000, replace=False)
        X_sample = X_train[idx]
    else:
        X_sample = X_train

    shap_values = explainer.shap_values(X_sample)

    # For classification, shap_values is a list per class
    if isinstance(shap_values, list):
        # Take mean absolute across classes
        importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        importances = np.abs(shap_values).mean(axis=0)

    # Rank by importance (descending)
    ranking = np.argsort(importances)[::-1]
    return ranking


def mrmr_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str,
                  random_state: int) -> np.ndarray:
    """Run mRMR feature selection, return feature ranking."""
    try:
        from mrmr import mrmr_classif, mrmr_regression
    except ImportError:
        logger.error("mrmr-selection not installed. Run: pip install mrmr-selection")
        raise

    # mRMR needs a DataFrame
    df = pd.DataFrame(X_train)
    y_series = pd.Series(y_train)

    n_features = X_train.shape[1]

    if task_type == "classification":
        selected = mrmr_classif(df, y_series, K=n_features, show_progress=False)
    else:
        selected = mrmr_regression(df, y_series, K=n_features, show_progress=False)

    # mRMR returns column names (integers since DataFrame has int columns)
    ranking = np.array(selected)
    return ranking


# ============================================================================
# Main Feature Selection Runner
# ============================================================================

def run_feature_selection(X: np.ndarray, y: np.ndarray, method: str,
                          task_type: str, seed: int) -> list[dict[str, Any]]:
    """Run feature selection with cross-validation, return per-fold results."""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set up CV
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        tic = time.perf_counter()
        embedding_data = None

        # Run appropriate method
        if method in ["mc", "mi", "rdc", "pc", "dc"]:
            ranking = filter_selector(X_train, y_train, method, task_type, seed + fold_idx)

        elif method.startswith("ptest_"):
            ranking = permutation_selector(X_train, y_train, method, task_type, seed + fold_idx)

        elif method in ["rf", "et", "xgb", "lgbm", "cit", "cif"]:
            ranking, embedding_data = embedding_selector(
                X_train, y_train, X_test, y_test, method, task_type, seed + fold_idx
            )

        elif method == "boruta":
            ranking = boruta_selector(X_train, y_train, task_type, seed + fold_idx)

        elif method == "pi":
            ranking = permutation_importance_selector(
                X_train, y_train, X_test, y_test, task_type, seed + fold_idx
            )

        elif method == "shap":
            ranking = shap_selector(X_train, y_train, task_type, seed + fold_idx)

        elif method == "mrmr":
            ranking = mrmr_selector(X_train, y_train, task_type, seed + fold_idx)

        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.perf_counter() - tic

        fold_result = {
            "fold_idx": fold_idx,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "feature_ranking": ranking.tolist(),
            "selection_time_seconds": elapsed,
        }

        # Add embedding data if present
        if embedding_data:
            fold_result["embedding_train_preds"] = embedding_data.get("train_preds")
            fold_result["embedding_test_preds"] = embedding_data.get("test_preds")
            fold_result["embedding_train_proba"] = embedding_data.get("train_proba")
            fold_result["embedding_test_proba"] = embedding_data.get("test_proba")

        results.append(fold_result)
        logger.debug(f"  Fold {fold_idx}: {elapsed:.2f}s")

    return results


# ============================================================================
# S3 and DynamoDB Helpers
# ============================================================================

def upload_to_s3(data: list[dict], s3_path: str) -> None:
    """Upload results to S3 as parquet."""
    df = pd.DataFrame(data)

    # Convert list columns to proper format for parquet
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    # Parse s3_path: s3://bucket/key
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
        "s3_path": config["s3_path"],
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "worker_id": socket.gethostname(),
    })


def mark_completed(config: dict[str, Any], elapsed: float) -> None:
    """Mark config as completed in DynamoDB."""
    table.update_item(
        Key={"config_idx": config["config_idx"]},
        UpdateExpression="SET #s = :s, completed_at = :t, elapsed_seconds = :e",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "completed",
            ":t": datetime.now(timezone.utc).isoformat(),
            ":e": str(elapsed),
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
            ":e": error[:1000],  # Truncate long errors
        },
    )


# ============================================================================
# Worker Loop
# ============================================================================

def run_worker(server_url: str) -> None:
    """Main worker loop - pull configs and process."""
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
        if s3_file_exists(config["s3_path"]):
            logger.info(f"  Already exists in S3, skipping")
            continue

        # Mark as pending
        mark_pending(config)

        try:
            # Load dataset if not cached
            if config["dataset"] not in DATASETS:
                load_datasets(config["task_type"])

            X, y = DATASETS[config["dataset"]]

            # Run feature selection
            tic = time.perf_counter()
            results = run_feature_selection(
                X, y,
                method=config["method"],
                task_type=config["task_type"],
                seed=config["seed"],
            )
            elapsed = time.perf_counter() - tic

            # Upload to S3
            upload_to_s3(results, config["s3_path"])

            # Mark completed
            mark_completed(config, elapsed)
            logger.info(f"  Completed in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            mark_failed(config, str(e))


# ============================================================================
# Local Test
# ============================================================================

def test_local() -> None:
    """Test feature selection locally without server."""
    from sklearn.datasets import make_classification, make_regression

    print("=" * 70)
    print("LOCAL FEATURE SELECTION TEST")
    print("=" * 70)

    # Test classification
    print("\n=== CLASSIFICATION ===")
    X_clf, y_clf = make_classification(
        n_samples=200, n_features=20, n_informative=5,
        n_redundant=5, random_state=42
    )

    for method in ["mc", "rf", "mrmr"]:
        print(f"\n--- {method} ---")
        results = run_feature_selection(X_clf, y_clf, method, "classification", seed=42)
        print(f"  Folds: {len(results)}")
        print(f"  Ranking (first 5): {results[0]['feature_ranking'][:5]}")
        if "embedding_test_preds" in results[0] and results[0]["embedding_test_preds"]:
            print(f"  Has embedding predictions: Yes")

    # Test regression
    print("\n=== REGRESSION ===")
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=20, n_informative=5,
        random_state=42
    )

    for method in ["pc", "rf"]:
        print(f"\n--- {method} ---")
        results = run_feature_selection(X_reg, y_reg, method, "regression", seed=42)
        print(f"  Folds: {len(results)}")
        print(f"  Ranking (first 5): {results[0]['feature_ranking'][:5]}")

    # Test S3 upload (if bucket exists)
    print("\n=== S3 UPLOAD TEST ===")
    test_s3_path = f"s3://{S3_BUCKET}/test/test_rankings.parquet"
    try:
        upload_to_s3(results, test_s3_path)
        print(f"  Uploaded to {test_s3_path}")

        # Verify
        if s3_file_exists(test_s3_path):
            print("  Verified: file exists in S3")

            # Clean up
            parts = test_s3_path.replace("s3://", "").split("/", 1)
            s3.delete_object(Bucket=parts[0], Key=parts[1])
            print("  Cleaned up test file")
    except Exception as e:
        print(f"  S3 test failed (may need AWS credentials): {e}")

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
