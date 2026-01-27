"""Stage 2: Downstream model evaluation using feature rankings.

This module implements the evaluation stage of the experiment pipeline.
It uses feature rankings from Stage 1 to train downstream models and
compute performance metrics.
"""

from __future__ import annotations

import socket
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import ray
from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from paper.scripts.adapters.data import (
    get_cv_splitter,
    get_dataset_metadata,
    load_dataset,
)
from paper.scripts.adapters.store import Store
from paper.scripts.config import load_config
from paper.scripts.config.constants import (
    CLF_DOWNSTREAM_MODELS,
    N_SPLITS,
    REG_DOWNSTREAM_MODELS,
)
from paper.scripts.pipeline.types import ExperimentConfig, Result
from paper.scripts.utils.env import get_git_sha, get_library_versions, utc_now_iso

config = load_config()


# =============================================================================
# Resource allocation
# =============================================================================


def evaluation_num_cpus(task_type: str, *, downstream_models: list[str] | None = None) -> int:
    """Return Ray CPU reservation for an evaluation task."""
    exp = config.experiment

    models = downstream_models
    if models is None:
        models = CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS

    cpus = int(exp.evaluation_cpus_default)
    for model_name in models:
        override = exp.evaluation_cpus_overrides.get(model_name)
        if override is not None:
            cpus = max(cpus, int(override))

    return cpus


def evaluation_memory_bytes(task_type: str, *, downstream_models: list[str] | None = None) -> int:
    """Return Ray memory reservation (in bytes) for an evaluation task."""
    exp = config.experiment

    models = downstream_models
    if models is None:
        models = CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS

    memory_gb = exp.evaluation_memory_gb_default
    for model_name in models:
        override = exp.evaluation_memory_gb_overrides.get(model_name)
        if override is not None:
            memory_gb = max(memory_gb, override)

    return int(memory_gb * 1024 * 1024 * 1024)


# =============================================================================
# Downstream models
# =============================================================================


def get_clf_models(random_state: int, *, n_jobs: int = 1) -> dict[str, Any]:
    """Get classification model instances."""
    return {
        "lr": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "svm": SVC(
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=n_jobs,
        ),
    }


def get_reg_models(random_state: int, *, n_jobs: int = 1) -> dict[str, Any]:
    """Get regression model instances."""
    return {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "svr": SVR(),
        "knn": KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
            n_jobs=n_jobs,
        ),
    }


def compute_roc_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray | None,
) -> float:
    """Compute ROC AUC with proper handling of edge cases.

    Returns NaN for undefined cases (single-class, missing classes).
    """
    unique = np.unique(y_true)
    if unique.size < 2:
        logger.debug("ROC AUC undefined: y_true has only {} unique class(es)", unique.size)
        return float(np.nan)

    if y_proba.ndim == 1:
        if classes is None or len(classes) < 2:
            return float(np.nan)
        y_bin = (y_true == classes[1]).astype(int)
        return float(roc_auc_score(y_bin, y_proba))

    if y_proba.shape[1] == 2:
        if classes is None or len(classes) < 2:
            return float(np.nan)
        y_bin = (y_true == classes[1]).astype(int)
        return float(roc_auc_score(y_bin, y_proba[:, 1]))

    # Multiclass
    if classes is None or y_proba.shape[1] != len(classes):
        return float(np.nan)
    if np.unique(y_true).size < len(classes):
        return float(np.nan)

    return float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=classes)
    )


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ranking: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate downstream models for a single fold.

    Tests models at various feature subset sizes (k values).
    """
    n_features = X_train.shape[1]
    k_values = [5, 10, 25, 50, 100, n_features]
    k_values = sorted(set(k for k in k_values if k <= n_features))

    downstream_models = (
        CLF_DOWNSTREAM_MODELS if task_type == "classification" else REG_DOWNSTREAM_MODELS
    )
    model_factory = get_clf_models if task_type == "classification" else get_reg_models

    results = []
    for k in k_values:
        top_k_features = ranking[:k]
        X_train_k = X_train[:, top_k_features]
        X_test_k = X_test[:, top_k_features]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_k)
        X_test_scaled = scaler.transform(X_test_k)

        models = model_factory(random_state, n_jobs=n_jobs)
        for model_name in downstream_models:
            model = models[model_name]
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            if task_type == "classification":
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0.0),
                    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0.0),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                }
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test_scaled)
                    classes = getattr(model, "classes_", None)
                    roc_auc = compute_roc_auc(y_test, y_proba, classes)
                    metrics["roc_auc"] = roc_auc
                    metrics["auc"] = roc_auc
            else:
                metrics = {
                    "r2": r2_score(y_test, y_pred),
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                }

            results.append(
                {
                    "k": k,
                    "n_features_selected": k,
                    "downstream_model": model_name,
                    **metrics,
                }
            )

    return results


def run_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    rankings_df: pd.DataFrame,
    cfg: ExperimentConfig,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run evaluation across all folds.

    Uses the feature rankings from Stage 1 to evaluate downstream models.
    """
    task_type = cfg.task
    seed = cfg.seed
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])

    # Reconstruct deterministic CV splits
    cv = get_cv_splitter(task_type, N_SPLITS, seed)
    splits = list(cv.split(X, y))

    results = []
    for _, row in rankings_df.iterrows():
        fold_idx = int(row["fold_idx"])
        try:
            train_idx, test_idx = splits[fold_idx]
        except IndexError as e:
            raise ValueError(f"Invalid fold_idx={fold_idx}; expected 0..{len(splits) - 1}") from e

        ranking = np.array(row["feature_ranking"])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rs = seed * 1000 + fold_idx
        fold_results = evaluate_fold(
            X_train,
            y_train,
            X_test,
            y_test,
            ranking,
            task_type,
            rs,
            n_jobs,
        )

        for res in fold_results:
            res.update(
                {
                    "dataset": cfg.dataset,
                    "method_id": cfg.method.label,
                    "method": cfg.method.label,
                    "method_base": cfg.method.name,
                    "task_type": task_type,
                    "seed": seed,
                    "fold_idx": fold_idx,
                    "artifact_version": 2,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "evaluation_cpus": n_jobs,
                }
            )
            results.append(res)

    return results


# =============================================================================
# Ray task
# =============================================================================


@ray.remote(resources={"evaluation": 1})
def run_evaluation_task(cfg: ExperimentConfig, store: Store) -> Result:
    """Ray task for evaluation.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.
    store : Store
        Storage backend.

    Returns
    -------
    Result
        Execution result.
    """
    return _run_evaluation(cfg, store)


def _run_evaluation(cfg: ExperimentConfig, store: Store) -> Result:
    """Synchronous evaluation for a single configuration.

    Can be called directly for local execution or wrapped in a Ray task.
    """
    task_type = cfg.task
    hostname = socket.gethostname()
    git_sha = get_git_sha()

    # Check if output exists
    if store.exists("metrics", cfg):
        return Result(
            config=cfg,
            status="skipped",
            hostname=hostname,
        )

    # Check if rankings exist
    if not store.exists("rankings", cfg):
        return Result(
            config=cfg,
            status="no_rankings",
            hostname=hostname,
        )

    try:
        # Load rankings and dataset
        rankings_df = store.load("rankings", cfg)
        X, y = load_dataset(cfg.dataset, task_type)
        n_jobs = evaluation_num_cpus(task_type)

        # Get dataset metadata
        dataset_meta = get_dataset_metadata(cfg.dataset, task_type)

        # Run evaluation
        created_at_utc = utc_now_iso()
        tic = time.perf_counter()
        results = run_evaluation(X, y, rankings_df, cfg, n_jobs=n_jobs)
        elapsed = time.perf_counter() - tic

        # Enrich results with metadata
        for row in results:
            row.update(
                {
                    "elapsed_seconds": float(elapsed),
                    "created_at_utc": created_at_utc,
                    "git_sha": git_sha,
                    "library_versions": get_library_versions(),
                    "dataset_source": dataset_meta.get("dataset_source"),
                    "dataset_type": dataset_meta.get("dataset_type"),
                    "dataset_family": dataset_meta.get("dataset_family"),
                    "n_informative": dataset_meta.get("n_informative"),
                }
            )

        # Save to store
        df = pd.DataFrame(results)
        s3_path = store.save("metrics", cfg, df)

        return Result(
            config=cfg,
            status="done",
            elapsed_seconds=elapsed,
            data=df,
            s3_path=s3_path,
            hostname=hostname,
        )

    except Exception as e:
        return Result(
            config=cfg,
            status="failed",
            error=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc(),
            hostname=hostname,
        )
