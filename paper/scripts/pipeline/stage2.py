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
    EVALUATION_K_VALUES,
    HIGH_P_EVALUATION_EXTRA_K_FRACTIONS,
    HIGH_P_EVALUATION_EXTRA_K_VALUES,
    HIGH_P_EVALUATION_P_THRESHOLD,
    N_SPLITS,
    REG_DOWNSTREAM_MODELS,
)
from paper.scripts.pipeline.types import ExperimentConfig, Result
from paper.scripts.utils.env import get_git_sha, get_library_versions, utc_now_iso

config = load_config()


# =============================================================================
# Resource allocation
# =============================================================================


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


def resolve_evaluation_k_values(
    n_features: int,
    *,
    base_k_values: list[int] | None = None,
    extra_k_values: list[int] | None = None,
    extra_k_fractions: list[float] | None = None,
    include_endpoint: bool = True,
) -> list[int]:
    """Build the feature-budget schedule for Stage 2 evaluation."""
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    resolved_base_k_values = base_k_values or EVALUATION_K_VALUES
    high_p_floor = max(resolved_base_k_values)
    k_values = set(resolved_base_k_values)
    for k in extra_k_values or []:
        if k > high_p_floor:
            k_values.add(k)

    for frac in extra_k_fractions or []:
        frac_k = max(1, int(np.ceil(frac * n_features)))
        if frac_k > high_p_floor:
            k_values.add(frac_k)

    if include_endpoint:
        k_values.add(n_features)

    return sorted(k for k in k_values if k <= n_features)


def get_requested_evaluation_k_values(n_features: int) -> list[int]:
    """Resolve the Stage 2 k schedule from the benchmark defaults."""
    extra_k_values: list[int] = []
    extra_k_fractions: list[float] = []
    if n_features > HIGH_P_EVALUATION_P_THRESHOLD:
        extra_k_values = list(HIGH_P_EVALUATION_EXTRA_K_VALUES)
        extra_k_fractions = list(HIGH_P_EVALUATION_EXTRA_K_FRACTIONS)
    return resolve_evaluation_k_values(
        n_features,
        extra_k_values=extra_k_values,
        extra_k_fractions=extra_k_fractions,
    )


def metrics_cover_requested_k_values(
    metrics_df: pd.DataFrame, required_k_values: list[int]
) -> bool:
    """Return True when an existing metrics artifact already covers the active k schedule."""
    if "k" not in metrics_df.columns:
        return False
    observed = set(pd.to_numeric(metrics_df["k"], errors="coerce").dropna().astype(int))
    return set(required_k_values).issubset(observed)


def infer_n_features_from_rankings(rankings_df: pd.DataFrame) -> int:
    """Infer the ranking length from a Stage 1 rankings artifact."""
    if rankings_df.empty:
        raise ValueError("rankings_df is empty; cannot infer n_features")
    ranking = rankings_df.iloc[0]["feature_ranking"]
    return int(len(ranking))


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ranking: np.ndarray,
    task: str,
    random_state: int,
    k_values: list[int] | None = None,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate downstream models for a single fold.

    Tests models at various feature subset sizes (k values).
    """
    n_features = X_train.shape[1]
    if k_values is None:
        k_values = get_requested_evaluation_k_values(n_features)

    downstream_models = CLF_DOWNSTREAM_MODELS if task == "classification" else REG_DOWNSTREAM_MODELS
    model_factory = get_clf_models if task == "classification" else get_reg_models

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

            if task == "classification":
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
    task = cfg.task
    seed = cfg.seed
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    k_values = get_requested_evaluation_k_values(n_features)

    # Reconstruct deterministic CV splits
    cv = get_cv_splitter(task, N_SPLITS, seed)
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
            task,
            rs,
            k_values,
            n_jobs,
        )

        for res in fold_results:
            res.update(
                {
                    "dataset": cfg.dataset,
                    "method_id": cfg.method.label,
                    "method": cfg.method.label,
                    "method_base": cfg.method.name,
                    "task": task,
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


def _run_evaluation(cfg: ExperimentConfig, store: Store) -> Result:
    """Run evaluation for a single configuration."""
    task = cfg.task
    hostname = socket.gethostname()
    git_sha = get_git_sha()

    # Check if rankings exist
    if not store.exists("rankings", cfg):
        return Result(
            config=cfg,
            status="no_rankings",
            hostname=hostname,
        )

    try:
        # Load rankings first so the active k schedule can be checked without
        # assuming the current metrics artifact is complete.
        rankings_df = store.load("rankings", cfg)
        ranking_n_features = infer_n_features_from_rankings(rankings_df)
        requested_k_values = get_requested_evaluation_k_values(ranking_n_features)

        if store.exists("metrics", cfg):
            existing_metrics = store.load("metrics", cfg)
            if metrics_cover_requested_k_values(existing_metrics, requested_k_values):
                return Result(
                    config=cfg,
                    status="skipped",
                    hostname=hostname,
                )

        X, y = load_dataset(cfg.dataset, task)
        n_jobs = -1
        # Get dataset metadata
        dataset_meta = get_dataset_metadata(cfg.dataset, task)

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
