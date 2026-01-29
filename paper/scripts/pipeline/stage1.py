"""Stage 1: Feature selection using various methods.

This module implements the feature selection stage of the experiment pipeline.
It computes feature rankings using different methods and saves them to S3.
"""

from __future__ import annotations

import socket
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import ray
from sklearn.preprocessing import StandardScaler

from paper.scripts.adapters.data import (
    get_cv_splitter,
    get_dataset_metadata,
    load_dataset,
)
from paper.scripts.adapters.store import Store
from paper.scripts.config import load_config
from paper.scripts.config.constants import N_SPLITS
from paper.scripts.pipeline.methods import THREADED_METHODS
from paper.scripts.pipeline.types import ExperimentConfig, Result
from paper.scripts.utils.env import get_git_sha, get_library_versions, utc_now_iso

config = load_config()


# =============================================================================
# Resource allocation
# =============================================================================


def selection_num_cpus(
    method: str,
    *,
    n_samples: int | None = None,
    n_features: int | None = None,
) -> int:
    """Return Ray CPU reservation for a selection task."""
    exp = config.experiment

    override = exp.selection_cpus_overrides.get(method)
    if override is not None:
        return int(override)

    if method == "cif":
        if n_samples is not None and n_features is not None:
            complexity = n_samples * n_features
            if complexity >= exp.selection_cif_large_threshold:
                return exp.selection_cpus_cif_large
        return exp.selection_cpus_cif

    if method in THREADED_METHODS:
        return exp.selection_cpus_threaded

    return exp.selection_cpus_default


def selection_memory_bytes(method: str) -> int:
    """Return Ray memory reservation (in bytes) for a selection task."""
    exp = config.experiment

    override = exp.selection_memory_gb_overrides.get(method)
    if override is not None:
        return int(override * 1024 * 1024 * 1024)

    return int(exp.selection_memory_gb_default * 1024 * 1024 * 1024)


# =============================================================================
# Selection methods
# =============================================================================


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_encoded, classes) with labels mapped to 0..K-1."""
    y_arr = np.asarray(y).ravel()
    classes, y_encoded = np.unique(y_arr, return_inverse=True)
    return y_encoded.astype(np.int64), classes


def filter_selector(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: str,
    random_state: int,
) -> np.ndarray:
    """Compute feature ranking using filter methods."""
    from citrees._selector import ClassifierSelectors, RegressorSelectors

    n_features = X.shape[1]
    selectors = ClassifierSelectors if task == "classification" else RegressorSelectors
    selector_fn = selectors[method]
    scores = np.zeros(n_features)
    rng = np.random.default_rng(random_state)

    if task == "classification":
        y_enc, classes = _encode_labels(y)
        n_classes = len(classes)
        for j in range(n_features):
            scores[j] = selector_fn(X[:, j], y_enc, n_classes, random_state=rng.integers(0, 2**31))
    else:
        use_abs = method == "pc"
        for j in range(n_features):
            score = selector_fn(X[:, j], y, standardize=True, random_state=rng.integers(0, 2**31))
            scores[j] = abs(score) if use_abs else score

    return np.argsort(scores)[::-1]


def _resolve_n_resamples(n_resamples: int | str, alpha: float) -> int:
    """Resolve n_resamples string values to integers.

    Uses the same formulas as citrees._tree:
    - "minimum": ceil(1 / alpha)
    - "maximum": ceil(1 / (4 * alpha^2))
    - "auto": max(lower_limit, ceil(z^2 * (1-alpha) / alpha))

    Raises
    ------
    ValueError
        If n_resamples is a string other than "minimum", "maximum", or "auto".
    """
    from math import ceil

    from scipy.stats import norm

    if isinstance(n_resamples, int):
        return n_resamples

    lower_limit = ceil(1 / alpha)
    if n_resamples == "minimum":
        return lower_limit
    elif n_resamples == "maximum":
        return ceil(1 / (4 * alpha * alpha))
    elif n_resamples == "auto":
        z = norm.ppf(1 - alpha)
        upper_limit = ceil(z * z * (1 - alpha) / alpha)
        return max(lower_limit, upper_limit)
    else:
        raise ValueError(
            f"n_resamples must be int or one of 'minimum', 'maximum', 'auto', got: {n_resamples!r}"
        )


def permutation_selector(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: str,
    random_state: int,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using permutation tests.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    method : str
        Method name (e.g., "ptest_mc", "ptest_pc").
    task : str
        "classification" or "regression".
    random_state : int
        Random seed.
    params : dict, optional
        Parameters for the permutation test:
        - alpha: significance level (default 0.05)
        - n_resamples: number of permutations (default 1000, or "minimum"/"maximum"/"auto")
        - early_stopping: "adaptive", "simple", or None (default None)
    """
    from citrees._selector import (
        ClassifierSelectors,
        ClassifierSelectorTests,
        RegressorSelectors,
        RegressorSelectorTests,
    )

    params = params or {}
    alpha = params.get("alpha", 0.05)
    n_resamples_raw = params.get("n_resamples", 1000)
    n_resamples = _resolve_n_resamples(n_resamples_raw, alpha)
    early_stopping = params.get("early_stopping")

    base_method = method.replace("ptest_", "")
    n_features = X.shape[1]
    selectors = ClassifierSelectors if task == "classification" else RegressorSelectors
    selector_tests = ClassifierSelectorTests if task == "classification" else RegressorSelectorTests
    selector_fn = selectors[base_method]
    test_fn = selector_tests[base_method]

    scores = np.zeros(n_features)
    pvalues = np.ones(n_features)
    rng = np.random.default_rng(random_state)

    if task == "classification":
        y_enc, classes = _encode_labels(y)
        n_classes = len(classes)
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            scores[j] = selector_fn(X[:, j], y_enc, n_classes, random_state=rs)
            pvalues[j] = test_fn(
                x=X[:, j],
                y=y_enc,
                n_classes=n_classes,
                alpha=alpha,
                n_resamples=n_resamples,
                early_stopping=early_stopping,
                random_state=rs,
            )
    else:
        use_abs = base_method == "pc"
        for j in range(n_features):
            rs = rng.integers(0, 2**31)
            score = selector_fn(X[:, j], y, standardize=True, random_state=rs)
            scores[j] = abs(score) if use_abs else score
            pvalues[j] = test_fn(
                x=X[:, j],
                y=y,
                standardize=True,
                alpha=alpha,
                n_resamples=n_resamples,
                early_stopping=early_stopping,
                random_state=rs,
            )

    return np.lexsort((-scores, pvalues))


def embedding_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str,
    task: str,
    random_state: int,
    params: dict[str, Any] | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute feature ranking using tree-based embedding methods."""
    from paper.scripts.pipeline.selectors import get_embedding_model

    params = params or {}
    model = get_embedding_model(method, task, random_state, n_jobs=n_jobs, params=params)
    model.fit(X_train, y_train)
    return np.argsort(model.feature_importances_)[::-1]


def wrapper_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    task: str,
    random_state: int,
    n_jobs: int = 1,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using wrapper methods.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target array.
    method : str
        Method name: "boruta", "pi", "shap", "cpi", "mrmr", "rfe".
    task : str
        "classification" or "regression".
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.
    params : dict, optional
        Method-specific parameters. See individual selector functions.
    """
    from paper.scripts.pipeline.selectors import (
        boruta_selector,
        cpi_selector,
        mrmr_selector,
        pi_selector,
        rfe_selector,
        shap_selector,
    )

    params = params or {}

    if method == "boruta":
        return boruta_selector(X_train, y_train, task, random_state, n_jobs=n_jobs, params=params)
    if method == "pi":
        return pi_selector(X_train, y_train, task, random_state, n_jobs=n_jobs, params=params)
    if method == "shap":
        return shap_selector(X_train, y_train, task, random_state, n_jobs=n_jobs, params=params)
    if method == "cpi":
        return cpi_selector(X_train, y_train, task, random_state, n_jobs=n_jobs, params=params)
    if method == "mrmr":
        return mrmr_selector(X_train, y_train, task)
    if method == "rfe":
        return rfe_selector(X_train, y_train, task, random_state, n_jobs=n_jobs)

    raise ValueError(f"Unknown wrapper method: {method}")


def run_selection(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: str,
    seed: int,
    params: dict[str, Any] | None = None,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run feature selection across CV folds.

    Returns a list of result dicts, one per fold, containing:
    - fold_idx: fold index
    - feature_ranking: list of feature indices in ranked order
    """
    params = params or {}
    cv = get_cv_splitter(task, N_SPLITS, seed)

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit preprocessing only on training fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        rs = seed * 1000 + fold_idx

        # Select appropriate method
        if method in ["mc", "mi", "rdc", "pc", "dc"]:
            ranking = filter_selector(X_train, y_train, method, task, rs)
        elif method.startswith("ptest_"):
            ranking = permutation_selector(X_train, y_train, method, task, rs, params=params)
        elif method in ["rf", "et", "xgb", "lgbm", "cat", "cit", "cif"]:
            ranking = embedding_selector(
                X_train,
                y_train,
                X_test,
                y_test,
                method,
                task,
                rs,
                params=params,
                n_jobs=n_jobs,
            )
        elif method in ["boruta", "pi", "shap", "cpi", "mrmr", "rfe"]:
            ranking = wrapper_selector(
                X_train, y_train, method, task, rs, n_jobs=n_jobs, params=params
            )
        elif method == "r_ctree":
            from paper.scripts.pipeline.r_methods import r_ctree_ranking

            ranking = r_ctree_ranking(X_train, y_train, task=task, **params)
        elif method == "r_cforest":
            from paper.scripts.pipeline.r_methods import r_cforest_ranking

            ranking = r_cforest_ranking(X_train, y_train, task=task, **params)
        else:
            raise ValueError(f"Unknown method: {method}")

        if len(ranking) == 0:
            raise ValueError(f"Empty ranking returned by {method} for fold {fold_idx}")

        results.append(
            {
                "fold_idx": fold_idx,
                "feature_ranking": ranking.tolist(),
            }
        )

    return results


# =============================================================================
# Ray task
# =============================================================================


@ray.remote(resources={"selection": 1})
def run_selection_task(cfg: ExperimentConfig, store: Store) -> Result:
    """Ray task for feature selection.

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
    return _run_selection(cfg, store)


def _run_selection(cfg: ExperimentConfig, store: Store) -> Result:
    """Synchronous feature selection for a single configuration.

    Can be called directly for local execution or wrapped in a Ray task.
    """
    method = cfg.method.name
    params = cfg.method.params_dict
    dataset = cfg.dataset
    seed = cfg.seed
    task = cfg.task

    hostname = socket.gethostname()
    git_sha = get_git_sha()

    # Check if output exists
    if store.exists("rankings", cfg):
        return Result(
            config=cfg,
            status="skipped",
            hostname=hostname,
        )

    try:
        # Load dataset
        X, y = load_dataset(dataset, task)
        n_samples, n_features = int(X.shape[0]), int(X.shape[1])
        n_jobs = selection_num_cpus(method, n_samples=n_samples, n_features=n_features)

        # Get dataset metadata
        dataset_meta = get_dataset_metadata(dataset, task)

        # Run selection
        created_at_utc = utc_now_iso()
        tic = time.perf_counter()
        fold_results = run_selection(X, y, method, task, seed, params=params, n_jobs=n_jobs)
        elapsed = time.perf_counter() - tic

        # Enrich results with metadata
        for row in fold_results:
            row.update(
                {
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                    "method_id": cfg.method.label,
                    "method": cfg.method.label,
                    "method_base": method,
                    "artifact_version": 2,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "selection_cpus": n_jobs,
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
        df = pd.DataFrame(fold_results)
        s3_path = store.save("rankings", cfg, df)

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
