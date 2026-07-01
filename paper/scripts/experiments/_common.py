"""Shared utilities for ablation experiment scripts.

Provides dataset factories, model builders, evaluation helpers, and output
infrastructure so individual experiment scripts stay DRY and consistent.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
    make_friedman1,
    make_regression,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)
from paper.scripts.utils.metrics import f1_at_k, precision_at_k, recall_at_k

# =============================================================================
# Paths & constants
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = _SCRIPT_DIR.parents[1] / "results" / "tables"
FIGURES_DIR = _SCRIPT_DIR.parents[1] / "results" / "figures"
DATA_DIR = Path(__file__).resolve().parents[3] / ".." / "data" / "ablation"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 1718
K = 10

# =============================================================================
# Run configuration
# =============================================================================

N_SEEDS = 5
N_ESTIMATORS = 100
N_JOBS = -1

# Type alias for synthetic dataset factory return
SyntheticDataset = tuple[np.ndarray, np.ndarray, list[int], str]
RealDataset = tuple[np.ndarray, np.ndarray, str]


# =============================================================================
# CIF / CIT parameter builders
# =============================================================================


def build_cif_params(task: str, **overrides: Any) -> dict[str, Any]:
    """Return default CIF params for the given task, with optional overrides."""
    params: dict[str, Any] = dict(
        n_estimators=N_ESTIMATORS,
        selector="mc" if task == "clf" else "pc",
        splitter="gini" if task == "clf" else "mse",
        alpha_selector=0.05,
        alpha_splitter=0.05,
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        threshold_method="histogram",
        max_thresholds=256,
        feature_scanning=True,
        feature_muting=True,
        bootstrap=True,
        n_jobs=N_JOBS,
        verbose=0,
    )
    params.update(overrides)
    if task == "reg":
        params.pop("sampling_method", None)
    return params


def build_cit_params(task: str, **overrides: Any) -> dict[str, Any]:
    """Return default CIT (single tree) params for the given task."""
    params: dict[str, Any] = dict(
        selector="mc" if task == "clf" else "pc",
        splitter="gini" if task == "clf" else "mse",
        alpha_selector=0.05,
        alpha_splitter=0.05,
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        threshold_method="histogram",
        max_thresholds=256,
        feature_scanning=True,
        feature_muting=True,
        verbose=0,
    )
    params.update(overrides)
    return params


def build_cif(task: str, seed: int, **overrides: Any) -> BaseEstimator:
    """Build and return a CIF classifier or regressor."""
    params = build_cif_params(task, **overrides)
    if task == "clf":
        return ConditionalInferenceForestClassifier(**params, random_state=seed)
    return ConditionalInferenceForestRegressor(**params, random_state=seed)


def build_cit(task: str, seed: int, **overrides: Any) -> BaseEstimator:
    """Build and return a CIT classifier or regressor."""
    params = build_cit_params(task, **overrides)
    if task == "clf":
        return ConditionalInferenceTreeClassifier(**params, random_state=seed)
    return ConditionalInferenceTreeRegressor(**params, random_state=seed)


def build_baseline(method: str, task: str, seed: int) -> BaseEstimator:
    """Build a baseline model (rf, et, cit) for comparison."""
    if method == "rf":
        if task == "clf":
            return RandomForestClassifier(
                n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=seed
            )
        return RandomForestRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=seed)
    elif method == "et":
        if task == "clf":
            return ExtraTreesClassifier(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=seed)
        return ExtraTreesRegressor(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=seed)
    elif method == "cit":
        return build_cit(task, seed)
    else:
        raise ValueError(f"Unknown baseline: {method}")


# =============================================================================
# Downstream model factories (matches Stage 2 canonical params exactly)
# =============================================================================


def make_clf_downstream(seed: int, *, n_jobs: int = 1) -> dict[str, BaseEstimator]:
    """Downstream classifiers matching Stage 2 canonical params."""
    return {
        "lr": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "svm": SVC(
            class_weight="balanced",
            probability=True,
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=n_jobs,
        ),
    }


def make_reg_downstream(seed: int, *, n_jobs: int = 1) -> dict[str, BaseEstimator]:
    """Downstream regressors matching Stage 2 canonical params."""
    return {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "svr": SVR(),
        "knn": KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
            n_jobs=n_jobs,
        ),
    }


# =============================================================================
# Synthetic dataset factories — CLASSIFICATION
# =============================================================================


def shuffle_columns(X: np.ndarray, n_informative: int, seed: int) -> tuple[np.ndarray, list[int]]:
    """Randomly permute columns and return updated informative indices."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    inv = np.argsort(perm)
    informative = [int(inv[i]) for i in range(n_informative)]
    return X, informative


def clf_standard_easy(seed: int) -> SyntheticDataset:
    """Easy classification: n=1000, p=100, k=10, class_sep=2.0."""
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=10,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=2.0,
        flip_y=0.0,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 10, seed)
    return X, y, info, "clf_standard_easy"


def clf_standard_hard(seed: int) -> SyntheticDataset:
    """Hard classification: n=200, p=1000, k=5, class_sep=0.5."""
    X, y = make_classification(
        n_samples=200,
        n_features=1000,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=0.5,
        flip_y=0.0,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 5, seed)
    return X, y, info, "clf_standard_hard"


def clf_weak_signal(seed: int) -> SyntheticDataset:
    """Weak signal classification: class_sep=0.1, flip_y=0.1."""
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=10,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=0.1,
        flip_y=0.1,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 10, seed)
    return X, y, info, "clf_weak_signal"


def clf_nonlinear(seed: int) -> SyntheticDataset:
    """Nonlinear classification via binarized Friedman1."""
    X, y_cont = make_friedman1(n_samples=1000, n_features=100, noise=1.0, random_state=seed)
    y = (y_cont >= np.median(y_cont)).astype(int)
    X, info = shuffle_columns(X, 5, seed)
    return X, y, info, "clf_nonlinear"


def clf_toeplitz(seed: int) -> SyntheticDataset:
    """Classification with Toeplitz-correlated features (rho=0.95)."""
    rng = np.random.RandomState(seed)
    p, k, n = 100, 10, 1000
    idx = np.arange(p)
    cov = 0.95 ** np.abs(np.subtract.outer(idx, idx))
    cov += np.eye(p) * 1e-6
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    beta = rng.normal(size=k)
    signal = X[:, :k] @ beta + rng.normal(scale=1.0, size=n)
    y = (signal >= np.median(signal)).astype(int)
    perm = rng.permutation(p)
    X = X[:, perm]
    inv = np.argsort(perm)
    info = [int(inv[i]) for i in range(k)]
    return X, y, info, "clf_toeplitz"


def clf_confounder(seed: int) -> SyntheticDataset:
    """Classification with 20 confounder columns correlated at rho=0.9."""
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=10,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.0,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 10, seed)
    rng = np.random.RandomState(seed + 3)
    corr = 0.9
    noise_cols = []
    for i in range(20):
        base_idx = info[i % len(info)]
        col = corr * X[:, base_idx] + rng.randn(X.shape[0]) * np.sqrt(1 - corr**2)
        noise_cols.append(col.reshape(-1, 1))
    X = np.hstack([X, *noise_cols])
    return X, y, info, "clf_confounder"


def clf_bias(seed: int) -> SyntheticDataset:
    """Classification with 50 high-cardinality integer noise features."""
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=10,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.0,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 10, seed)
    rng = np.random.RandomState(seed + 1)
    noise = rng.randint(0, 500, size=(X.shape[0], 50)).astype(float)
    X = np.hstack([X, noise])
    return X, y, info, "clf_bias"


def clf_redundant(seed: int) -> SyntheticDataset:
    """Classification with 20 redundant features (linear combos of informative)."""
    rng = np.random.RandomState(seed)
    n, p_base, k, n_red = 1000, 50, 10, 20
    X_base = rng.randn(n, p_base)
    red_cols = []
    for _ in range(n_red):
        w = rng.randn(k)
        red_cols.append((X_base[:, :k] @ w + rng.randn(n) * 0.1).reshape(-1, 1))
    X = np.hstack([X_base, *red_cols])
    beta = rng.normal(size=k)
    signal = X_base[:, :k] @ beta
    y = (signal >= np.median(signal)).astype(int)
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]
    inv = np.argsort(perm)
    info = [int(inv[i]) for i in range(k)]
    return X, y, info, "clf_redundant"


CLF_ALL = [
    clf_standard_easy,
    clf_standard_hard,
    clf_weak_signal,
    clf_nonlinear,
    clf_toeplitz,
    clf_confounder,
    clf_bias,
    clf_redundant,
]
CLF_CHALLENGING = [clf_confounder, clf_toeplitz, clf_weak_signal]


# =============================================================================
# Synthetic dataset factories — REGRESSION
# =============================================================================


def reg_friedman(seed: int) -> SyntheticDataset:
    """Regression with Friedman1 function: n=1000, p=100, k=5."""
    X, y = make_friedman1(n_samples=1000, n_features=100, noise=1.0, random_state=seed)
    X, info = shuffle_columns(X, 5, seed)
    return X, y, info, "reg_friedman"


def reg_linear(seed: int) -> SyntheticDataset:
    """Linear regression: n=1000, p=100, k=10, noise=10."""
    X, y = make_regression(
        n_samples=1000,
        n_features=100,
        n_informative=10,
        noise=10.0,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 10, seed)
    return X, y, info, "reg_linear"


def reg_highdim(seed: int) -> SyntheticDataset:
    """High-dimensional regression: n=200, p=500, k=5."""
    X, y = make_regression(
        n_samples=200,
        n_features=500,
        n_informative=5,
        noise=10.0,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 5, seed)
    return X, y, info, "reg_highdim"


def reg_toeplitz(seed: int) -> SyntheticDataset:
    """Regression with Toeplitz-correlated features (rho=0.95)."""
    rng = np.random.RandomState(seed)
    p, k, n = 100, 10, 1000
    idx = np.arange(p)
    cov = 0.95 ** np.abs(np.subtract.outer(idx, idx))
    cov += np.eye(p) * 1e-6
    X = rng.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)
    beta = rng.normal(size=k)
    y = X[:, :k] @ beta + rng.normal(scale=1.0, size=n)
    perm = rng.permutation(p)
    X = X[:, perm]
    inv = np.argsort(perm)
    info = [int(inv[i]) for i in range(k)]
    return X, y, info, "reg_toeplitz"


def reg_weak_signal(seed: int) -> SyntheticDataset:
    """Weak signal regression: noise=100."""
    X, y = make_regression(
        n_samples=1000,
        n_features=100,
        n_informative=10,
        noise=100.0,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 10, seed)
    return X, y, info, "reg_weak_signal"


def reg_confounder(seed: int) -> SyntheticDataset:
    """Regression with 20 confounder columns correlated at rho=0.9."""
    X, y = make_regression(
        n_samples=1000,
        n_features=100,
        n_informative=10,
        noise=10.0,
        random_state=seed,
        shuffle=False,
    )
    X, info = shuffle_columns(X, 10, seed)
    rng = np.random.RandomState(seed + 3)
    corr = 0.9
    noise_cols = []
    for i in range(20):
        base_idx = info[i % len(info)]
        col = corr * X[:, base_idx] + rng.randn(X.shape[0]) * np.sqrt(1 - corr**2)
        noise_cols.append(col.reshape(-1, 1))
    X = np.hstack([X, *noise_cols])
    return X, y, info, "reg_confounder"


REG_ALL = [reg_friedman, reg_linear, reg_highdim, reg_toeplitz, reg_weak_signal, reg_confounder]
REG_CHALLENGING = [reg_confounder, reg_toeplitz, reg_weak_signal]


# =============================================================================
# Real dataset loaders
# =============================================================================

REAL_CLF_NAMES = [
    "iris",
    "wine",
    "breast_cancer",
    "digits",
    "openml_madelon",
    "openml_waveform",
    "openml_optdigits",
]
REAL_REG_NAMES = ["diabetes", "california"]


def load_real_clf(name: str) -> RealDataset:
    """Load a real classification dataset from sklearn/openml."""
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }

    if name in loaders:
        data = loaders[name]()
        X, y = data.data, data.target
    elif name == "openml_madelon":
        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=1485, as_frame=False, parser="auto")
        X, y = data.data, data.target.astype(int)
    elif name == "openml_waveform":
        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=60, as_frame=False, parser="auto")
        X, y = data.data, data.target.astype(int)
    elif name == "openml_optdigits":
        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=28, as_frame=False, parser="auto")
        X, y = data.data, data.target.astype(int)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = StandardScaler().fit_transform(X)
    return X, y, f"real_{name}"


def load_real_reg(name: str) -> RealDataset:
    """Load a real regression dataset from sklearn."""
    if name == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
    elif name == "california":
        data = fetch_california_housing()
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(data.data), 2000, replace=False)
        X, y = data.data[idx], data.target[idx]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = StandardScaler().fit_transform(X)
    return X, y, f"real_{name}"


# =============================================================================
# Evaluation helpers
# =============================================================================


def compute_spread(ranking: list[int], n_features: int, k: int = K) -> float:
    """Compute spread of top-k features as fraction of total feature range."""
    top_k = ranking[:k]
    if len(top_k) < 2 or n_features <= 1:
        return np.nan
    return (max(top_k) - min(top_k)) / (n_features - 1)


def compute_confounder_rate(
    ranking: list[int], true_info: list[int], n_base_features: int, k: int = K
) -> float:
    """Fraction of top-k features that are confounders (not informative, beyond base)."""
    top_k = ranking[:k]
    confounders = sum(1 for f in top_k if f >= n_base_features and f not in true_info)
    return confounders / k


def downstream_clf(
    X: np.ndarray, y: np.ndarray, ranking: list[int], k: int, seed: int
) -> dict[str, float]:
    """Evaluate classification downstream accuracy (LR, SVM, KNN) on top-k features."""
    top_k = ranking[:k]
    models = make_clf_downstream(seed)
    if len(top_k) == 0:
        return {f"{name}_ba": np.nan for name in models}
    X_sel = X[:, top_k]
    results: dict[str, float] = {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for name, clf in models.items():
        scores = []
        for tr, te in skf.split(X_sel, y):
            clf.fit(X_sel[tr], y[tr])
            scores.append(balanced_accuracy_score(y[te], clf.predict(X_sel[te])))
        results[f"{name}_ba"] = float(np.mean(scores))
    return results


def downstream_reg(
    X: np.ndarray, y: np.ndarray, ranking: list[int], k: int, seed: int
) -> dict[str, float]:
    """Evaluate regression downstream accuracy (Ridge, SVR, KNN) on top-k features."""
    top_k = ranking[:k]
    models = make_reg_downstream(seed)
    if len(top_k) == 0:
        return {f"{name}_r2": np.nan for name in models}
    X_sel = X[:, top_k]
    results: dict[str, float] = {}
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    for name, est in models.items():
        scores = []
        for tr, te in kf.split(X_sel, y):
            est.fit(X_sel[tr], y[tr])
            scores.append(r2_score(y[te], est.predict(X_sel[te])))
        results[f"{name}_r2"] = float(np.mean(scores))
    return results


def get_tree_stats(model: BaseEstimator) -> dict[str, float]:
    """Extract tree depth and n_features_used from a fitted tree or forest."""
    if hasattr(model, "estimators_"):
        depths = []
        n_used = []
        for est in model.estimators_:
            depths.append(getattr(est, "depth_", 0))
            fi = est.feature_importances_
            n_used.append(int((fi > 0).sum()))
        return {
            "mean_depth": float(np.mean(depths)),
            "max_depth": float(np.max(depths)),
            "mean_features_used": float(np.mean(n_used)),
            "n_estimators_actual": float(len(model.estimators_)),
        }
    else:
        return {
            "mean_depth": float(getattr(model, "depth_", 0)),
            "max_depth": float(getattr(model, "depth_", 0)),
            "mean_features_used": float((model.feature_importances_ > 0).sum()),
            "n_estimators_actual": 1.0,
        }


def fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    true_info: list[int] | None,
    model: BaseEstimator,
    seed: int,
    task: str,
    has_confounders: bool = False,
    n_base_features: int | None = None,
) -> dict[str, float]:
    """Fit a model, extract ranking, and compute all evaluation metrics."""
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0

    fi = model.feature_importances_
    ranking = list(np.argsort(fi)[::-1])
    n_features = X.shape[1]

    result: dict[str, float] = {
        "elapsed_seconds": elapsed,
        "spread_at_10": compute_spread(ranking, n_features, K),
    }

    if true_info is not None:
        result["precision_at_10"] = precision_at_k(ranking, true_info, K)
        result["recall_at_10"] = recall_at_k(ranking, true_info, K)
        result["f1_at_10"] = f1_at_k(ranking, true_info, K)

    if task == "clf":
        result.update(downstream_clf(X, y, ranking, K, seed))
    else:
        result.update(downstream_reg(X, y, ranking, K, seed))

    if has_confounders and n_base_features is not None and true_info is not None:
        result["confounder_rate_at_5"] = compute_confounder_rate(
            ranking, true_info, n_base_features, 5
        )
        result["confounder_rate_at_10"] = compute_confounder_rate(
            ranking, true_info, n_base_features, 10
        )

    return result


def fit_and_evaluate_with_structure(
    X: np.ndarray,
    y: np.ndarray,
    true_info: list[int] | None,
    model: BaseEstimator,
    seed: int,
    task: str,
    has_confounders: bool = False,
    n_base_features: int | None = None,
) -> dict[str, float]:
    """Like fit_and_evaluate but also captures tree structure stats."""
    result = fit_and_evaluate(X, y, true_info, model, seed, task, has_confounders, n_base_features)
    result.update(get_tree_stats(model))
    return result


# =============================================================================
# Aggregation
# =============================================================================

_ALL_METRICS = [
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
    "spread_at_10",
    "lr_ba",
    "svm_ba",
    "knn_ba",
    "ridge_r2",
    "svr_r2",
    "knn_r2",
    "elapsed_seconds",
    "confounder_rate_at_5",
    "confounder_rate_at_10",
    "mean_depth",
    "max_depth",
    "mean_features_used",
]


def aggregate_seeds(
    seed_results: list[dict[str, float]], base_row: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate per-seed results into mean/std for each metric."""
    agg: dict[str, Any] = dict(base_row)
    for metric in _ALL_METRICS:
        vals = [
            r[metric] for r in seed_results if metric in r and not np.isnan(r.get(metric, np.nan))
        ]
        if vals:
            agg[f"{metric}_mean"] = float(np.mean(vals))
            agg[f"{metric}_std"] = float(np.std(vals))
    return agg


def format_line(variant_name: str, agg: dict[str, Any], has_confounders: bool = False) -> str:
    """Format a single-line progress summary for console output."""
    p10 = agg.get("precision_at_10_mean")
    f1 = agg.get("f1_at_10_mean")
    spread = agg.get("spread_at_10_mean", 0)
    t = agg.get("elapsed_seconds_mean", 0)
    ds = agg.get("lr_ba_mean", agg.get("ridge_r2_mean", 0))
    conf = agg.get("confounder_rate_at_10_mean")

    parts = [f"  {variant_name:20s}:"]
    if p10 is not None:
        parts.append(f"P@10={p10:.3f}")
    if f1 is not None:
        parts.append(f"F1={f1:.3f}")
    parts.append(f"spread={spread:.3f} ds={ds:.3f} t={t:.1f}s")
    if has_confounders and conf is not None:
        parts.append(f"conf@10={conf:.3f}")
    return " ".join(parts)


def format_line_with_structure(
    variant_name: str, agg: dict[str, Any], has_confounders: bool = False
) -> str:
    """Format a progress line including tree depth and n_features_used."""
    depth = agg.get("mean_depth_mean", 0)
    n_feat_used = agg.get("mean_features_used_mean", 0)
    p10 = agg.get("precision_at_10_mean", 0)
    ds = agg.get("lr_ba_mean", agg.get("ridge_r2_mean", 0))
    t = agg.get("elapsed_seconds_mean", 0)
    conf = agg.get("confounder_rate_at_10_mean")
    line = (
        f"  {variant_name:22s}: P@10={p10:.3f} ds={ds:.3f} "
        f"depth={depth:.1f} feats={n_feat_used:.1f} t={t:.1f}s"
    )
    if has_confounders and conf is not None:
        line += f" conf@10={conf:.3f}"
    return line


# =============================================================================
# Shared experiment variant definitions
# =============================================================================

OPTIMIZATION_VARIANTS: dict[str, dict[str, Any]] = {
    "cif_default": dict(),
    "cif_no_scan": dict(feature_scanning=False),
    "cif_no_mute": dict(feature_muting=False),
    "cif_no_adaptive": dict(early_stopping_selector=None, early_stopping_splitter=None),
    "cif_no_bootstrap": dict(bootstrap=False, sampling_method=None),
    "cif_subsample_80": dict(max_samples=0.8),
    "cif_subsample_50": dict(max_samples=0.5),
    "cif_no_scan_mute": dict(feature_scanning=False, feature_muting=False),
    "cif_all_off": dict(
        feature_scanning=False,
        feature_muting=False,
        early_stopping_selector=None,
        early_stopping_splitter=None,
    ),
}

REAL_ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "cif_default": dict(),
    "cif_no_scan": dict(feature_scanning=False),
    "cif_no_mute": dict(feature_muting=False),
    "cif_no_bootstrap": dict(bootstrap=False, sampling_method=None),
    "cif_no_bonferroni": dict(adjust_alpha_selector=False),
    "cif_optimized": dict(adjust_alpha_selector=False, alpha_selector=0.20, max_samples=0.5),
}

BASELINES = ["rf", "et", "cit"]


# =============================================================================
# Output helpers
# =============================================================================


def save_results(df: pd.DataFrame, name: str) -> Path:
    """Save CSV to both TABLES_DIR and DATA_DIR, return the TABLES_DIR path."""
    tables_path = TABLES_DIR / f"{name}.csv"
    data_path = DATA_DIR / f"{name}.csv"
    df.to_csv(tables_path, index=False)
    df.to_csv(data_path, index=False)
    return tables_path


def load_results(name: str) -> pd.DataFrame:
    """Load experiment CSV from DATA_DIR (preferred) or TABLES_DIR."""
    data_path = DATA_DIR / f"{name}.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    tables_path = TABLES_DIR / f"{name}.csv"
    if tables_path.exists():
        return pd.read_csv(tables_path)
    raise FileNotFoundError(f"No CSV found for '{name}' in {DATA_DIR} or {TABLES_DIR}")


# =============================================================================
# JIT warmup
# =============================================================================


def warmup_jit() -> None:
    """Run one-time Numba JIT compilation warmup with tiny datasets."""
    print("Warming up Numba JIT (one-time)...")
    rng = np.random.RandomState(0)
    X = rng.randn(30, 5)
    y_clf = (X[:, 0] > 0).astype(int)
    y_reg = X[:, 0] + rng.randn(30) * 0.1

    clf = ConditionalInferenceForestClassifier(
        n_estimators=2,
        selector="mc",
        splitter="gini",
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        threshold_method="histogram",
        max_thresholds=8,
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    clf.fit(X, y_clf)

    reg = ConditionalInferenceForestRegressor(
        n_estimators=2,
        selector="pc",
        splitter="mse",
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        threshold_method="histogram",
        max_thresholds=8,
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    reg.fit(X, y_reg)

    cit = ConditionalInferenceTreeClassifier(
        selector="mc",
        splitter="gini",
        n_resamples_selector="minimum",
        n_resamples_splitter="minimum",
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        threshold_method="histogram",
        max_thresholds=8,
        random_state=0,
        verbose=0,
    )
    cit.fit(X, y_clf)
    print("JIT warmup complete.")
