"""Feature selection methods for embedding and wrapper approaches.

This module provides model factory and wrapper selector functions used by stage1.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from boruta import BorutaPy
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from mrmr import mrmr_classif, mrmr_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)

_DEFAULT_N_JOBS = 1


def get_embedding_model(
    method: str,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
    params: dict[str, Any] | None = None,
) -> Any:
    """Get embedding model for feature importance extraction.

    Parameters
    ----------
    method : str
        Model name: rf, et, xgb, lgbm, cat, cit, cif.
    task_type : str
        "classification" or "regression".
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.
    params : dict, optional
        Additional model parameters.

    Returns
    -------
    model
        Fitted model with feature_importances_ attribute.
    """
    params = params or {}

    if task_type == "classification":
        if method == "rf":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return RandomForestClassifier(**model_params)
        if method == "et":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return ExtraTreesClassifier(**model_params)
        if method == "cit":
            model_params = {**params, "random_state": random_state}
            return ConditionalInferenceTreeClassifier(**model_params)
        if method == "cif":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return ConditionalInferenceForestClassifier(**model_params)
        if method == "xgb":
            base = {
                "n_estimators": 100,
                "n_jobs": n_jobs,
                "random_state": random_state,
                "verbosity": 0,
            }
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return XGBClassifier(**model_params)
        if method == "lgbm":
            base = {
                "n_estimators": 100,
                "n_jobs": n_jobs,
                "random_state": random_state,
                "verbosity": -1,
            }
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return LGBMClassifier(**model_params)
        if method == "cat":
            base = {
                "n_estimators": 100,
                "random_state": random_state,
                "verbose": 0,
                "thread_count": n_jobs,
            }
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return CatBoostClassifier(**model_params)
    else:
        if method == "rf":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return RandomForestRegressor(**model_params)
        if method == "et":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return ExtraTreesRegressor(**model_params)
        if method == "cit":
            model_params = {**params, "random_state": random_state}
            return ConditionalInferenceTreeRegressor(**model_params)
        if method == "cif":
            base = {"n_estimators": 100, "n_jobs": n_jobs, "random_state": random_state}
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return ConditionalInferenceForestRegressor(**model_params)
        if method == "xgb":
            base = {
                "n_estimators": 100,
                "n_jobs": n_jobs,
                "random_state": random_state,
                "verbosity": 0,
            }
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return XGBRegressor(**model_params)
        if method == "lgbm":
            base = {
                "n_estimators": 100,
                "n_jobs": n_jobs,
                "random_state": random_state,
                "verbosity": -1,
            }
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return LGBMRegressor(**model_params)
        if method == "cat":
            base = {
                "n_estimators": 100,
                "random_state": random_state,
                "verbose": 0,
                "thread_count": n_jobs,
            }
            model_params = {**base, **params}
            model_params["random_state"] = random_state
            return CatBoostRegressor(**model_params)

    raise ValueError(f"Unknown embedding method: {method}")


def boruta_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using Boruta algorithm.

    Parameters
    ----------
    params : dict, optional
        - n_estimators: "auto", 100, or 200 (default "auto")
        - max_iter: max iterations (default 100)
    """
    params = params or {}
    n_estimators = params.get("n_estimators", "auto")
    max_iter = params.get("max_iter", 100)

    if task_type == "classification":
        base_model = RandomForestClassifier(
            n_estimators=100, n_jobs=n_jobs, random_state=random_state
        )
    else:
        base_model = RandomForestRegressor(
            n_estimators=100, n_jobs=n_jobs, random_state=random_state
        )

    boruta = BorutaPy(
        base_model,
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )
    boruta.fit(X_train, y_train)
    return np.argsort(boruta.ranking_)


def pi_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
    val_fraction: float = 0.2,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using permutation importance.

    Parameters
    ----------
    params : dict, optional
        - n_repeats: number of permutation repeats (default 10)
    """
    params = params or {}
    n_repeats = params.get("n_repeats", 10)

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
        scoring = "accuracy"
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
        scoring = "r2"

    stratify = y_train if task_type == "classification" else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_fraction,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(X_fit, y_fit)
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    return np.argsort(result.importances_mean)[::-1]


def shap_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using SHAP values.

    Parameters
    ----------
    params : dict, optional
        - max_samples: max samples for SHAP computation (default 1000)
    """
    params = params or {}
    max_samples = params.get("max_samples", 1000)

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)

    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)

    if X_train.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_train.shape[0], max_samples, replace=False)
        X_sample = X_train[idx]
    else:
        X_sample = X_train

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        importances = np.abs(shap_values).mean(axis=0)

    return np.argsort(importances)[::-1]


def _find_correlated(X: np.ndarray, j: int, threshold: float) -> list[int]:
    """Find features correlated with feature j above threshold."""
    correlated = []
    for k in range(X.shape[1]):
        if k != j:
            if np.std(X[:, j]) == 0 or np.std(X[:, k]) == 0:
                continue
            corr = np.abs(np.corrcoef(X[:, j], X[:, k])[0, 1])
            if not np.isnan(corr) and corr > threshold:
                correlated.append(k)
    return correlated


def _create_strata(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Create strata for conditional permutation."""
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=int)
    strata = np.zeros(X.shape[0], dtype=int)
    for j in range(X.shape[1]):
        if np.std(X[:, j]) == 0:
            continue
        bins = np.unique(np.percentile(X[:, j], np.linspace(0, 100, n_bins + 1)))
        if len(bins) > 1:
            binned = np.digitize(X[:, j], bins[1:-1])
            strata = strata * n_bins + binned
    return strata


def cpi_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
    val_fraction: float = 0.2,
    correlation_threshold: float = 0.5,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Compute feature ranking using conditional permutation importance.

    Parameters
    ----------
    params : dict, optional
        - n_repeats: number of permutation repeats (default 10)
    """
    params = params or {}
    n_repeats = params.get("n_repeats", 10)

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)

        def scoring_fn(m, X, y):
            return (m.predict(X) == y).mean()

    else:
        model = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)

        def scoring_fn(m, X, y):
            return 1 - ((y - m.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    stratify = y_train if task_type == "classification" else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_fraction,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(X_fit, y_fit)
    rng = np.random.default_rng(random_state)
    n_features = X_val.shape[1]
    baseline = scoring_fn(model, X_val, y_val)
    importances = np.zeros(n_features)

    for j in range(n_features):
        scores = []
        for _rep in range(n_repeats):
            X_perm = X_val.copy()
            corr_features = _find_correlated(X_val, j, correlation_threshold)

            if len(corr_features) > 0:
                strata = _create_strata(X_val[:, corr_features])
                for stratum in np.unique(strata):
                    mask = strata == stratum
                    idx = np.where(mask)[0]
                    if len(idx) > 1:
                        X_perm[idx, j] = rng.permutation(X_perm[idx, j])
            else:
                X_perm[:, j] = rng.permutation(X_perm[:, j])

            scores.append(scoring_fn(model, X_perm, y_val))
        importances[j] = baseline - np.mean(scores)

    return np.argsort(importances)[::-1]


def mrmr_selector(X_train: np.ndarray, y_train: np.ndarray, task_type: str) -> np.ndarray:
    """Select features using mRMR (minimum Redundancy Maximum Relevance).

    Note: The mrmr library has no random_state parameter and is non-deterministic.
    Results may vary between runs.

    Note: mRMR internally drops constant features (zero variance). This function
    appends any dropped features at the end of the ranking to ensure all features
    are included.
    """
    df = pd.DataFrame(X_train)
    y_series = pd.Series(y_train)
    n_features = X_train.shape[1]

    if task_type == "classification":
        selected = mrmr_classif(df, y_series, K=n_features, show_progress=False)
    else:
        selected = mrmr_regression(df, y_series, K=n_features, show_progress=False)

    ranking = np.array(selected)

    # Append missing features (e.g., constant features dropped by mRMR)
    if len(ranking) < n_features:
        all_features = set(range(n_features))
        ranked_features = set(ranking)
        missing = sorted(all_features - ranked_features)
        ranking = np.concatenate([ranking, np.array(missing)])

    return ranking


def rfe_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    random_state: int,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> np.ndarray:
    """Compute feature ranking using Recursive Feature Elimination."""
    if task_type == "classification":
        base_model = RandomForestClassifier(
            n_estimators=100, n_jobs=n_jobs, random_state=random_state
        )
    else:
        base_model = RandomForestRegressor(
            n_estimators=100, n_jobs=n_jobs, random_state=random_state
        )

    rfe = RFE(base_model, n_features_to_select=1, step=1)
    rfe.fit(X_train, y_train)
    return np.argsort(rfe.ranking_)
