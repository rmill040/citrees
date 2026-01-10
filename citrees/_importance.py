"""Feature importance methods for citrees.

Supports:
- mdi: Mean Decrease Impurity (default, built-in)
- permutation: Permutation importance
- shap: SHAP/TreeSHAP values
- cpi: Conditional Permutation Importance
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator

ImportanceMethod = Literal["mdi", "permutation", "shap", "cpi"]


def permutation_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Compute permutation importance."""
    from sklearn.inspection import permutation_importance as sklearn_pi

    result = sklearn_pi(model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return result.importances_mean


def shap_importance(
    model: BaseEstimator,
    X: np.ndarray,
) -> np.ndarray:
    """Compute SHAP-based importance using TreeExplainer."""
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed. Run: uv pip install shap")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle multi-class: shap_values is list of arrays
    if isinstance(shap_values, list):
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        importance = np.abs(shap_values).mean(axis=0)

    return importance


def conditional_permutation_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Compute Conditional Permutation Importance.

    CPI permutes features conditionally on correlated features,
    avoiding bias from feature correlation.

    Reference: Strobl et al. (2008) - Conditional Variable Importance
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    # Baseline score
    baseline = _score(model, X, y)

    importances = np.zeros(n_features)

    for j in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()

            # Find correlated features (|r| > 0.5)
            corr_features = _find_correlated(X, j, threshold=0.5)

            if len(corr_features) > 0:
                # Conditional permutation: permute within strata
                strata = _create_strata(X[:, corr_features])
                for stratum in np.unique(strata):
                    mask = strata == stratum
                    idx = np.where(mask)[0]
                    X_permuted[idx, j] = rng.permutation(X_permuted[idx, j])
            else:
                # Standard permutation
                X_permuted[:, j] = rng.permutation(X_permuted[:, j])

            scores.append(_score(model, X_permuted, y))

        importances[j] = baseline - np.mean(scores)

    # Normalize to sum to 1
    if importances.sum() > 0:
        importances = np.maximum(importances, 0)
        importances /= importances.sum()

    return importances


def _score(model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    """Score model (accuracy for clf, r2 for reg)."""
    if hasattr(model, "predict_proba"):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, model.predict(X))
    else:
        from sklearn.metrics import r2_score
        return r2_score(y, model.predict(X))


def _find_correlated(X: np.ndarray, j: int, threshold: float = 0.5) -> list[int]:
    """Find features correlated with feature j."""
    n_features = X.shape[1]
    correlated = []
    for k in range(n_features):
        if k != j:
            corr = np.abs(np.corrcoef(X[:, j], X[:, k])[0, 1])
            if corr > threshold:
                correlated.append(k)
    return correlated


def _create_strata(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Create strata for conditional permutation."""
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=int)

    # Bin each feature and combine into strata IDs
    strata = np.zeros(X.shape[0], dtype=int)
    for j in range(X.shape[1]):
        bins = np.percentile(X[:, j], np.linspace(0, 100, n_bins + 1))
        binned = np.digitize(X[:, j], bins[1:-1])
        strata = strata * n_bins + binned

    return strata


def compute_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    method: ImportanceMethod = "mdi",
    **kwargs,
) -> np.ndarray:
    """Compute feature importance using specified method."""
    if method == "mdi":
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model doesn't have feature_importances_")
        return model.feature_importances_

    elif method == "permutation":
        return permutation_importance(model, X, y, **kwargs)

    elif method == "shap":
        return shap_importance(model, X)

    elif method == "cpi":
        return conditional_permutation_importance(model, X, y, **kwargs)

    else:
        raise ValueError(f"Unknown method: {method}")
