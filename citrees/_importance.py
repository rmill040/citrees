"""Feature importance methods for citrees.

Supports:
- mdi: Mean Decrease Impurity (default, built-in)
- permutation: Permutation importance (sklearn)
- shap: SHAP values (model-agnostic)
- cpi: Conditional Permutation Importance (Strobl et al. 2008)
"""

import numpy as np
from sklearn.base import BaseEstimator

from citrees._types import ImportanceMethod


class SHAPExplainer:
    """SHAP explainer for citrees models.

    Since citrees uses a custom tree structure not directly compatible with
    SHAP's TreeExplainer, this uses the model-agnostic Explainer which
    automatically selects an appropriate algorithm.

    Parameters
    ----------
    model : BaseEstimator
        Fitted citrees model (tree or forest).
    background_data : np.ndarray, optional
        Background dataset for SHAP. If None, uses a sample of training data.
        Smaller background datasets are faster but may be less accurate.
    max_background : int, default=100
        Maximum number of background samples to use.

    Attributes
    ----------
    explainer_ : shap.Explainer
        The underlying SHAP explainer.

    Examples
    --------
    >>> from citrees import ConditionalInferenceTreeClassifier
    >>> from citrees._importance import SHAPExplainer
    >>> clf = ConditionalInferenceTreeClassifier()
    >>> clf.fit(X_train, y_train)
    >>> explainer = SHAPExplainer(clf, X_train)
    >>> shap_values = explainer.shap_values(X_test)
    >>> importances = explainer.feature_importance()
    """

    def __init__(
        self,
        model: BaseEstimator,
        background_data: np.ndarray | None = None,
        max_background: int = 100,
    ):
        try:
            import shap
        except ImportError:
            raise ImportError("shap not installed. Run: uv pip install shap")

        self.model = model
        self.max_background = max_background

        # Subsample background if needed
        if background_data is not None:
            if len(background_data) > max_background:
                idx = np.random.choice(len(background_data), max_background, replace=False)
                background_data = background_data[idx]

        # Use model-agnostic explainer
        # For classifiers, wrap predict_proba if available
        if hasattr(model, "predict_proba"):
            self.explainer_ = shap.Explainer(model.predict_proba, background_data)
            self._is_classifier = True
        else:
            self.explainer_ = shap.Explainer(model.predict, background_data)
            self._is_classifier = False

        self._shap_values = None

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to explain.

        Returns
        -------
        np.ndarray
            SHAP values. Shape depends on task:
            - Regression: (n_samples, n_features)
            - Binary classification: (n_samples, n_features) for positive class
            - Multiclass: (n_samples, n_features, n_classes)
        """
        explanation = self.explainer_(X)
        self._shap_values = explanation.values

        # For binary classification, return SHAP values for positive class
        if self._is_classifier and self._shap_values.ndim == 3 and self._shap_values.shape[2] == 2:
            return self._shap_values[:, :, 1]

        return self._shap_values

    def feature_importance(self, X: np.ndarray | None = None) -> np.ndarray:
        """Compute mean absolute SHAP values as feature importance.

        Parameters
        ----------
        X : np.ndarray, optional
            Data to compute importance from. If None, uses cached values
            from last shap_values() call.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Mean absolute SHAP value per feature.
        """
        if X is not None:
            self.shap_values(X)

        if self._shap_values is None:
            raise ValueError("No SHAP values computed. Call shap_values(X) first.")

        values = self._shap_values
        # Handle multiclass: average across classes
        if values.ndim == 3:
            values = np.abs(values).mean(axis=2)
        else:
            values = np.abs(values)

        return values.mean(axis=0)


def permutation_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Compute permutation importance.

    Parameters
    ----------
    model : BaseEstimator
        Fitted model.
    X : np.ndarray
        Features.
    y : np.ndarray
        Target.
    n_repeats : int, default=10
        Number of times to permute each feature.
    seed : int, default=42
        Random seed.

    Returns
    -------
    np.ndarray of shape (n_features,)
        Mean importance decrease for each feature.
    """
    from sklearn.inspection import permutation_importance as sklearn_pi

    result = sklearn_pi(model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return result.importances_mean


def shap_importance(
    model: BaseEstimator,
    X: np.ndarray,
    background_data: np.ndarray | None = None,
    max_background: int = 100,
) -> np.ndarray:
    """Compute SHAP-based feature importance.

    Parameters
    ----------
    model : BaseEstimator
        Fitted model.
    X : np.ndarray
        Data to compute importance from.
    background_data : np.ndarray, optional
        Background dataset. If None, uses X.
    max_background : int, default=100
        Maximum background samples.

    Returns
    -------
    np.ndarray of shape (n_features,)
        Mean absolute SHAP value per feature.
    """
    if background_data is None:
        background_data = X

    explainer = SHAPExplainer(model, background_data, max_background)
    return explainer.feature_importance(X)


def conditional_permutation_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    seed: int = 42,
    correlation_threshold: float = 0.5,
) -> np.ndarray:
    """Compute Conditional Permutation Importance.

    CPI permutes features conditionally on correlated features,
    avoiding the bias that standard permutation importance has
    when features are correlated.

    Parameters
    ----------
    model : BaseEstimator
        Fitted model.
    X : np.ndarray
        Features.
    y : np.ndarray
        Target.
    n_repeats : int, default=10
        Number of times to permute each feature.
    seed : int, default=42
        Random seed.
    correlation_threshold : float, default=0.5
        Features with |correlation| > threshold are considered correlated.

    Returns
    -------
    np.ndarray of shape (n_features,)
        Normalized importance scores (sum to 1).

    References
    ----------
    Strobl et al. (2008) - "Conditional Variable Importance for Random Forests"
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

            # Find correlated features
            corr_features = _find_correlated(X, j, threshold=correlation_threshold)

            if len(corr_features) > 0:
                # Conditional permutation: permute within strata
                strata = _create_strata(X[:, corr_features])
                for stratum in np.unique(strata):
                    mask = strata == stratum
                    idx = np.where(mask)[0]
                    if len(idx) > 1:  # Need at least 2 samples to permute
                        X_permuted[idx, j] = rng.permutation(X_permuted[idx, j])
            else:
                # Standard permutation
                X_permuted[:, j] = rng.permutation(X_permuted[:, j])

            scores.append(_score(model, X_permuted, y))

        importances[j] = baseline - np.mean(scores)

    # Normalize to sum to 1 (only positive values contribute)
    if importances.sum() > 0:
        importances = np.maximum(importances, 0)
        total = importances.sum()
        if total > 0:
            importances /= total

    return importances


def _score(model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    """Score model (accuracy for classifier, R2 for regressor)."""
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
            # Handle constant features
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

    # Bin each feature and combine into strata IDs
    strata = np.zeros(X.shape[0], dtype=int)
    for j in range(X.shape[1]):
        # Handle constant features
        if np.std(X[:, j]) == 0:
            continue
        bins = np.percentile(X[:, j], np.linspace(0, 100, n_bins + 1))
        # Ensure unique bin edges
        bins = np.unique(bins)
        if len(bins) > 1:
            binned = np.digitize(X[:, j], bins[1:-1])
            strata = strata * n_bins + binned

    return strata


def compute_importance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray | None = None,
    method: ImportanceMethod = "mdi",
    **kwargs,
) -> np.ndarray:
    """Compute feature importance using specified method.

    Parameters
    ----------
    model : BaseEstimator
        Fitted citrees model.
    X : np.ndarray
        Features.
    y : np.ndarray, optional
        Target. Required for permutation and CPI methods.
    method : {"mdi", "permutation", "shap", "cpi"}, default="mdi"
        Importance method:
        - "mdi": Mean Decrease Impurity (built-in feature_importances_)
        - "permutation": Permutation importance
        - "shap": SHAP values
        - "cpi": Conditional Permutation Importance
    **kwargs
        Additional arguments passed to the importance function.

    Returns
    -------
    np.ndarray of shape (n_features,)
        Feature importance scores.

    Examples
    --------
    >>> from citrees import ConditionalInferenceForestClassifier
    >>> from citrees._importance import compute_importance
    >>> clf = ConditionalInferenceForestClassifier()
    >>> clf.fit(X_train, y_train)
    >>> # Built-in MDI importance
    >>> imp_mdi = compute_importance(clf, X_train, method="mdi")
    >>> # Permutation importance
    >>> imp_perm = compute_importance(clf, X_train, y_train, method="permutation")
    >>> # SHAP importance
    >>> imp_shap = compute_importance(clf, X_train, method="shap")
    >>> # Conditional Permutation Importance
    >>> imp_cpi = compute_importance(clf, X_train, y_train, method="cpi")
    """
    if method == ImportanceMethod.MDI:
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model doesn't have feature_importances_. Fit the model first.")
        return model.feature_importances_

    elif method == ImportanceMethod.PERMUTATION:
        if y is None:
            raise ValueError("y is required for permutation importance")
        return permutation_importance(model, X, y, **kwargs)

    elif method == ImportanceMethod.SHAP:
        return shap_importance(model, X, **kwargs)

    elif method == ImportanceMethod.CPI:
        if y is None:
            raise ValueError("y is required for CPI")
        return conditional_permutation_importance(model, X, y, **kwargs)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: {list(ImportanceMethod)}")
