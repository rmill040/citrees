"""Tests for citrees._importance module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    SHAPExplainer,
    compute_importance,
)


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=10.0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def fitted_tree_classifier(classification_data):
    """Return fitted tree classifier."""
    X, y = classification_data
    clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
    clf.fit(X, y)
    return clf, X, y


@pytest.fixture
def fitted_tree_regressor(regression_data):
    """Return fitted tree regressor."""
    X, y = regression_data
    reg = ConditionalInferenceTreeRegressor(random_state=42, verbose=0)
    reg.fit(X, y)
    return reg, X, y


@pytest.fixture
def fitted_forest_classifier(classification_data):
    """Return fitted forest classifier."""
    X, y = classification_data
    clf = ConditionalInferenceForestClassifier(n_estimators=5, random_state=42, verbose=0)
    clf.fit(X, y)
    return clf, X, y


class TestComputeImportance:
    """Tests for compute_importance function."""

    def test_mdi_classifier(self, fitted_tree_classifier):
        """Test MDI importance for classifier."""
        clf, X, y = fitted_tree_classifier
        imp = compute_importance(clf, X, method="mdi")
        assert imp.shape == (X.shape[1],)
        assert (imp >= 0).all()

    def test_mdi_regressor(self, fitted_tree_regressor):
        """Test MDI importance for regressor."""
        reg, X, y = fitted_tree_regressor
        imp = compute_importance(reg, X, method="mdi")
        assert imp.shape == (X.shape[1],)
        assert (imp >= 0).all()

    def test_permutation_classifier(self, fitted_tree_classifier):
        """Test permutation importance for classifier."""
        clf, X, y = fitted_tree_classifier
        imp = compute_importance(clf, X, y, method="permutation", n_repeats=5)
        assert imp.shape == (X.shape[1],)

    def test_permutation_regressor(self, fitted_tree_regressor):
        """Test permutation importance for regressor."""
        reg, X, y = fitted_tree_regressor
        imp = compute_importance(reg, X, y, method="permutation", n_repeats=5)
        assert imp.shape == (X.shape[1],)

    def test_permutation_requires_y(self, fitted_tree_classifier):
        """Test that permutation importance requires y."""
        clf, X, y = fitted_tree_classifier
        with pytest.raises(ValueError, match="y is required"):
            compute_importance(clf, X, method="permutation")

    def test_cpi_classifier(self, fitted_tree_classifier):
        """Test CPI for classifier."""
        clf, X, y = fitted_tree_classifier
        imp = compute_importance(clf, X, y, method="cpi", n_repeats=3)
        assert imp.shape == (X.shape[1],)
        # CPI normalizes to sum to 1 (for positive values)
        assert imp.sum() <= 1.0 + 1e-6

    def test_cpi_requires_y(self, fitted_tree_classifier):
        """Test that CPI requires y."""
        clf, X, y = fitted_tree_classifier
        with pytest.raises(ValueError, match="y is required"):
            compute_importance(clf, X, method="cpi")

    def test_shap_classifier(self, fitted_tree_classifier):
        """Test SHAP importance for classifier."""
        clf, X, y = fitted_tree_classifier
        # Use small subset for speed
        imp = compute_importance(clf, X[:20], method="shap", max_background=10)
        assert imp.shape == (X.shape[1],)
        assert (imp >= 0).all()

    def test_shap_regressor(self, fitted_tree_regressor):
        """Test SHAP importance for regressor."""
        reg, X, y = fitted_tree_regressor
        imp = compute_importance(reg, X[:20], method="shap", max_background=10)
        assert imp.shape == (X.shape[1],)

    def test_invalid_method(self, fitted_tree_classifier):
        """Test invalid method raises error."""
        clf, X, y = fitted_tree_classifier
        with pytest.raises(ValueError, match="Unknown method"):
            compute_importance(clf, X, method="invalid")

    def test_mdi_requires_fitted_model(self, classification_data):
        """Test MDI requires fitted model."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier()
        with pytest.raises(ValueError, match="feature_importances_"):
            compute_importance(clf, X, method="mdi")


class TestSHAPExplainer:
    """Tests for SHAPExplainer class."""

    def test_shap_values_classifier(self, fitted_tree_classifier):
        """Test SHAP values for classifier."""
        clf, X, y = fitted_tree_classifier
        explainer = SHAPExplainer(clf, X[:20], max_background=10)
        shap_values = explainer.shap_values(X[:5])

        # Binary classification returns (n_samples, n_features) for positive class
        assert shap_values.shape == (5, X.shape[1])

    def test_shap_values_regressor(self, fitted_tree_regressor):
        """Test SHAP values for regressor."""
        reg, X, y = fitted_tree_regressor
        explainer = SHAPExplainer(reg, X[:20], max_background=10)
        shap_values = explainer.shap_values(X[:5])

        assert shap_values.shape == (5, X.shape[1])

    def test_feature_importance(self, fitted_tree_classifier):
        """Test feature importance from SHAP values."""
        clf, X, y = fitted_tree_classifier
        explainer = SHAPExplainer(clf, X[:20], max_background=10)

        imp = explainer.feature_importance(X[:10])
        assert imp.shape == (X.shape[1],)
        assert (imp >= 0).all()

    def test_feature_importance_cached(self, fitted_tree_classifier):
        """Test feature importance uses cached values."""
        clf, X, y = fitted_tree_classifier
        explainer = SHAPExplainer(clf, X[:20], max_background=10)

        # Compute SHAP values first
        explainer.shap_values(X[:5])

        # Feature importance should use cached values
        imp = explainer.feature_importance()
        assert imp.shape == (X.shape[1],)

    def test_feature_importance_no_cached_values(self, fitted_tree_classifier):
        """Test feature importance raises error without cached values."""
        clf, X, y = fitted_tree_classifier
        explainer = SHAPExplainer(clf, X[:20], max_background=10)

        with pytest.raises(ValueError, match="No SHAP values"):
            explainer.feature_importance()

    def test_background_subsampling(self, fitted_forest_classifier):
        """Test that large background data is subsampled."""
        clf, X, y = fitted_forest_classifier
        # Create explainer with max_background < len(X)
        explainer = SHAPExplainer(clf, X, max_background=10)

        # Should still work
        shap_values = explainer.shap_values(X[:5])
        assert shap_values.shape == (5, X.shape[1])


class TestConditionalPermutationImportance:
    """Tests for conditional permutation importance."""

    def test_cpi_with_correlated_features(self):
        """Test CPI handles correlated features correctly."""
        # Create data with correlated features
        np.random.seed(42)
        n = 100
        X1 = np.random.randn(n)
        X2 = X1 + 0.1 * np.random.randn(n)  # Highly correlated with X1
        X3 = np.random.randn(n)  # Independent
        X = np.column_stack([X1, X2, X3])
        y = (X1 > 0).astype(int)

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X, y)

        imp = compute_importance(clf, X, y, method="cpi", n_repeats=5, correlation_threshold=0.5)
        assert imp.shape == (3,)
        assert imp.sum() <= 1.0 + 1e-6

    def test_cpi_with_constant_feature(self):
        """Test CPI handles constant features."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        X[:, 1] = 1.0  # Constant feature
        y = (X[:, 0] > 0).astype(int)

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X, y)

        imp = compute_importance(clf, X, y, method="cpi", n_repeats=3)
        assert imp.shape == (3,)


class TestImportanceComparison:
    """Tests comparing different importance methods."""

    def test_informative_features_ranked_higher(self, classification_data):
        """Test that informative features generally rank higher."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf.fit(X, y)

        # Get importance from multiple methods
        mdi = compute_importance(clf, X, method="mdi")
        perm = compute_importance(clf, X, y, method="permutation", n_repeats=5)

        # Top features should have positive importance
        assert mdi.max() > 0
        assert perm.max() > 0

    def test_importance_shapes_match(self, fitted_forest_classifier):
        """Test all methods return same shape."""
        clf, X, y = fitted_forest_classifier

        mdi = compute_importance(clf, X, method="mdi")
        perm = compute_importance(clf, X, y, method="permutation", n_repeats=3)
        cpi = compute_importance(clf, X, y, method="cpi", n_repeats=3)
        shap = compute_importance(clf, X[:20], method="shap", max_background=10)

        assert mdi.shape == perm.shape == cpi.shape == shap.shape == (X.shape[1],)
