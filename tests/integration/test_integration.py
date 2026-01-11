"""Integration tests for citrees classifiers and regressors."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, train_test_split

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)


# Fixtures
@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate multiclass classification dataset."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X, y


# Tree Classifier Tests
class TestConditionalInferenceTreeClassifier:
    """Tests for ConditionalInferenceTreeClassifier."""

    def test_fit_predict_binary(self, classification_data):
        """Test fit and predict on binary classification."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X_train, y_train)

        # Check fitted attributes
        assert hasattr(clf, "n_features_in_")
        assert clf.n_features_in_ == X.shape[1]
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "feature_importances_")

        # Check predictions
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
        assert set(y_pred).issubset(set(y))

        # Check accuracy is reasonable
        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.5, f"Accuracy {accuracy} too low"

    def test_fit_predict_multiclass(self, multiclass_data):
        """Test fit and predict on multiclass classification."""
        X, y = multiclass_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X_train, y_train)

        assert len(clf.classes_) == 3

        y_pred = clf.predict(X_test)
        assert set(y_pred).issubset(set(y))

    def test_predict_proba(self, classification_data):
        """Test predict_proba returns valid probabilities."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_feature_importances(self, classification_data):
        """Test feature importances are computed."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X, y)

        fi = clf.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert (fi >= 0).all()
        # Importances should sum to approximately 1 for a fitted tree with splits
        # (could be 0 if tree is trivial)

    def test_clone_and_refit(self, classification_data):
        """Test that cloning works correctly."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0, alpha_selector=0.1)
        clf.fit(X, y)

        clf2 = clone(clf)
        assert clf2.alpha_selector == 0.1
        assert not hasattr(clf2, "classes_")  # Not fitted yet

        clf2.fit(X, y)
        assert hasattr(clf2, "classes_")

    def test_selector_methods(self, classification_data):
        """Test all individual selector methods."""
        X, y = classification_data
        # Test each selector individually
        for selector in ["mc", "mi", "rdc"]:
            clf = ConditionalInferenceTreeClassifier(selector=selector, random_state=42, verbose=0)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_list_selector_all_combos(self, classification_data):
        """Test all valid list selector combinations for classification.

        Only mc and rdc can be combined (both on [0,1] scale). mi cannot be in a list because it's
        unbounded.
        """
        X, y = classification_data
        # All valid combinations: only [mc, rdc] since mi is not on same scale
        for selector in [["mc", "rdc"]]:
            clf = ConditionalInferenceTreeClassifier(selector=selector, random_state=42, verbose=0)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_list_selector_no_ptest(self, classification_data):
        """Test list-based selector without permutation testing."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            selector=["mc", "rdc"], n_resamples_selector=None, random_state=42, verbose=0
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_list_selector_mi_invalid(self, classification_data):
        """Test that mi cannot be in a list selector for classification."""
        import pytest

        X, y = classification_data
        # mi + mc is invalid
        with pytest.raises(ValueError, match="mi.*cannot be used in a list"):
            ConditionalInferenceTreeClassifier(selector=["mc", "mi"], random_state=42, verbose=0)
        # mi + rdc is invalid
        with pytest.raises(ValueError, match="mi.*cannot be used in a list"):
            ConditionalInferenceTreeClassifier(selector=["mi", "rdc"], random_state=42, verbose=0)
        # mi + mc + rdc is invalid
        with pytest.raises(ValueError, match="mi.*cannot be used in a list"):
            ConditionalInferenceTreeClassifier(
                selector=["mc", "mi", "rdc"], random_state=42, verbose=0
            )

    def test_splitter_methods(self, classification_data):
        """Test different splitter methods."""
        X, y = classification_data
        for splitter in ["gini", "entropy"]:
            clf = ConditionalInferenceTreeClassifier(splitter=splitter, random_state=42, verbose=0)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape


# Tree Regressor Tests
class TestConditionalInferenceTreeRegressor:
    """Tests for ConditionalInferenceTreeRegressor."""

    def test_fit_predict(self, regression_data):
        """Test fit and predict on regression."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = ConditionalInferenceTreeRegressor(random_state=42, verbose=0)
        reg.fit(X_train, y_train)

        # Check fitted attributes
        assert hasattr(reg, "n_features_in_")
        assert reg.n_features_in_ == X.shape[1]
        assert hasattr(reg, "feature_importances_")

        # Check predictions
        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape

        # Check R2 is reasonable
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        assert r2 > 0, f"R2 {r2} is negative"

    def test_selector_methods(self, regression_data):
        """Test all individual selector methods."""
        X, y = regression_data
        # Test each selector individually
        for selector in ["pc", "dc", "rdc"]:
            reg = ConditionalInferenceTreeRegressor(selector=selector, random_state=42, verbose=0)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_list_selector_all_combos(self, regression_data):
        """Test all valid list selector combinations for regression.

        All three selectors (pc, dc, rdc) are on [0,1] scale and can be combined.
        """
        X, y = regression_data
        # All valid 2-way and 3-way combinations
        for selector in [["pc", "dc"], ["pc", "rdc"], ["dc", "rdc"], ["pc", "dc", "rdc"]]:
            reg = ConditionalInferenceTreeRegressor(selector=selector, random_state=42, verbose=0)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_list_selector_no_ptest(self, regression_data):
        """Test list-based selector without permutation testing."""
        X, y = regression_data
        # Test all combinations without ptest
        for selector in [["pc", "dc"], ["pc", "rdc"], ["dc", "rdc"], ["pc", "dc", "rdc"]]:
            reg = ConditionalInferenceTreeRegressor(
                selector=selector, n_resamples_selector=None, random_state=42, verbose=0
            )
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_splitter_methods(self, regression_data):
        """Test different splitter methods."""
        X, y = regression_data
        for splitter in ["mse", "mae"]:
            reg = ConditionalInferenceTreeRegressor(splitter=splitter, random_state=42, verbose=0)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape


# Forest Classifier Tests
class TestConditionalInferenceForestClassifier:
    """Tests for ConditionalInferenceForestClassifier."""

    def test_fit_predict_binary(self, classification_data):
        """Test fit and predict on binary classification."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf.fit(X_train, y_train)

        # Check fitted attributes
        assert hasattr(clf, "estimators_")
        assert len(clf.estimators_) == 10
        assert hasattr(clf, "feature_importances_")

        # Check predictions
        y_pred = clf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.5

    def test_fit_predict_multiclass(self, multiclass_data):
        """Test fit and predict on multiclass classification."""
        X, y = multiclass_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf.fit(X, y)

        assert clf.n_classes_ == 3
        y_pred = clf.predict(X)
        assert set(y_pred).issubset(set(y))

    def test_predict_proba(self, classification_data):
        """Test predict_proba returns valid probabilities."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_bootstrap_methods(self, classification_data):
        """Test different bootstrap methods."""
        X, y = classification_data
        for method in ["bayesian", "classic", None]:
            clf = ConditionalInferenceForestClassifier(
                n_estimators=5,
                bootstrap_method=method,
                random_state=42,
                verbose=0,
                check_for_unused_parameters=False,
            )
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_sampling_methods(self, classification_data):
        """Test different sampling methods."""
        X, y = classification_data
        for method in ["stratified", "balanced", None]:
            clf = ConditionalInferenceForestClassifier(
                n_estimators=5,
                sampling_method=method,
                random_state=42,
                verbose=0,
                check_for_unused_parameters=False,
            )
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_parallel_training(self, classification_data):
        """Test parallel training with n_jobs."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=10, n_jobs=2, random_state=42, verbose=0
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 10


# Forest Regressor Tests
class TestConditionalInferenceForestRegressor:
    """Tests for ConditionalInferenceForestRegressor."""

    def test_fit_predict(self, regression_data):
        """Test fit and predict on regression."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = ConditionalInferenceForestRegressor(n_estimators=10, random_state=42, verbose=0)
        reg.fit(X_train, y_train)

        # Check fitted attributes
        assert hasattr(reg, "estimators_")
        assert len(reg.estimators_) == 10

        # Check predictions
        y_pred = reg.predict(X_test)
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        assert r2 > 0

    def test_bootstrap_methods(self, regression_data):
        """Test different bootstrap methods."""
        X, y = regression_data
        for method in ["bayesian", "classic", None]:
            reg = ConditionalInferenceForestRegressor(
                n_estimators=5,
                bootstrap_method=method,
                random_state=42,
                verbose=0,
                check_for_unused_parameters=False,
            )
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape


# Cross-validation Tests
class TestCrossValidation:
    """Test sklearn cross-validation compatibility."""

    def test_tree_classifier_cv(self, classification_data):
        """Test tree classifier with cross_val_score."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        scores = cross_val_score(clf, X, y, cv=3)
        assert len(scores) == 3
        assert scores.mean() > 0.5

    def test_tree_regressor_cv(self, regression_data):
        """Test tree regressor with cross_val_score."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(random_state=42, verbose=0)
        scores = cross_val_score(reg, X, y, cv=3)
        assert len(scores) == 3

    def test_forest_classifier_cv(self, classification_data):
        """Test forest classifier with cross_val_score."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, random_state=42, verbose=0)
        scores = cross_val_score(clf, X, y, cv=3)
        assert len(scores) == 3
        assert scores.mean() > 0.5

    def test_forest_regressor_cv(self, regression_data):
        """Test forest regressor with cross_val_score."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, random_state=42, verbose=0)
        scores = cross_val_score(reg, X, y, cv=3)
        assert len(scores) == 3


# Edge Cases
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_single_class(self):
        """Test with single class (should still work)."""
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert (preds == 0).all()

    def test_constant_feature(self, classification_data):
        """Test with constant feature."""
        X, y = classification_data
        X[:, 0] = 1.0  # Make first feature constant

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_high_dimensional(self):
        """Test with more features than samples."""
        X = np.random.randn(50, 100)
        y = np.random.randint(0, 2, 50)

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0, max_features="sqrt")
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_depth_1(self, classification_data):
        """Test with max_depth=1 (stump)."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_depth=1, random_state=42, verbose=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_small_sample(self):
        """Test with small sample size."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        clf = ConditionalInferenceTreeClassifier(random_state=42, verbose=0, min_samples_split=2)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape


# Parameter Validation Tests
class TestParameterValidation:
    """Test parameter validation."""

    def test_invalid_selector_classifier(self):
        """Test invalid selector for classifier."""
        with pytest.raises(ValueError):
            clf = ConditionalInferenceTreeClassifier(selector="invalid")

    def test_invalid_splitter_classifier(self):
        """Test invalid splitter for classifier."""
        with pytest.raises(ValueError):
            clf = ConditionalInferenceTreeClassifier(splitter="mse")  # mse is for regressor

    def test_invalid_selector_regressor(self):
        """Test invalid selector for regressor."""
        with pytest.raises(ValueError):
            reg = ConditionalInferenceTreeRegressor(selector="mc")  # mc is for classifier

    def test_invalid_alpha(self):
        """Test invalid alpha values."""
        with pytest.raises(ValueError):
            clf = ConditionalInferenceTreeClassifier(alpha_selector=0.0)

        with pytest.raises(ValueError):
            clf = ConditionalInferenceTreeClassifier(alpha_selector=1.5)


# Reproducibility Tests
class TestReproducibility:
    """Test reproducibility with random_state."""

    def test_tree_reproducibility(self, classification_data):
        """Test tree produces same results with same seed."""
        X, y = classification_data

        clf1 = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ConditionalInferenceTreeClassifier(random_state=42, verbose=0)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        assert np.array_equal(pred1, pred2)

    def test_forest_reproducibility(self, classification_data):
        """Test forest produces same results with same seed."""
        X, y = classification_data

        clf1 = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ConditionalInferenceForestClassifier(n_estimators=10, random_state=42, verbose=0)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        assert np.array_equal(pred1, pred2)
