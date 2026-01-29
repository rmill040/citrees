"""Integration tests for ConditionalInferenceTree classifiers and regressors."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from citrees import (
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
)

# Fast parameters for integration tests
FAST_PARAMS = {
    "n_resamples_selector": "minimum",
    "n_resamples_splitter": "minimum",
    "verbose": 0,
    "random_state": 42,
}


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=80,
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
        n_samples=80,
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
        n_samples=120,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X, y


class TestTreeClassifier:
    """Tests for ConditionalInferenceTreeClassifier."""

    def test_fit_predict_binary(self, classification_data):
        """Test fit and predict on binary classification."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X_train, y_train)

        assert hasattr(clf, "n_features_in_")
        assert clf.n_features_in_ == X.shape[1]
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "feature_importances_")

        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
        assert set(y_pred).issubset(set(y))

        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.5, f"Accuracy {accuracy} too low"

    def test_fit_predict_multiclass(self, multiclass_data):
        """Test fit and predict on multiclass classification."""
        X, y = multiclass_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X_train, y_train)

        assert len(clf.classes_) == 3
        y_pred = clf.predict(X_test)
        assert set(y_pred).issubset(set(y))

    def test_predict_proba(self, classification_data):
        """Test predict_proba returns valid probabilities."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_feature_importances(self, classification_data):
        """Test feature importances are computed."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)

        fi = clf.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert (fi >= 0).all()

    def test_clone_and_refit(self, classification_data):
        """Test that cloning works correctly."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS, alpha_selector=0.1)
        clf.fit(X, y)

        clf2 = clone(clf)
        assert clf2.alpha_selector == 0.1
        assert not hasattr(clf2, "classes_")

        clf2.fit(X, y)
        assert hasattr(clf2, "classes_")

    def test_selector_methods(self, classification_data):
        """Test all individual selector methods."""
        X, y = classification_data
        for selector in ["mc", "mi", "rdc"]:
            clf = ConditionalInferenceTreeClassifier(selector=selector, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_list_selector(self, classification_data):
        """Test list selector combinations for classification."""
        X, y = classification_data
        # Only mc and rdc can be combined (both on [0,1] scale)
        clf = ConditionalInferenceTreeClassifier(selector=["mc", "rdc"], **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_list_selector_no_ptest(self, classification_data):
        """Test list-based selector without permutation testing."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            selector=["mc", "rdc"],
            n_resamples_selector=None,
            n_resamples_splitter="minimum",
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_list_selector_mi_invalid(self, classification_data):
        """Test that mi cannot be in a list selector for classification."""
        X, y = classification_data
        with pytest.raises(ValueError, match="mi.*cannot be used in a list"):
            ConditionalInferenceTreeClassifier(selector=["mc", "mi"], random_state=42, verbose=0)

    def test_splitter_methods(self, classification_data):
        """Test different splitter methods."""
        X, y = classification_data
        for splitter in ["gini", "entropy"]:
            clf = ConditionalInferenceTreeClassifier(splitter=splitter, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape


class TestTreeRegressor:
    """Tests for ConditionalInferenceTreeRegressor."""

    def test_fit_predict(self, regression_data):
        """Test fit and predict on regression."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X_train, y_train)

        assert hasattr(reg, "n_features_in_")
        assert reg.n_features_in_ == X.shape[1]
        assert hasattr(reg, "feature_importances_")

        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape

        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        assert r2 > -0.5, f"R2 {r2} is too negative"

    def test_selector_methods(self, regression_data):
        """Test all individual selector methods."""
        X, y = regression_data
        for selector in ["pc", "dc", "rdc"]:
            reg = ConditionalInferenceTreeRegressor(selector=selector, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_list_selector_all_combos(self, regression_data):
        """Test all valid list selector combinations for regression."""
        X, y = regression_data
        for selector in [["pc", "dc"], ["pc", "rdc"], ["dc", "rdc"], ["pc", "dc", "rdc"]]:
            reg = ConditionalInferenceTreeRegressor(selector=selector, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_list_selector_no_ptest(self, regression_data):
        """Test list-based selector without permutation testing."""
        X, y = regression_data
        for selector in [["pc", "dc"], ["pc", "rdc"], ["dc", "rdc"], ["pc", "dc", "rdc"]]:
            reg = ConditionalInferenceTreeRegressor(
                selector=selector,
                n_resamples_selector=None,
                n_resamples_splitter="minimum",
                verbose=0,
                random_state=42,
            )
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_splitter_methods(self, regression_data):
        """Test different splitter methods."""
        X, y = regression_data
        for splitter in ["mse", "mae"]:
            reg = ConditionalInferenceTreeRegressor(splitter=splitter, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_feature_importances(self, regression_data):
        """Test feature importances are computed for regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)

        fi = reg.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert (fi >= 0).all()

    def test_clone_and_refit(self, regression_data):
        """Test that cloning works correctly for regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS, alpha_selector=0.1)
        reg.fit(X, y)

        reg2 = clone(reg)
        assert reg2.alpha_selector == 0.1
        assert not hasattr(reg2, "tree_")

        reg2.fit(X, y)
        assert hasattr(reg2, "tree_")

    def test_apply_and_decision_path(self, regression_data):
        """Test apply and decision_path for regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)

        leaf_ids = reg.apply(X)
        assert leaf_ids.shape == (len(X),)

        path = reg.decision_path(X)
        assert path.shape[0] == len(X)
        assert leaf_ids.max() < path.shape[1]
