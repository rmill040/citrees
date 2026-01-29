"""Integration tests for edge cases, boundary conditions, and special modes."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
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
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=80,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return X, y


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_single_class(self):
        """Test with single class (should still work)."""
        X = np.random.randn(100, 5)
        y = np.zeros(100, dtype=int)

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert (preds == 0).all()

    def test_constant_feature(self, classification_data):
        """Test with constant feature."""
        X, y = classification_data
        X[:, 0] = 1.0

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_high_dimensional(self):
        """Test with more features than samples."""
        X = np.random.randn(50, 100)
        y = np.random.randint(0, 2, 50)

        clf = ConditionalInferenceTreeClassifier(max_features="sqrt", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_depth_1(self, classification_data):
        """Test with max_depth=1 (stump)."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_depth=1, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_small_sample(self):
        """Test with small sample size."""
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        clf = ConditionalInferenceTreeClassifier(min_samples_split=2, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape


class TestStatisticalCorrectness:
    """Test statistical correctness with full permutation testing parameters."""

    @pytest.mark.slow
    def test_full_ptest(self, classification_data):
        """Test that full permutation testing produces valid p-values."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="auto",
            n_resamples_splitter="auto",
            random_state=42,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        assert hasattr(clf, "tree_")
        y_pred = clf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.5, f"Accuracy {accuracy} too low"


class TestHonestyMode:
    """Test honest estimation (sample splitting for unbiased leaf estimates)."""

    def test_honest_tree_classifier(self, classification_data):
        """Test honest tree classifier."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceTreeClassifier(honesty=True, honesty_fraction=0.5, **FAST_PARAMS)
        clf.fit(X_train, y_train)

        assert hasattr(clf, "tree_")
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

    def test_honest_tree_regressor(self, regression_data):
        """Test honest tree regressor."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = ConditionalInferenceTreeRegressor(honesty=True, honesty_fraction=0.5, **FAST_PARAMS)
        reg.fit(X_train, y_train)

        assert hasattr(reg, "tree_")
        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape

    def test_honest_forest_classifier(self, classification_data):
        """Test honest forest classifier."""
        X, y = classification_data

        clf = ConditionalInferenceForestClassifier(
            n_estimators=5, honesty=True, honesty_fraction=0.5, **FAST_PARAMS
        )
        clf.fit(X, y)

        assert len(clf.estimators_) == 5
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape

    def test_honest_forest_regressor(self, regression_data):
        """Test honest forest regressor."""
        X, y = regression_data

        reg = ConditionalInferenceForestRegressor(
            n_estimators=5, honesty=True, honesty_fraction=0.5, **FAST_PARAMS
        )
        reg.fit(X, y)

        assert len(reg.estimators_) == 5
        y_pred = reg.predict(X)
        assert y_pred.shape == y.shape

    def test_honesty_fraction_values(self, classification_data):
        """Test different honesty_fraction values."""
        X, y = classification_data

        for fraction in [0.3, 0.5, 0.7]:
            clf = ConditionalInferenceTreeClassifier(
                honesty=True, honesty_fraction=fraction, **FAST_PARAMS
            )
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape
