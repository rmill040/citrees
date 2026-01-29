"""Integration tests for ConditionalInferenceForest classifiers and regressors."""

import os

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
)

# Fast parameters for integration tests
FAST_PARAMS = {
    "n_resamples_selector": "minimum",
    "n_resamples_splitter": "minimum",
    "verbose": 0,
    "random_state": 42,
}


def _skip_if_loky_unavailable() -> None:
    """Skip test if loky backend is unavailable."""
    try:
        os.sysconf("SC_SEM_NSEMS_MAX")
    except PermissionError:
        pytest.skip("loky backend is unavailable in this environment")


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


class TestForestClassifier:
    """Tests for ConditionalInferenceForestClassifier."""

    def test_fit_predict_binary(self, classification_data):
        """Test fit and predict on binary classification."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceForestClassifier(n_estimators=10, **FAST_PARAMS)
        clf.fit(X_train, y_train)

        assert hasattr(clf, "estimators_")
        assert len(clf.estimators_) == 10
        assert hasattr(clf, "feature_importances_")

        y_pred = clf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.5

    def test_fit_predict_multiclass(self, multiclass_data):
        """Test fit and predict on multiclass classification."""
        X, y = multiclass_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, **FAST_PARAMS)
        clf.fit(X, y)

        assert clf.n_classes_ == 3
        y_pred = clf.predict(X)
        assert set(y_pred).issubset(set(y))

    def test_predict_proba(self, classification_data):
        """Test predict_proba returns valid probabilities."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, **FAST_PARAMS)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_bootstrap_methods(self, classification_data):
        """Test different bootstrap methods."""
        X, y = classification_data
        for method in ["bayesian", "classic", None]:
            kwargs = {
                "n_estimators": 5,
                "bootstrap_method": method,
                "check_for_unused_parameters": False,
                **FAST_PARAMS,
            }
            if method is None:
                kwargs["sampling_method"] = None
            clf = ConditionalInferenceForestClassifier(**kwargs)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_sampling_methods(self, classification_data):
        """Test different sampling methods."""
        X, y = classification_data
        for method in ["stratified", "undersample", "oversample", None]:
            clf = ConditionalInferenceForestClassifier(
                n_estimators=5,
                sampling_method=method,
                check_for_unused_parameters=False,
                **FAST_PARAMS,
            )
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_parallel_training(self, classification_data):
        """Test parallel training with n_jobs."""
        _skip_if_loky_unavailable()
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, n_jobs=2, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 10

    def test_n_jobs_minus_one(self, classification_data):
        """Test forest with n_jobs=-1 (all cores)."""
        _skip_if_loky_unavailable()
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, n_jobs=-1, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_n_jobs_none(self, classification_data):
        """Test forest with n_jobs=None (sequential)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, n_jobs=None, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 5


class TestForestRegressor:
    """Tests for ConditionalInferenceForestRegressor."""

    def test_fit_predict(self, regression_data):
        """Test fit and predict on regression."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = ConditionalInferenceForestRegressor(n_estimators=10, **FAST_PARAMS)
        reg.fit(X_train, y_train)

        assert hasattr(reg, "estimators_")
        assert len(reg.estimators_) == 10

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
                check_for_unused_parameters=False,
                **FAST_PARAMS,
            )
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_parallel_training(self, regression_data):
        """Test parallel training with n_jobs for regressor."""
        _skip_if_loky_unavailable()
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=10, n_jobs=2, **FAST_PARAMS)
        reg.fit(X, y)
        assert len(reg.estimators_) == 10
        assert reg.predict(X).shape == y.shape

    def test_feature_importances(self, regression_data):
        """Test feature importances for forest regressor."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)
        reg.fit(X, y)

        fi = reg.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert (fi >= 0).all()

    def test_max_samples_configurations(self, regression_data):
        """Test different max_samples configurations for regressor."""
        X, y = regression_data

        # Test with integer max_samples
        reg1 = ConditionalInferenceForestRegressor(n_estimators=5, max_samples=50, **FAST_PARAMS)
        reg1.fit(X, y)
        assert reg1.predict(X).shape == y.shape

        # Test with float max_samples
        reg2 = ConditionalInferenceForestRegressor(n_estimators=5, max_samples=0.8, **FAST_PARAMS)
        reg2.fit(X, y)
        assert reg2.predict(X).shape == y.shape

    def test_n_jobs_minus_one(self, regression_data):
        """Test forest regressor with n_jobs=-1 (all cores)."""
        _skip_if_loky_unavailable()
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, n_jobs=-1, **FAST_PARAMS)
        reg.fit(X, y)
        assert len(reg.estimators_) == 5


class TestForestSampling:
    """Test forest bootstrap and sampling configurations."""

    def test_max_samples_int(self, classification_data):
        """Test max_samples as integer."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, max_samples=100, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_samples_float(self, classification_data):
        """Test max_samples as float (fraction)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, max_samples=0.8, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_no_bootstrap(self, classification_data):
        """Test forest without bootstrap (uses full dataset)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            bootstrap_method=None,
            sampling_method=None,
            check_for_unused_parameters=False,
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_classic_bootstrap(self, classification_data):
        """Test forest with classic bootstrap."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5, bootstrap_method="classic", **FAST_PARAMS
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_bayesian_bootstrap(self, classification_data):
        """Test forest with Bayesian bootstrap."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5, bootstrap_method="bayesian", **FAST_PARAMS
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_regressor_sampling(self, regression_data):
        """Test regressor forest sampling configurations."""
        X, y = regression_data
        for method in ["bayesian", "classic", None]:
            reg = ConditionalInferenceForestRegressor(
                n_estimators=5,
                bootstrap_method=method,
                check_for_unused_parameters=False,
                **FAST_PARAMS,
            )
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape
