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

# Fast parameters for integration tests - reduces permutation resamples significantly
FAST_PARAMS = {
    "n_resamples_selector": "minimum",  # 20 resamples instead of ~52+
    "n_resamples_splitter": "minimum",
    "verbose": 0,
    "random_state": 42,
}


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

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
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
        # Importances should sum to approximately 1 for a fitted tree with splits
        # (could be 0 if tree is trivial)

    def test_clone_and_refit(self, classification_data):
        """Test that cloning works correctly."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS, alpha_selector=0.1)
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
            clf = ConditionalInferenceTreeClassifier(selector=selector, **FAST_PARAMS)
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
            clf = ConditionalInferenceTreeClassifier(selector=selector, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_list_selector_no_ptest(self, classification_data):
        """Test list-based selector without permutation testing."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            selector=["mc", "rdc"], n_resamples_selector=None, n_resamples_splitter="minimum",
            verbose=0, random_state=42
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
            clf = ConditionalInferenceTreeClassifier(splitter=splitter, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape


# Tree Regressor Tests
class TestConditionalInferenceTreeRegressor:
    """Tests for ConditionalInferenceTreeRegressor."""

    def test_fit_predict(self, regression_data):
        """Test fit and predict on regression."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
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
            reg = ConditionalInferenceTreeRegressor(selector=selector, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_list_selector_all_combos(self, regression_data):
        """Test all valid list selector combinations for regression.

        All three selectors (pc, dc, rdc) are on [0,1] scale and can be combined.
        """
        X, y = regression_data
        # All valid 2-way and 3-way combinations
        for selector in [["pc", "dc"], ["pc", "rdc"], ["dc", "rdc"], ["pc", "dc", "rdc"]]:
            reg = ConditionalInferenceTreeRegressor(selector=selector, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_list_selector_no_ptest(self, regression_data):
        """Test list-based selector without permutation testing."""
        X, y = regression_data
        # Test all combinations without ptest
        for selector in [["pc", "dc"], ["pc", "rdc"], ["dc", "rdc"], ["pc", "dc", "rdc"]]:
            reg = ConditionalInferenceTreeRegressor(
                selector=selector, n_resamples_selector=None, n_resamples_splitter="minimum",
                verbose=0, random_state=42
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


# Forest Classifier Tests
class TestConditionalInferenceForestClassifier:
    """Tests for ConditionalInferenceForestClassifier."""

    def test_fit_predict_binary(self, classification_data):
        """Test fit and predict on binary classification."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = ConditionalInferenceForestClassifier(n_estimators=10, **FAST_PARAMS)
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
            clf = ConditionalInferenceForestClassifier(
                n_estimators=5,
                bootstrap_method=method,
                check_for_unused_parameters=False,
                **FAST_PARAMS,
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
                check_for_unused_parameters=False,
                **FAST_PARAMS,
            )
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_parallel_training(self, classification_data):
        """Test parallel training with n_jobs."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, n_jobs=2, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 10


# Forest Regressor Tests
class TestConditionalInferenceForestRegressor:
    """Tests for ConditionalInferenceForestRegressor."""

    def test_fit_predict(self, regression_data):
        """Test fit and predict on regression."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = ConditionalInferenceForestRegressor(n_estimators=10, **FAST_PARAMS)
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
                check_for_unused_parameters=False,
                **FAST_PARAMS,
            )
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape


# Cross-validation Tests
class TestCrossValidation:
    """Test sklearn cross-validation compatibility."""

    def test_tree_classifier_cv(self, classification_data):
        """Test tree classifier with cross_val_score."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        scores = cross_val_score(clf, X, y, cv=3)
        assert len(scores) == 3
        assert scores.mean() > 0.5

    def test_tree_regressor_cv(self, regression_data):
        """Test tree regressor with cross_val_score."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        scores = cross_val_score(reg, X, y, cv=3)
        assert len(scores) == 3

    def test_forest_classifier_cv(self, classification_data):
        """Test forest classifier with cross_val_score."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        scores = cross_val_score(clf, X, y, cv=3)
        assert len(scores) == 3
        assert scores.mean() > 0.5

    def test_forest_regressor_cv(self, regression_data):
        """Test forest regressor with cross_val_score."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)
        scores = cross_val_score(reg, X, y, cv=3)
        assert len(scores) == 3


# Edge Cases
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
        X[:, 0] = 1.0  # Make first feature constant

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

        clf1 = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        assert np.array_equal(pred1, pred2)

    def test_forest_reproducibility(self, classification_data):
        """Test forest produces same results with same seed."""
        X, y = classification_data

        clf1 = ConditionalInferenceForestClassifier(n_estimators=10, **FAST_PARAMS)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ConditionalInferenceForestClassifier(n_estimators=10, **FAST_PARAMS)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        assert np.array_equal(pred1, pred2)


# Statistical Correctness Tests (slower, uses full parameters)
class TestStatisticalCorrectness:
    """Test statistical correctness with full permutation testing parameters.

    These tests use 'auto' resamples to verify p-value computation works correctly.
    They are slower but ensure the statistical machinery is functioning properly.
    """

    def test_full_ptest(self, classification_data):
        """Test that full permutation testing produces valid p-values."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Use default 'auto' resamples for full statistical testing
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="auto",
            n_resamples_splitter="auto",
            random_state=42,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        # Verify tree was built and makes reasonable predictions
        assert hasattr(clf, "tree_")
        y_pred = clf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.5, f"Accuracy {accuracy} too low"


# Honesty Mode Tests
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


# Threshold Method Tests
class TestThresholdMethods:
    """Test different threshold generation methods."""

    def test_threshold_method_exact(self, classification_data):
        """Test exact threshold method (default)."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(threshold_method="exact", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_threshold_method_random(self, classification_data):
        """Test random threshold method."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            threshold_method="random", max_thresholds=10, **FAST_PARAMS
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_threshold_method_percentile(self, classification_data):
        """Test percentile threshold method."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            threshold_method="percentile", max_thresholds=10, **FAST_PARAMS
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_threshold_method_histogram(self, classification_data):
        """Test histogram threshold method."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            threshold_method="histogram", max_thresholds=10, **FAST_PARAMS
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_thresholds_int(self, classification_data):
        """Test max_thresholds as integer."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_thresholds=5, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_thresholds_sqrt(self, classification_data):
        """Test max_thresholds as 'sqrt'."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_thresholds="sqrt", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_thresholds_log2(self, classification_data):
        """Test max_thresholds as 'log2'."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_thresholds="log2", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_threshold_methods_regressor(self, regression_data):
        """Test threshold methods with regressor."""
        X, y = regression_data
        for method in ["exact", "random", "percentile", "histogram"]:
            reg = ConditionalInferenceTreeRegressor(
                threshold_method=method,
                max_thresholds=10 if method != "exact" else None,
                **FAST_PARAMS,
            )
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape


# Feature Control Tests
class TestFeatureControl:
    """Test feature selection control parameters."""

    def test_feature_muting_enabled(self, classification_data):
        """Test with feature muting enabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(feature_muting=True, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_feature_muting_disabled(self, classification_data):
        """Test with feature muting disabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(feature_muting=False, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_feature_scanning_enabled(self, classification_data):
        """Test with feature scanning enabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(feature_scanning=True, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_feature_scanning_disabled(self, classification_data):
        """Test with feature scanning disabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(feature_scanning=False, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_threshold_scanning_enabled(self, classification_data):
        """Test with threshold scanning enabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(threshold_scanning=True, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_threshold_scanning_disabled(self, classification_data):
        """Test with threshold scanning disabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(threshold_scanning=False, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_features_sqrt(self, classification_data):
        """Test max_features='sqrt'."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_features="sqrt", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_features_log2(self, classification_data):
        """Test max_features='log2'."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_features="log2", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_features_float(self, classification_data):
        """Test max_features as float (fraction of features)."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_features=0.5, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_features_int(self, classification_data):
        """Test max_features as integer."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(max_features=3, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_feature_controls_regressor(self, regression_data):
        """Test feature controls with regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(
            feature_muting=True,
            feature_scanning=True,
            max_features="sqrt",
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


# Early Stopping Tests
class TestEarlyStopping:
    """Test early stopping behavior in permutation tests."""

    def test_early_stopping_selector_enabled(self, classification_data):
        """Test with early stopping for selector enabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(early_stopping_selector="adaptive", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_early_stopping_selector_disabled(self, classification_data):
        """Test with early stopping for selector disabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(early_stopping_selector=None, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_early_stopping_splitter_enabled(self, classification_data):
        """Test with early stopping for splitter enabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(early_stopping_splitter="adaptive", **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_early_stopping_splitter_disabled(self, classification_data):
        """Test with early stopping for splitter disabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(early_stopping_splitter=None, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_both_early_stopping_disabled(self, classification_data):
        """Test with both early stopping disabled (rigorous mode)."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            early_stopping_selector=None,
            early_stopping_splitter=None,
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_early_stopping_regressor(self, regression_data):
        """Test early stopping with regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(
            early_stopping_selector=None,
            early_stopping_splitter=None,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


# Alpha and Bonferroni Correction Tests
class TestAlphaCorrection:
    """Test alpha levels and Bonferroni correction."""

    def test_bonferroni_selector_enabled(self, classification_data):
        """Test with Bonferroni correction for selector enabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(adjust_alpha_selector=True, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_bonferroni_selector_disabled(self, classification_data):
        """Test with Bonferroni correction for selector disabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(adjust_alpha_selector=False, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_bonferroni_splitter_enabled(self, classification_data):
        """Test with Bonferroni correction for splitter enabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(adjust_alpha_splitter=True, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_bonferroni_splitter_disabled(self, classification_data):
        """Test with Bonferroni correction for splitter disabled."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(adjust_alpha_splitter=False, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_different_alpha_selector(self, classification_data):
        """Test different alpha_selector values."""
        X, y = classification_data
        for alpha in [0.01, 0.05, 0.1, 0.2]:
            clf = ConditionalInferenceTreeClassifier(alpha_selector=alpha, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_different_alpha_splitter(self, classification_data):
        """Test different alpha_splitter values."""
        X, y = classification_data
        for alpha in [0.01, 0.05, 0.1, 0.2]:
            clf = ConditionalInferenceTreeClassifier(alpha_splitter=alpha, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_alpha_settings_regressor(self, regression_data):
        """Test alpha settings with regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(
            alpha_selector=0.1,
            alpha_splitter=0.1,
            adjust_alpha_selector=False,
            adjust_alpha_splitter=False,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


# Resamples Configuration Tests
class TestResamplesConfiguration:
    """Test different n_resamples configurations."""

    def test_n_resamples_minimum(self, classification_data):
        """Test n_resamples='minimum'."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_n_resamples_maximum(self, classification_data):
        """Test n_resamples='maximum' (slower but more accurate p-values)."""
        X, y = classification_data
        # Use smaller dataset for maximum resamples test
        X_small, y_small = X[:50], y[:50]
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="maximum",
            n_resamples_splitter="maximum",
            max_depth=2,  # Limit depth for speed
            verbose=0,
            random_state=42,
        )
        clf.fit(X_small, y_small)
        assert clf.predict(X_small).shape == y_small.shape

    def test_n_resamples_int(self, classification_data):
        """Test n_resamples as integer."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=50,
            n_resamples_splitter=50,
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_n_resamples_none(self, classification_data):
        """Test n_resamples=None (no permutation testing)."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_mixed_resamples(self, classification_data):
        """Test mixed resamples configuration."""
        X, y = classification_data
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="minimum",
            n_resamples_splitter=None,  # No splitter ptest
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape


# Forest Parallel Training Tests
class TestForestParallelTraining:
    """Test parallel training for forests."""

    def test_forest_classifier_parallel(self, classification_data):
        """Test forest classifier with parallel training."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, n_jobs=2, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 10
        assert clf.predict(X).shape == y.shape

    def test_forest_regressor_parallel(self, regression_data):
        """Test forest regressor with parallel training."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=10, n_jobs=2, **FAST_PARAMS)
        reg.fit(X, y)
        assert len(reg.estimators_) == 10
        assert reg.predict(X).shape == y.shape

    def test_forest_n_jobs_minus_one(self, classification_data):
        """Test forest with n_jobs=-1 (all cores)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, n_jobs=-1, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_forest_n_jobs_none(self, classification_data):
        """Test forest with n_jobs=None (sequential)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, n_jobs=None, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 5


# Tree Depth and Sample Control Tests
class TestTreeControl:
    """Test tree depth and sample control parameters."""

    def test_max_depth_values(self, classification_data):
        """Test various max_depth values."""
        X, y = classification_data
        for depth in [1, 2, 3, 5, 10, None]:
            clf = ConditionalInferenceTreeClassifier(max_depth=depth, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_min_samples_split_values(self, classification_data):
        """Test various min_samples_split values."""
        X, y = classification_data
        for min_split in [2, 5, 10, 20]:
            clf = ConditionalInferenceTreeClassifier(min_samples_split=min_split, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_min_samples_leaf_values(self, classification_data):
        """Test various min_samples_leaf values."""
        X, y = classification_data
        for min_leaf in [1, 2, 5, 10]:
            clf = ConditionalInferenceTreeClassifier(min_samples_leaf=min_leaf, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_min_impurity_decrease_values(self, classification_data):
        """Test various min_impurity_decrease values."""
        X, y = classification_data
        for min_impurity in [0.0, 0.01, 0.05, 0.1]:
            clf = ConditionalInferenceTreeClassifier(
                min_impurity_decrease=min_impurity, **FAST_PARAMS
            )
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_tree_control_regressor(self, regression_data):
        """Test tree control parameters with regressor."""
        X, y = regression_data
        reg = ConditionalInferenceTreeRegressor(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            min_impurity_decrease=0.01,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


# Forest Bootstrap and Sampling Tests
class TestForestSampling:
    """Test forest bootstrap and sampling configurations."""

    def test_max_samples_int(self, classification_data):
        """Test max_samples as integer."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5, max_samples=100, **FAST_PARAMS
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_max_samples_float(self, classification_data):
        """Test max_samples as float (fraction)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5, max_samples=0.8, **FAST_PARAMS
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_no_bootstrap(self, classification_data):
        """Test forest without bootstrap (uses full dataset)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            bootstrap_method=None,
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
