"""Integration tests for parameter configurations and validation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score, train_test_split

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
    X, y = make_regression(
        n_samples=80,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return X, y


class TestParameterValidation:
    """Test parameter validation."""

    def test_invalid_selector_classifier(self):
        """Test invalid selector for classifier."""
        with pytest.raises(ValueError):
            ConditionalInferenceTreeClassifier(selector="invalid")

    def test_invalid_splitter_classifier(self):
        """Test invalid splitter for classifier."""
        with pytest.raises(ValueError):
            ConditionalInferenceTreeClassifier(splitter="mse")

    def test_invalid_selector_regressor(self):
        """Test invalid selector for regressor."""
        with pytest.raises(ValueError):
            ConditionalInferenceTreeRegressor(selector="mc")

    def test_invalid_alpha(self):
        """Test invalid alpha values."""
        with pytest.raises(ValueError):
            ConditionalInferenceTreeClassifier(alpha_selector=0.0)

        with pytest.raises(ValueError):
            ConditionalInferenceTreeClassifier(alpha_selector=1.5)


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
        for alpha in [0.05, 0.2]:
            clf = ConditionalInferenceTreeClassifier(alpha_selector=alpha, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_different_alpha_splitter(self, classification_data):
        """Test different alpha_splitter values."""
        X, y = classification_data
        for alpha in [0.05, 0.2]:
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

    @pytest.mark.slow
    def test_n_resamples_maximum(self, classification_data):
        """Test n_resamples='maximum' (slower but more accurate p-values)."""
        X, y = classification_data
        X_small, y_small = X[:50], y[:50]
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="maximum",
            n_resamples_splitter="maximum",
            max_depth=2,
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
            n_resamples_splitter=None,
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape


class TestTreeControl:
    """Test tree depth and sample control parameters."""

    def test_max_depth_values(self, classification_data):
        """Test various max_depth values."""
        X, y = classification_data
        for depth in [1, 5, None]:
            clf = ConditionalInferenceTreeClassifier(max_depth=depth, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_min_samples_split_values(self, classification_data):
        """Test various min_samples_split values."""
        X, y = classification_data
        for min_split in [2, 10]:
            clf = ConditionalInferenceTreeClassifier(min_samples_split=min_split, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_min_samples_leaf_values(self, classification_data):
        """Test various min_samples_leaf values."""
        X, y = classification_data
        for min_leaf in [1, 5]:
            clf = ConditionalInferenceTreeClassifier(min_samples_leaf=min_leaf, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_min_impurity_decrease_values(self, classification_data):
        """Test various min_impurity_decrease values."""
        X, y = classification_data
        for min_impurity in [0.0, 0.05]:
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
