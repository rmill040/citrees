"""Tests for citrees._forest.py."""

import os

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_classification, make_regression

from citrees import (
    ConditionalInferenceForestClassifier,
    ConditionalInferenceForestRegressor,
)

# Fast parameters for unit tests
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
        n_samples=100,
        n_features=5,
        n_informative=3,
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


class TestForestClassifierBasics:
    """Basic tests for ConditionalInferenceForestClassifier."""

    def test_estimators_created(self, classification_data):
        """Test that correct number of estimators are created."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 10

    def test_feature_importances_shape(self, classification_data):
        """Test feature importances have correct shape."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.feature_importances_.shape == (X.shape[1],)
        assert (clf.feature_importances_ >= 0).all()

    def test_classes_attribute(self, classification_data):
        """Test classes_ attribute is set."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf.fit(X, y)
        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2

    def test_n_features_in_attribute(self, classification_data):
        """Test n_features_in_ attribute is set."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.n_features_in_ == X.shape[1]


class TestForestLabelHandling:
    """Tests for classifier label handling in forests."""

    def test_string_labels(self):
        """Forest should preserve original string labels."""
        X = np.array([[0.0], [0.1], [0.2], [0.9], [1.0], [1.1]])
        y = np.array(["cat", "cat", "dog", "dog", "cat", "dog"], dtype=object)

        clf = ConditionalInferenceForestClassifier(
            n_estimators=3,
            bootstrap_method=None,
            sampling_method=None,
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        assert set(clf.classes_) == set(y)
        preds = clf.predict(X)
        assert set(preds).issubset(set(y))

    def test_nonzero_int_labels(self):
        """Forest should preserve non-zero-based integer labels."""
        X = np.array([[0.0], [0.1], [0.2], [0.9], [1.0], [1.1]])
        y = np.array([1, 1, 3, 3, 1, 3], dtype=np.int64)

        clf = ConditionalInferenceForestClassifier(
            n_estimators=3,
            bootstrap_method=None,
            sampling_method=None,
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        assert set(clf.classes_) == set(y)
        preds = clf.predict(X)
        assert set(preds).issubset(set(y))


class TestForestRegressorBasics:
    """Basic tests for ConditionalInferenceForestRegressor."""

    def test_estimators_created(self, regression_data):
        """Test that correct number of estimators are created."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=10, **FAST_PARAMS)
        reg.fit(X, y)
        assert len(reg.estimators_) == 10

    def test_feature_importances_shape(self, regression_data):
        """Test feature importances have correct shape."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)
        reg.fit(X, y)
        assert reg.feature_importances_.shape == (X.shape[1],)
        assert (reg.feature_importances_ >= 0).all()


class TestForestParameterValidation:
    """Tests for forest parameter validation."""

    def test_invalid_n_estimators(self):
        """Test that invalid n_estimators raises error."""
        with pytest.raises(ValidationError):
            ConditionalInferenceForestClassifier(n_estimators=0)

        with pytest.raises(ValidationError):
            ConditionalInferenceForestClassifier(n_estimators=-1)

    def test_invalid_bootstrap_method(self):
        """Test that invalid bootstrap_method raises error."""
        with pytest.raises(ValidationError):
            ConditionalInferenceForestClassifier(bootstrap_method="invalid")

    def test_invalid_sampling_method(self):
        """Test that invalid sampling_method raises error."""
        with pytest.raises(ValidationError):
            ConditionalInferenceForestClassifier(sampling_method="invalid")

    def test_regressor_check_for_unused_parameters(self):
        """Test regressor with check_for_unused_parameters=True and bootstrap_method=None.

        Previously, this raised KeyError because the regressor doesn't have
        sampling_method but _validate_parameter_combinations tried to access it.
        """
        # This should not raise KeyError
        reg = ConditionalInferenceForestRegressor(
            n_estimators=2,
            check_for_unused_parameters=True,
            bootstrap_method=None,
            random_state=42,
            verbose=0,
        )
        # Verify it can be used
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


class TestForestBootstrap:
    """Tests for forest bootstrap functionality."""

    def test_bayesian_bootstrap(self, classification_data):
        """Test Bayesian bootstrap."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            bootstrap_method="bayesian",
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_classic_bootstrap(self, classification_data):
        """Test classic bootstrap."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            bootstrap_method="classic",
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_no_bootstrap(self, classification_data):
        """Test without bootstrap."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            bootstrap_method=None,
            sampling_method=None,
            check_for_unused_parameters=False,
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 5


class TestForestSampling:
    """Tests for forest sampling methods."""

    def test_stratified_sampling(self, classification_data):
        """Test stratified sampling."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            sampling_method="stratified",
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    @pytest.mark.parametrize("sampling_method", ["undersample", "oversample"])
    def test_resampling_methods(self, classification_data, sampling_method: str) -> None:
        """Verify class-balancing sampling methods run end-to-end."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            sampling_method=sampling_method,
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_invalid_sampling_method_raises(self) -> None:
        """Verify invalid sampling_method values are rejected at init."""
        with pytest.raises(ValidationError):
            ConditionalInferenceForestClassifier(
                sampling_method="invalid",
                **FAST_PARAMS,
            )

    def test_sampling_method_requires_bootstrap(self) -> None:
        """sampling_method is only valid when bootstrap_method is enabled."""
        with pytest.raises(ValidationError):
            ConditionalInferenceForestClassifier(
                bootstrap_method=None,
                sampling_method="stratified",
                **FAST_PARAMS,
            )

    def test_max_samples_requires_bootstrap(self) -> None:
        """max_samples is only valid when bootstrap_method is enabled."""
        with pytest.raises(ValidationError):
            ConditionalInferenceForestClassifier(
                bootstrap_method=None,
                sampling_method=None,
                max_samples=0.8,
                **FAST_PARAMS,
            )


class TestForestMaxSamples:
    """Tests for max_samples parameter."""

    def test_max_samples_int(self, classification_data):
        """Test max_samples as integer."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            max_samples=50,
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_max_samples_float(self, classification_data):
        """Test max_samples as float fraction."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            max_samples=0.5,
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 5


class TestForestPredictions:
    """Tests for forest predictions."""

    def test_predict_shape_classifier(self, classification_data):
        """Test predict returns correct shape for classifier."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape

    def test_predict_proba_shape(self, classification_data):
        """Test predict_proba returns correct shape."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_shape_regressor(self, regression_data):
        """Test predict returns correct shape for regressor."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(n_estimators=5, **FAST_PARAMS)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape


class TestForestFeatureNameValidation:
    """Tests for forest feature name validation behavior."""

    def test_feature_name_order_validation_raises(self):
        """Test that reordered columns raise an error."""
        X, y = make_classification(
            n_samples=80,
            n_features=3,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=["a", "b", "c"])

        clf = ConditionalInferenceForestClassifier(n_estimators=3, **FAST_PARAMS)
        clf.fit(X_df, y)

        X_reordered = X_df[["b", "a", "c"]]
        with pytest.raises(ValueError, match="out of order"):
            clf.predict(X_reordered)


class TestForestPaths:
    """Tests for forest apply/decision_path."""

    def test_apply_and_decision_path_classifier(self, classification_data):
        """apply and decision_path should return sklearn-compatible shapes."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=4,
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        leaf_ids = clf.apply(X)
        assert leaf_ids.shape == (len(X), clf.n_estimators)

        indicator, n_nodes_ptr = clf.decision_path(X)
        assert indicator.shape[0] == len(X)
        assert n_nodes_ptr.shape == (clf.n_estimators + 1,)
        assert n_nodes_ptr[0] == 0
        assert (np.diff(n_nodes_ptr) > 0).all()
        assert indicator.shape[1] == n_nodes_ptr[-1]

        for i in range(clf.n_estimators):
            n_nodes_tree = n_nodes_ptr[i + 1] - n_nodes_ptr[i]
            assert leaf_ids[:, i].max() < n_nodes_tree


class TestForestPredictProbaAlignment:
    """Tests for predict_proba with missing classes in some trees."""

    def test_predict_proba_handles_single_class_trees(self):
        """predict_proba should align per-tree outputs to global classes."""
        X = np.arange(40, dtype=float).reshape(-1, 1)
        y = np.array([0] * 30 + [1] * 10, dtype=int)

        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            bootstrap_method="classic",
            sampling_method=None,
            max_samples=1,
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        assert any(len(est.classes_) == 1 for est in clf.estimators_)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestForestOOB:
    """Tests for out-of-bag scoring."""

    def test_oob_score_requires_bootstrap(self, classification_data):
        """oob_score should error when bootstrap is disabled."""
        with pytest.raises(ValidationError, match="oob_score"):
            ConditionalInferenceForestClassifier(
                n_estimators=5,
                bootstrap_method=None,
                sampling_method=None,
                oob_score=True,
                n_resamples_selector=None,
                n_resamples_splitter=None,
                random_state=42,
                verbose=0,
            )

    def test_oob_score_classifier_attributes(self, classification_data):
        """oob_score should populate classifier OOB attributes."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=6,
            bootstrap_method="classic",
            sampling_method=None,
            oob_score=True,
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        assert hasattr(clf, "oob_score_")
        assert hasattr(clf, "oob_decision_function_")
        assert clf.oob_decision_function_.shape == (len(X), len(clf.classes_))

    def test_oob_score_regressor_attributes(self, regression_data):
        """oob_score should populate regressor OOB attributes."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(
            n_estimators=6,
            bootstrap_method="classic",
            oob_score=True,
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        reg.fit(X, y)

        assert hasattr(reg, "oob_score_")
        assert hasattr(reg, "oob_prediction_")
        assert reg.oob_prediction_.shape == (len(X),)


class TestForestReproducibility:
    """Tests for forest reproducibility."""

    def test_same_seed_same_results(self, classification_data):
        """Test that same seed produces same results."""
        X, y = classification_data

        clf1 = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf1.fit(X, y)
        pred1 = clf1.predict(X)

        clf2 = ConditionalInferenceForestClassifier(n_estimators=5, **FAST_PARAMS)
        clf2.fit(X, y)
        pred2 = clf2.predict(X)

        assert np.array_equal(pred1, pred2)

    def test_different_seed_different_results(self, classification_data):
        """Test that different seeds produce different results (usually)."""
        X, y = classification_data

        params1 = {**FAST_PARAMS, "random_state": 42}
        params2 = {**FAST_PARAMS, "random_state": 123}

        clf1 = ConditionalInferenceForestClassifier(n_estimators=5, **params1)
        clf1.fit(X, y)

        clf2 = ConditionalInferenceForestClassifier(n_estimators=5, **params2)
        clf2.fit(X, y)

        # At least one estimator should differ
        trees_differ = False
        for t1, t2 in zip(clf1.estimators_, clf2.estimators_, strict=False):
            if t1.random_state != t2.random_state:
                trees_differ = True
                break
        assert trees_differ


class TestForestEstimatorRandomStates:
    """Tests for individual estimator random states."""

    def test_estimators_have_unique_random_states(self, classification_data):
        """Test that each estimator has a unique random state."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=10, **FAST_PARAMS)
        clf.fit(X, y)

        random_states = [est.random_state for est in clf.estimators_]
        # All should be unique
        assert len(random_states) == len(set(random_states))

    def test_estimators_random_states_sequential(self, classification_data):
        """Test that estimator random states are sequential from base."""
        X, y = classification_data
        base_seed = 42
        params = {**FAST_PARAMS, "random_state": base_seed}
        clf = ConditionalInferenceForestClassifier(n_estimators=5, **params)
        clf.fit(X, y)

        for i, est in enumerate(clf.estimators_):
            assert est.random_state == base_seed + i


class TestForestNJobs:
    """Tests for n_jobs parameter."""

    def test_n_jobs_none(self, classification_data):
        """Test n_jobs=None (sequential)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, n_jobs=None, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_n_jobs_1(self, classification_data):
        """Test n_jobs=1 (single core)."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, n_jobs=1, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 5

    def test_n_jobs_2(self, classification_data):
        """Test n_jobs=2 (two cores)."""
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except PermissionError:
            pytest.skip("loky backend is unavailable in this environment")
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(n_estimators=5, n_jobs=2, **FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.estimators_) == 5


class TestForestHonesty:
    """Tests for honest forest estimation."""

    def test_honest_forest_classifier(self, classification_data):
        """Test honest forest classifier."""
        X, y = classification_data
        clf = ConditionalInferenceForestClassifier(
            n_estimators=5,
            honesty=True,
            honesty_fraction=0.5,
            **FAST_PARAMS,
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 5
        preds = clf.predict(X)
        assert preds.shape == y.shape

    def test_honest_forest_regressor(self, regression_data):
        """Test honest forest regressor."""
        X, y = regression_data
        reg = ConditionalInferenceForestRegressor(
            n_estimators=5,
            honesty=True,
            honesty_fraction=0.5,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert len(reg.estimators_) == 5
        preds = reg.predict(X)
        assert preds.shape == y.shape
