"""Tests for citrees._.tree.py."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.datasets import make_classification, make_regression

from citrees._tree import (
    BaseConditionalInferenceTreeParameters,
    ConditionalInferenceTreeClassifier,
    ConditionalInferenceTreeRegressor,
    Node,
)

# Fast parameters for unit tests
FAST_PARAMS = {
    "n_resamples_selector": "minimum",
    "n_resamples_splitter": "minimum",
    "verbose": 0,
    "random_state": 42,
}

pytestmark = pytest.mark.tree


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "feature": 0,
            "pval_feature": 0.5,
            "threshold": 2.5,
            "pval_threshold": 0.5,
            "impurity": 1.0,
            "left_child": (np.ones(6).reshape(3, 2), np.zeros(6)),
            "right_child": (np.ones(6).reshape(3, 2), np.zeros(6)),
            "n_samples": 12,
        },
        {"value": 0.5},
    ],
)
def test_node(kwargs: dict[str, Any]) -> None:
    """Test Node functionality."""
    node = Node(**kwargs)

    for key, value in node.items():
        assert value == kwargs[key]


def test_base_conditional_inference_tree_parameters():
    """Test BaseConditionalInferenceTreeParameters functionality."""
    # Failure
    with pytest.raises(ValidationError) as e:
        BaseConditionalInferenceTreeParameters()
    assert e.type is ValidationError, (
        f"Wrong exception, got ({e.type}) but expected ({ValidationError})"
    )

    # TODO: ADD HERE

    # Success
    params = BaseConditionalInferenceTreeParameters(
        estimator_type="classifier",
        selector="mc",
        splitter="gini",
        alpha_selector=0.05,
        alpha_splitter=0.05,
        adjust_alpha_selector=True,
        adjust_alpha_splitter=True,
        n_resamples_selector="auto",
        n_resamples_splitter="auto",
        early_stopping_selector="adaptive",
        early_stopping_splitter="adaptive",
        early_stopping_confidence_selector=0.95,
        early_stopping_confidence_splitter=0.95,
        feature_muting=True,
        feature_scanning=True,
        max_features=None,
        threshold_method="exact",
        threshold_scanning=True,
        max_thresholds=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        honesty=False,
        honesty_fraction=0.5,
        random_state=None,
        verbose=1,
        check_for_unused_parameters=False,
    )
    assert type(params) is BaseConditionalInferenceTreeParameters, (
        f"Wrong class, got ({type(params)}) but expected ({BaseConditionalInferenceTreeParameters})"
    )


class TestHonestEstimation:
    """Tests for honest estimation feature."""

    def test_classifier_honest_basic(self):
        """Test honest classifier can fit and predict."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        clf = ConditionalInferenceTreeClassifier(
            honesty=True,
            honesty_fraction=0.5,
            **FAST_PARAMS,
        )
        clf.fit(X, y)

        # Check predictions work
        preds = clf.predict(X)
        assert preds.shape == y.shape
        assert hasattr(clf, "tree_")

    def test_regressor_honest_basic(self):
        """Test honest regressor can fit and predict."""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        reg = ConditionalInferenceTreeRegressor(
            honesty=True,
            honesty_fraction=0.5,
            **FAST_PARAMS,
        )
        reg.fit(X, y)

        # Check predictions work
        preds = reg.predict(X)
        assert preds.shape == y.shape
        assert hasattr(reg, "tree_")

    def test_honest_vs_non_honest(self):
        """Test that honest and non-honest trees produce different results."""
        X, y = make_classification(n_samples=300, n_features=10, random_state=42)

        # Non-honest tree
        clf_regular = ConditionalInferenceTreeClassifier(
            honesty=False,
            **FAST_PARAMS,
        )
        clf_regular.fit(X, y)

        # Honest tree
        clf_honest = ConditionalInferenceTreeClassifier(
            honesty=True,
            honesty_fraction=0.5,
            **FAST_PARAMS,
        )
        clf_honest.fit(X, y)

        # Predictions should differ (honest uses estimation sample for leaf values)
        preds_regular = clf_regular.predict_proba(X)
        preds_honest = clf_honest.predict_proba(X)

        # Not all predictions should be the same
        assert not np.allclose(preds_regular, preds_honest)

    def test_honesty_fraction_validation(self):
        """Test that honesty_fraction is validated."""
        with pytest.raises(ValidationError):
            ConditionalInferenceTreeClassifier(
                honesty=True,
                honesty_fraction=0.0,  # Invalid: must be > 0
            )

        with pytest.raises(ValidationError):
            ConditionalInferenceTreeClassifier(
                honesty=True,
                honesty_fraction=1.0,  # Invalid: must be < 1
            )


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_selector_classifier(self):
        """Test invalid selector raises error for classifier."""
        with pytest.raises(ValueError, match="selector"):
            clf = ConditionalInferenceTreeClassifier(selector="invalid", **FAST_PARAMS)
            clf.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))

    def test_invalid_splitter_classifier(self):
        """Test invalid splitter raises error for classifier."""
        with pytest.raises(ValueError, match="splitter"):
            clf = ConditionalInferenceTreeClassifier(splitter="invalid", **FAST_PARAMS)
            clf.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))

    def test_empty_selector_list(self):
        """Test empty selector list raises error."""
        with pytest.raises(ValueError, match="empty"):
            clf = ConditionalInferenceTreeClassifier(selector=[], **FAST_PARAMS)
            clf.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))

    def test_mi_in_list_raises_error(self):
        """Test mi in selector list raises error for classifier."""
        with pytest.raises(ValueError, match="mi"):
            clf = ConditionalInferenceTreeClassifier(
                selector=["mc", "mi"], **FAST_PARAMS
            )
            clf.fit(np.random.randn(10, 5), np.random.randint(0, 2, 10))

    def test_n_resamples_too_low(self):
        """Test n_resamples below minimum raises error."""
        with pytest.raises(ValueError, match="n_resamples"):
            clf = ConditionalInferenceTreeClassifier(
                n_resamples_selector=5,  # Too low for alpha=0.05 (needs 20)
                alpha_selector=0.05,
                **{k: v for k, v in FAST_PARAMS.items() if k != "n_resamples_selector"},
            )
            clf.fit(np.random.randn(50, 5), np.random.randint(0, 2, 50))

    def test_invalid_alpha(self):
        """Test invalid alpha raises error."""
        with pytest.raises(ValidationError):
            ConditionalInferenceTreeClassifier(alpha_selector=0.0)

        with pytest.raises(ValidationError):
            ConditionalInferenceTreeClassifier(alpha_selector=1.5)


class TestTreeRepr:
    """Tests for tree string representation."""

    def test_repr(self):
        """Test __repr__ produces valid string."""
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        repr_str = repr(clf)
        assert "ConditionalInferenceTreeClassifier" in repr_str
        assert "random_state=42" in repr_str

    def test_str(self):
        """Test __str__ produces valid string."""
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        str_repr = str(clf)
        assert "ConditionalInferenceTreeClassifier" in str_repr


class TestTreeAttributes:
    """Tests for tree attributes after fitting."""

    def test_n_features_in(self):
        """Test n_features_in_ is set correctly."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert clf.n_features_in_ == 10

    def test_classes_attribute(self):
        """Test classes_ attribute is set for classifier."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2

    def test_tree_attribute(self):
        """Test tree_ attribute is set after fitting."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert hasattr(clf, "tree_")
        assert clf.tree_ is not None

    def test_feature_importances(self):
        """Test feature_importances_ is set correctly."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert hasattr(clf, "feature_importances_")
        assert clf.feature_importances_.shape == (5,)
        assert (clf.feature_importances_ >= 0).all()


class TestTreePredictions:
    """Tests for tree predictions."""

    def test_predict_before_fit_raises(self):
        """Test predict before fit raises error."""
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        with pytest.raises(Exception):  # NotFittedError or similar
            clf.predict(np.random.randn(10, 5))

    def test_predict_proba_sums_to_one(self):
        """Test predict_proba sums to 1."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_matches_proba(self):
        """Test predict matches argmax of predict_proba."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        preds_from_proba = clf.classes_[np.argmax(proba, axis=1)]
        assert np.array_equal(preds, preds_from_proba)


class TestRegressorSpecific:
    """Tests specific to regressor."""

    def test_regressor_predict(self):
        """Test regressor predict works."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape

    def test_regressor_selectors(self):
        """Test different regressor selectors."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        for selector in ["pc", "dc", "rdc"]:
            reg = ConditionalInferenceTreeRegressor(selector=selector, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_regressor_splitters(self):
        """Test different regressor splitters."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        for splitter in ["mse", "mae"]:
            reg = ConditionalInferenceTreeRegressor(splitter=splitter, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape


class TestClassifierSpecific:
    """Tests specific to classifier."""

    def test_classifier_selectors(self):
        """Test different classifier selectors."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        for selector in ["mc", "mi", "rdc"]:
            clf = ConditionalInferenceTreeClassifier(selector=selector, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_classifier_splitters(self):
        """Test different classifier splitters."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        for splitter in ["gini", "entropy"]:
            clf = ConditionalInferenceTreeClassifier(splitter=splitter, **FAST_PARAMS)
            clf.fit(X, y)
            assert clf.predict(X).shape == y.shape

    def test_multiclass(self):
        """Test classifier with multiclass."""
        X, y = make_classification(
            n_samples=150, n_features=5, n_classes=3,
            n_informative=3, n_clusters_per_class=1, random_state=42
        )
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        assert len(clf.classes_) == 3
        proba = clf.predict_proba(X)
        assert proba.shape == (150, 3)


class TestListSelector:
    """Tests for list-based selector."""

    def test_list_selector_classifier(self):
        """Test list selector for classifier."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(
            selector=["mc", "rdc"], **FAST_PARAMS
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_list_selector_regressor(self):
        """Test list selector for regressor."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        reg = ConditionalInferenceTreeRegressor(
            selector=["pc", "dc", "rdc"], **FAST_PARAMS
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


class TestNoPtest:
    """Tests with permutation testing disabled."""

    def test_no_selector_ptest(self):
        """Test with n_resamples_selector=None."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter="minimum",
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_no_splitter_ptest(self):
        """Test with n_resamples_splitter=None."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector="minimum",
            n_resamples_splitter=None,
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_no_ptest_at_all(self):
        """Test with both ptests disabled."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            verbose=0,
            random_state=42,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape
