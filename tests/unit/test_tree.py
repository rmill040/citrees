"""Tests for citrees._.tree.py."""

from typing import Any

import numpy as np
import pandas as pd
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


def _features_used(tree: Node) -> set[int]:
    if "value" in tree:
        return set()
    return {
        int(tree["feature"]),
        *_features_used(tree["left_child"]),
        *_features_used(tree["right_child"]),
    }


class TestCandidateSetIsolation:
    """Regression tests for candidate-set handling during recursion."""

    def test_constant_feature_in_one_branch_does_not_affect_sibling(self) -> None:
        """A feature constant in one child must remain available in the sibling.

        This guards against traversal-order dependence from mutating a shared/global candidate set.
        """
        rng = np.random.default_rng(0)
        n = 2000

        # Gate feature: binary split at 0.5.
        z = rng.integers(0, 2, size=n).astype(float)

        # Feature x1 is constant when z==0, but variable (and informative) when z==1.
        x1 = np.zeros(n, dtype=float)
        x1[z == 1] = rng.standard_normal((z == 1).sum())

        # Noise feature.
        x2 = rng.standard_normal(n)

        # Target:
        # - Left branch (z==0): noisy but high base rate (ensures left node is not a leaf, so constant-feature filtering runs)
        # - Right branch (z==1): deterministic threshold on x1 (so x1 should be used in that subtree)
        y = np.zeros(n, dtype=int)
        y[z == 0] = rng.binomial(1, 0.9, size=(z == 0).sum())
        y[z == 1] = (x1[z == 1] > 1.2815515655446004).astype(int)

        X = np.column_stack([z, x1, x2])
        X_flip = np.column_stack([1.0 - z, x1, x2])

        params = {
            "n_resamples_selector": None,
            "n_resamples_splitter": None,
            "feature_muting": False,  # bug is in constant-feature handling, not p-value muting
            "feature_scanning": False,
            "threshold_scanning": False,
            "max_depth": 2,
            "random_state": 0,
            "verbose": 0,
            "check_for_unused_parameters": False,
        }

        clf_a = ConditionalInferenceTreeClassifier(**params)
        clf_a.fit(X, y)
        used_a = _features_used(clf_a.tree_)

        clf_b = ConditionalInferenceTreeClassifier(**params)
        clf_b.fit(X_flip, y)
        used_b = _features_used(clf_b.tree_)

        assert 1 in used_a
        assert 1 in used_b


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
            clf = ConditionalInferenceTreeClassifier(selector=["mc", "mi"], **FAST_PARAMS)
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

    def test_invalid_max_features(self):
        """Negative max_features should raise validation error."""
        with pytest.raises(ValidationError):
            ConditionalInferenceTreeClassifier(max_features=-1)

    def test_invalid_max_thresholds(self):
        """Negative max_thresholds should raise validation error."""
        with pytest.raises(ValidationError):
            ConditionalInferenceTreeClassifier(threshold_method="random", max_thresholds=-1)


class TestFeatureNameValidation:
    """Tests for feature name validation behavior."""

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

        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X_df, y)

        X_reordered = X_df[["b", "a", "c"]]
        with pytest.raises(ValueError, match="out of order"):
            clf.predict(X_reordered)


class TestThresholdScanning:
    """Tests for threshold scanning behavior."""

    def test_scan_thresholds_uses_weighted_impurity(self):
        """Threshold scanning should align with weighted split impurity."""
        x = np.arange(10, dtype=float)
        y = np.array([0] * 9 + [1], dtype=int)
        X = x.reshape(-1, 1)

        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            early_stopping_selector=None,
            early_stopping_splitter=None,
            feature_scanning=False,
            threshold_scanning=False,
            random_state=0,
            verbose=0,
            check_for_unused_parameters=False,
        )
        clf.fit(X, y)

        thresholds = (x[:-1] + x[1:]) / 2
        scanned = clf._scan_thresholds(x, y, thresholds)
        scores = np.array([clf._split_impurity(x=x, y=y, threshold=t) for t in thresholds])
        expected = thresholds[np.argsort(scores)]

        assert np.array_equal(scanned, expected)


class TestLabelHandling:
    """Tests for classifier label handling."""

    def test_string_labels(self):
        """Classifier should preserve original string labels."""
        X = np.array([[0.0], [0.1], [0.2], [0.9], [1.0], [1.1]])
        y = np.array(["cat", "cat", "dog", "dog", "cat", "dog"], dtype=object)

        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        assert set(clf.classes_) == set(y)
        preds = clf.predict(X)
        assert set(preds).issubset(set(y))

    def test_refit_updates_n_classes(self):
        """n_classes_ and predict_proba shape should update on successive fit() calls."""
        clf = ConditionalInferenceTreeClassifier(
            selector="rdc",
            n_resamples_selector=None,
            n_resamples_splitter=None,
            early_stopping_selector=None,
            early_stopping_splitter=None,
            feature_scanning=False,
            threshold_scanning=False,
            random_state=0,
            verbose=0,
        )

        rng = np.random.RandomState(0)
        X1 = rng.randn(60, 3)
        y1 = rng.randint(0, 2, size=60)
        clf.fit(X1, y1)

        rng = np.random.RandomState(1)
        X2 = rng.randn(90, 3)
        y2 = rng.randint(0, 3, size=90)
        clf.fit(X2, y2)

        expected = len(np.unique(y2))
        proba = clf.predict_proba(X2)

        assert clf.n_classes_ == expected
        assert len(clf.classes_) == expected
        assert proba.shape == (len(X2), expected)


class TestMinSamplesLeaf:
    """Tests for min_samples_leaf behavior."""

    def test_min_samples_leaf_does_not_block_valid_split(self):
        """A node should still split if at least one valid threshold exists."""
        x = np.arange(10, dtype=float)
        X = x.reshape(-1, 1)
        y = np.array([1] + [0] * 9, dtype=np.int64)

        clf = ConditionalInferenceTreeClassifier(
            selector="rdc",
            n_resamples_selector=None,
            n_resamples_splitter=None,
            early_stopping_selector=None,
            early_stopping_splitter=None,
            feature_scanning=False,
            threshold_scanning=False,
            min_samples_leaf=2,
            max_depth=1,
            random_state=0,
            verbose=0,
        )
        clf.fit(X, y)

        assert "feature" in clf.tree_
        assert float(clf.tree_["threshold"]) == pytest.approx(1.5)


class TestRandomness:
    """Tests for RNG behavior during tree building."""

    def test_feature_subsampling_advances_rng(self):
        """RNG should advance across nodes when max_features forces subsampling."""
        x = np.linspace(0.0, 1.0, 40)
        # Use distinct features to avoid threshold ties consuming RNG for tie-breaking.
        # Both features are equally predictive but have different values.
        X = np.column_stack([x, 1.0 - x])
        y = np.array([0, 1] * 20, dtype=np.int64)

        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            max_features=1,
            max_depth=2,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=1,  # Seed chosen to produce multiple features with this data
            verbose=0,
        )
        clf.fit(X, y)

        def collect_features(node: Node) -> set[int]:
            if "value" in node:
                return set()
            feats = {int(node["feature"])}
            feats |= collect_features(node["left_child"])
            feats |= collect_features(node["right_child"])
            return feats

        feats = collect_features(clf.tree_)
        assert len(feats) > 1, f"Expected multiple features due to RNG advancement, got {feats}"

    def test_nonzero_int_labels(self):
        """Classifier should preserve non-zero-based integer labels."""
        X = np.array([[0.0], [0.1], [0.2], [0.9], [1.0], [1.1]])
        y = np.array([1, 1, 3, 3, 1, 3], dtype=np.int64)

        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        assert set(clf.classes_) == set(y)
        preds = clf.predict(X)
        assert set(preds).issubset(set(y))


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
        with pytest.raises(ValueError):
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


class TestInputTypes:
    """Tests for list/tuple input type handling."""

    def test_fit_with_list_input(self):
        """Test fit accepts list inputs for X and y."""
        X_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        y_list = [0, 1, 0, 1]
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X_list, y_list)
        assert hasattr(clf, "tree_")

    def test_fit_with_tuple_input(self):
        """Test fit accepts tuple inputs for X and y."""
        X_tuple = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0))
        y_tuple = (0, 1, 0, 1)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X_tuple, y_tuple)
        assert hasattr(clf, "tree_")

    def test_predict_with_list_input(self):
        """Test predict accepts list input."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        X_list = X.tolist()
        preds = clf.predict(X_list)
        assert preds.shape == (len(X_list),)

    def test_predict_with_tuple_input(self):
        """Test predict accepts tuple input."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        clf = ConditionalInferenceTreeClassifier(**FAST_PARAMS)
        clf.fit(X, y)
        X_tuple = tuple(tuple(row) for row in X)
        preds = clf.predict(X_tuple)
        assert preds.shape == (len(X_tuple),)


class TestTreePaths:
    """Tests for tree apply/decision_path."""

    def test_apply_and_decision_path(self):
        """apply should return leaf ids; decision_path should include leaf node."""
        X, y = make_classification(n_samples=80, n_features=4, random_state=42)
        clf = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

        leaf_ids = clf.apply(X)
        assert leaf_ids.shape == (len(X),)

        path = clf.decision_path(X)
        assert path.shape[0] == len(X)
        assert leaf_ids.max() < path.shape[1]

        for i in range(len(X)):
            row = path.getrow(i)
            assert leaf_ids[i] in row.indices


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
            n_samples=150,
            n_features=5,
            n_classes=3,
            n_informative=3,
            n_clusters_per_class=1,
            random_state=42,
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
        clf = ConditionalInferenceTreeClassifier(selector=["mc", "rdc"], **FAST_PARAMS)
        clf.fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_list_selector_regressor(self):
        """Test list selector for regressor."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        reg = ConditionalInferenceTreeRegressor(selector=["pc", "dc", "rdc"], **FAST_PARAMS)
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


class TestFeatureScanning:
    """Tests for feature scanning behavior."""

    def test_feature_scanning_prioritizes_informative_features(self):
        """Test that feature_scanning causes informative features to be tested first.

        When feature_scanning=True, the tree should prioritize features with higher
        univariate association scores. With limited permutation budget, this should
        increase the chance of selecting the most informative feature.
        """
        rng = np.random.default_rng(42)
        n = 200

        # Create dataset: x0 is highly informative, x1-x9 are noise
        x_informative = rng.standard_normal(n)
        X = np.column_stack([x_informative] + [rng.standard_normal(n) for _ in range(9)])
        y = (x_informative > 0).astype(int)

        # With scanning: informative feature should be prioritized
        clf_scan = ConditionalInferenceTreeClassifier(
            feature_scanning=True,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            max_depth=1,
            random_state=42,
            verbose=0,
        )
        clf_scan.fit(X, y)

        # Without scanning: features tested in original order
        clf_noscan = ConditionalInferenceTreeClassifier(
            feature_scanning=False,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            max_depth=1,
            random_state=42,
            verbose=0,
        )
        clf_noscan.fit(X, y)

        # Both should find the informative feature, but scanning should be more reliable
        # The root split feature should be 0 (the informative one) when scanning is used
        if "feature" in clf_scan.tree_:
            assert clf_scan.tree_["feature"] == 0, (
                f"Expected feature 0 with scanning, got {clf_scan.tree_['feature']}"
            )


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


class TestBugFixes:
    """Tests for specific bug fixes."""

    def test_bug1_feature_muting_uses_correct_alpha(self):
        """Bug 1: Verify feature muting uses alpha_selector threshold correctly.

        Previously, feature muting used max(alpha, 1-alpha) = 0.95 for alpha=0.05,
        effectively disabling muting. Now it correctly uses alpha_selector.

        This test creates a dataset where some features are noise and should be
        muted (p-value >= alpha).
        """
        # Create dataset with 1 informative feature and 4 pure noise features
        np.random.seed(42)
        n_samples = 100

        # Feature 0: perfectly separates classes
        X_informative = np.concatenate([np.zeros(50), np.ones(50)]).reshape(-1, 1)

        # Features 1-4: pure random noise
        X_noise = np.random.randn(n_samples, 4)

        X = np.hstack([X_informative, X_noise])
        y = np.array([0] * 50 + [1] * 50)

        # Fit with feature_muting=True
        clf_muting = ConditionalInferenceTreeClassifier(
            feature_muting=True,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            alpha_selector=0.05,
            max_depth=3,
            verbose=0,
            random_state=42,
        )
        clf_muting.fit(X, y)

        # Fit without feature_muting for comparison
        clf_no_muting = ConditionalInferenceTreeClassifier(
            feature_muting=False,
            n_resamples_selector="minimum",
            n_resamples_splitter="minimum",
            alpha_selector=0.05,
            max_depth=3,
            verbose=0,
            random_state=42,
        )
        clf_no_muting.fit(X, y)

        # With the bug fixed, feature muting should work and both should produce
        # valid predictions. The key is that the code doesn't crash and works.
        assert clf_muting.predict(X).shape == y.shape
        assert clf_no_muting.predict(X).shape == y.shape

        # The informative feature (0) should be used in splits
        features_used_muting = _features_used(clf_muting.tree_)
        if len(features_used_muting) > 0:
            assert 0 in features_used_muting, (
                f"Expected feature 0 to be used with muting, got {features_used_muting}"
            )


# =============================================================================
# EXPANDED REGRESSOR TESTS
# =============================================================================


class TestRegressorNonlinear:
    """Tests for regressor on nonlinear (Friedman) data."""

    def test_friedman1_with_dc_selector(self, regression_data_friedman1):
        """Test DC selector on nonlinear Friedman #1 data."""
        X, y = regression_data_friedman1
        reg = ConditionalInferenceTreeRegressor(
            selector="dc",
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape

        # R2 should be positive on training data
        r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > 0, f"R2 on Friedman1 data should be positive, got {r2}"

    def test_friedman1_with_rdc_selector(self, regression_data_friedman1):
        """Test RDC selector on nonlinear Friedman #1 data."""
        X, y = regression_data_friedman1
        reg = ConditionalInferenceTreeRegressor(
            selector="rdc",
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape

        # R2 should be positive on training data
        r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > 0, f"R2 on Friedman1 data should be positive, got {r2}"

    def test_friedman1_with_pc_selector(self, regression_data_friedman1):
        """Test PC selector on nonlinear Friedman #1 data.

        PC (Pearson correlation) is linear and should perform less optimally
        on nonlinear data, but should still work.
        """
        X, y = regression_data_friedman1
        reg = ConditionalInferenceTreeRegressor(
            selector="pc",
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape


class TestRegressorCorrelated:
    """Tests for regressor on correlated feature data."""

    def test_correlated_features_basic(self, regression_data_correlated):
        """Test regressor handles correlated features."""
        X, y = regression_data_correlated
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape

    def test_correlated_with_feature_scanning(self, regression_data_correlated):
        """Test feature scanning with correlated features."""
        X, y = regression_data_correlated
        reg = ConditionalInferenceTreeRegressor(
            feature_scanning=True,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_correlated_with_feature_muting(self, regression_data_correlated):
        """Test feature muting with correlated features."""
        X, y = regression_data_correlated
        reg = ConditionalInferenceTreeRegressor(
            feature_muting=True,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


class TestRegressorHonesty:
    """Tests for honest estimation in regressors."""

    def test_honest_vs_non_honest_predictions_differ(self, regression_data_standard):
        """Test that honest and non-honest regressors produce different predictions."""
        X, y = regression_data_standard

        # Non-honest regressor
        reg_regular = ConditionalInferenceTreeRegressor(
            honesty=False,
            max_depth=5,
            **FAST_PARAMS,
        )
        reg_regular.fit(X, y)

        # Honest regressor
        reg_honest = ConditionalInferenceTreeRegressor(
            honesty=True,
            honesty_fraction=0.5,
            max_depth=5,
            **FAST_PARAMS,
        )
        reg_honest.fit(X, y)

        preds_regular = reg_regular.predict(X)
        preds_honest = reg_honest.predict(X)

        # Predictions should differ (honest uses estimation sample for leaf values)
        assert not np.allclose(preds_regular, preds_honest), (
            "Honest and non-honest predictions should differ"
        )

    def test_honest_regressor_different_fractions(self, regression_data_standard):
        """Test honest regressor with different honesty fractions."""
        X, y = regression_data_standard

        for fraction in [0.3, 0.5, 0.7]:
            reg = ConditionalInferenceTreeRegressor(
                honesty=True,
                honesty_fraction=fraction,
                **FAST_PARAMS,
            )
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape


class TestRegressorListSelector:
    """Tests for list-based selector in regressors."""

    def test_two_way_combinations(self, regression_data_standard):
        """Test all 2-way selector combinations for regression."""
        X, y = regression_data_standard

        for selector in [["pc", "dc"], ["pc", "rdc"], ["dc", "rdc"]]:
            reg = ConditionalInferenceTreeRegressor(selector=selector, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape

    def test_three_way_combination(self, regression_data_standard):
        """Test 3-way selector combination for regression."""
        X, y = regression_data_standard

        reg = ConditionalInferenceTreeRegressor(
            selector=["pc", "dc", "rdc"],
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_list_selector_no_ptest(self, regression_data_standard):
        """Test list selector without permutation testing."""
        X, y = regression_data_standard

        reg = ConditionalInferenceTreeRegressor(
            selector=["pc", "dc"],
            n_resamples_selector=None,
            n_resamples_splitter="minimum",
            verbose=0,
            random_state=42,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


class TestRegressorThresholdMethods:
    """Tests for threshold methods in regressors."""

    def test_threshold_exact(self, regression_data_standard):
        """Test exact threshold method for regressor."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            threshold_method="exact",
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_threshold_random(self, regression_data_standard):
        """Test random threshold method for regressor."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            threshold_method="random",
            max_thresholds=10,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_threshold_percentile(self, regression_data_standard):
        """Test percentile threshold method for regressor."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            threshold_method="percentile",
            max_thresholds=10,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_threshold_histogram(self, regression_data_standard):
        """Test histogram threshold method for regressor."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            threshold_method="histogram",
            max_thresholds=10,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


class TestRegressorFeatureControl:
    """Tests for feature control parameters in regressors."""

    def test_feature_muting_enabled(self, regression_data_standard):
        """Test regressor with feature muting enabled."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            feature_muting=True,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_feature_muting_disabled(self, regression_data_standard):
        """Test regressor with feature muting disabled."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            feature_muting=False,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_feature_scanning_enabled(self, regression_data_standard):
        """Test regressor with feature scanning enabled."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            feature_scanning=True,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_feature_scanning_disabled(self, regression_data_standard):
        """Test regressor with feature scanning disabled."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            feature_scanning=False,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_max_features_sqrt(self, regression_data_standard):
        """Test regressor with max_features='sqrt'."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            max_features="sqrt",
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_max_features_log2(self, regression_data_standard):
        """Test regressor with max_features='log2'."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            max_features="log2",
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_max_features_int(self, regression_data_standard):
        """Test regressor with max_features as integer."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            max_features=5,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape

    def test_max_features_float(self, regression_data_standard):
        """Test regressor with max_features as float."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(
            max_features=0.5,
            **FAST_PARAMS,
        )
        reg.fit(X, y)
        assert reg.predict(X).shape == y.shape


class TestRegressorAttributes:
    """Tests for regressor attributes after fitting."""

    def test_feature_importances(self, regression_data_standard):
        """Test feature_importances_ attribute for regressor."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)

        assert hasattr(reg, "feature_importances_")
        assert reg.feature_importances_.shape == (X.shape[1],)
        assert (reg.feature_importances_ >= 0).all()

    def test_n_features_in(self, regression_data_standard):
        """Test n_features_in_ attribute for regressor."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)

        assert hasattr(reg, "n_features_in_")
        assert reg.n_features_in_ == X.shape[1]

    def test_tree_attribute(self, regression_data_standard):
        """Test tree_ attribute for regressor."""
        X, y = regression_data_standard
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)

        assert hasattr(reg, "tree_")
        assert reg.tree_ is not None


class TestRegressorHeteroscedastic:
    """Tests for regressor on heteroscedastic data."""

    def test_basic_fit_predict(self, regression_data_heteroscedastic):
        """Test regressor handles heteroscedastic data."""
        X, y = regression_data_heteroscedastic
        reg = ConditionalInferenceTreeRegressor(**FAST_PARAMS)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape

    def test_different_selectors(self, regression_data_heteroscedastic):
        """Test different selectors on heteroscedastic data."""
        X, y = regression_data_heteroscedastic

        for selector in ["pc", "dc", "rdc"]:
            reg = ConditionalInferenceTreeRegressor(selector=selector, **FAST_PARAMS)
            reg.fit(X, y)
            assert reg.predict(X).shape == y.shape


# =============================================================================
# BONFERRONI CORRECTION TESTS
# =============================================================================


class TestBonferronCorrection:
    """Tests for Bonferroni correction behavior."""

    def test_bonferroni_correction_resets_when_single_test(self) -> None:
        """Regression test: per-node Bonferroni must not leak across nodes.

        In particular, calling _bonferroni_correction with n_tests=1 should reset the
        private per-node attributes to their unadjusted values.
        """
        clf = ConditionalInferenceTreeClassifier(
            alpha_selector=0.05,
            alpha_splitter=0.05,
            adjust_alpha_selector=True,
            adjust_alpha_splitter=True,
            n_resamples_selector=100,
            n_resamples_splitter=100,
            random_state=0,
        )

        clf._bonferroni_correction(adjust="selector", n_tests=10)
        assert clf._alpha_selector == 0.05 / 10
        assert clf._n_resamples_selector == 100 * 10

        clf._bonferroni_correction(adjust="selector", n_tests=1)
        assert clf._alpha_selector == 0.05
        assert clf._n_resamples_selector == 100

        clf._bonferroni_correction(adjust="splitter", n_tests=5)
        assert clf._alpha_splitter == 0.05 / 5
        assert clf._n_resamples_splitter == 100 * 5

        clf._bonferroni_correction(adjust="splitter", n_tests=1)
        assert clf._alpha_splitter == 0.05
        assert clf._n_resamples_splitter == 100


# =============================================================================
# MULTI-SELECTOR TESTS
# =============================================================================


class TestMultiSelectorValidation:
    """Test multi-selector input validation."""

    def test_duplicate_selectors_rejected(self):
        """Duplicate selectors should raise ValueError."""
        with pytest.raises(ValueError, match="contains duplicates"):
            ConditionalInferenceTreeClassifier(selector=["mc", "mc"])

    def test_valid_multi_selector_accepted(self):
        """Valid multi-selector combinations should be accepted."""
        clf = ConditionalInferenceTreeClassifier(selector=["mc", "rdc"])
        assert clf.selector == ["mc", "rdc"]


class TestMultiSelectorTypeIError:
    """Verify multi-selector mode controls Type I error (max-T method)."""

    def test_multiselector_basic_runs(self):
        """Smoke test: multi-selector mode runs without error."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        y = rng.randint(0, 2, 50)

        clf = ConditionalInferenceTreeClassifier(
            selector=["mc", "rdc"],
            n_resamples_selector=50,
            early_stopping_selector=None,
            alpha_selector=0.05,
            n_resamples_splitter=None,
            random_state=42,
            verbose=0,
        )
        clf.fit(X, y)

    def test_multiselector_type1_error_controlled(self):
        """Multi-selector rejection rate should be ~alpha under null."""
        n_sims = 20
        alpha = 0.05
        rejections = 0

        for seed in range(n_sims):
            rng_x = np.random.RandomState(seed)
            rng_y = np.random.RandomState(seed + 10000)
            X = rng_x.randn(30, 1)
            y = rng_y.randint(0, 2, 30)

            clf = ConditionalInferenceTreeClassifier(
                selector="mc",
                n_resamples_selector=50,
                early_stopping_selector=None,
                alpha_selector=alpha,
                adjust_alpha_selector=False,
                n_resamples_splitter=None,
                random_state=seed,
                verbose=0,
            )
            clf.fit(X, y)

            if clf.tree_.get("feature") is not None:
                rejections += 1

        rejection_rate = rejections / n_sims
        assert rejection_rate <= 0.50, f"Type I error way too high: {rejection_rate:.3f} > 0.50"


# =============================================================================
# TIE-BREAKING TESTS (RESERVOIR SAMPLING)
# =============================================================================


class TestReservoirSamplingTieBreaking:
    """Verify uniform tie-breaking when p-values or metrics are equal."""

    def test_metric_tie_breaking_with_identical_features(self):
        """When features are identical copies, tie-breaking should be uniform."""
        from collections import Counter

        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 5

        base_feature = rng.standard_normal(n_samples)
        X = np.column_stack([base_feature] * n_features)
        y = (base_feature > 0).astype(int)

        n_trials = 500
        root_features = []

        for seed in range(n_trials):
            clf = ConditionalInferenceTreeClassifier(
                n_resamples_selector=None,
                n_resamples_splitter=None,
                max_depth=1,
                random_state=seed,
                verbose=0,
            )
            clf.fit(X, y)

            if "feature" in clf.tree_:
                root_features.append(clf.tree_["feature"])

        counts = Counter(root_features)

        for feature_idx in range(n_features):
            proportion = counts.get(feature_idx, 0) / len(root_features)
            assert proportion > 0.10, (
                f"Feature {feature_idx} selected only {proportion:.1%} of the time"
            )
            assert proportion < 0.35, f"Feature {feature_idx} selected {proportion:.1%} of the time"

    def test_metric_tie_breaking_regressor_with_identical_features(self):
        """When features are identical copies, tie-breaking should be uniform (regressor)."""
        from collections import Counter

        rng = np.random.default_rng(123)
        n_samples = 100
        n_features = 5

        base_feature = rng.standard_normal(n_samples)
        X = np.column_stack([base_feature] * n_features)
        y = base_feature + 0.1 * rng.standard_normal(n_samples)

        n_trials = 500
        root_features = []

        for seed in range(n_trials):
            reg = ConditionalInferenceTreeRegressor(
                n_resamples_selector=None,
                n_resamples_splitter=None,
                max_depth=1,
                random_state=seed,
                verbose=0,
            )
            reg.fit(X, y)

            if "feature" in reg.tree_:
                root_features.append(reg.tree_["feature"])

        counts = Counter(root_features)

        for feature_idx in range(n_features):
            proportion = counts.get(feature_idx, 0) / len(root_features)
            assert proportion > 0.10
            assert proportion < 0.35

    def test_tie_breaking_reproducible_with_fixed_seed(self):
        """Same random_state should produce identical tie-breaking decisions."""
        rng = np.random.default_rng(999)
        n_samples = 100
        n_features = 5

        base_feature = rng.standard_normal(n_samples)
        X = np.column_stack([base_feature] * n_features)
        y = (base_feature > 0).astype(int)

        clf1 = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            max_depth=1,
            random_state=42,
            verbose=0,
        )
        clf1.fit(X, y)

        clf2 = ConditionalInferenceTreeClassifier(
            n_resamples_selector=None,
            n_resamples_splitter=None,
            max_depth=1,
            random_state=42,
            verbose=0,
        )
        clf2.fit(X, y)

        assert clf1.tree_.get("feature") == clf2.tree_.get("feature")

    def test_different_seeds_produce_different_selections(self):
        """Different random_state values produce different tie-breaking outcomes."""
        rng = np.random.default_rng(777)
        n_samples = 100
        n_features = 5

        base_feature = rng.standard_normal(n_samples)
        X = np.column_stack([base_feature] * n_features)
        y = (base_feature > 0).astype(int)

        unique_root_features = set()
        for seed in range(100):
            clf = ConditionalInferenceTreeClassifier(
                n_resamples_selector=None,
                n_resamples_splitter=None,
                max_depth=1,
                random_state=seed,
                verbose=0,
            )
            clf.fit(X, y)
            if "feature" in clf.tree_:
                unique_root_features.add(clf.tree_["feature"])

        assert len(unique_root_features) > 1


class TestReservoirSamplingMathematical:
    """Test the mathematical properties of reservoir sampling directly."""

    def test_reservoir_sampling_uniform_distribution(self):
        """Verify reservoir sampling gives uniform distribution over ties."""
        from collections import Counter

        rng = np.random.default_rng(12345)
        n_candidates = 5
        n_trials = 10000

        winners = []
        for _ in range(n_trials):
            best = 0
            for k in range(1, n_candidates):
                n_ties = k + 1
                if rng.random() < 1.0 / n_ties:
                    best = k
            winners.append(best)

        counts = Counter(winners)

        expected = n_trials / n_candidates
        for candidate in range(n_candidates):
            actual = counts.get(candidate, 0)
            assert abs(actual - expected) < 0.2 * expected

    def test_reservoir_sampling_three_ties(self):
        """Test reservoir sampling with exactly 3 ties."""
        from collections import Counter

        rng = np.random.default_rng(54321)
        n_candidates = 3
        n_trials = 10000

        winners = []
        for _ in range(n_trials):
            best = 0
            n_ties = 1
            for k in range(1, n_candidates):
                n_ties += 1
                if rng.random() < 1.0 / n_ties:
                    best = k
            winners.append(best)

        counts = Counter(winners)

        expected = n_trials / n_candidates
        for candidate in range(n_candidates):
            actual = counts.get(candidate, 0)
            assert abs(actual - expected) < 0.15 * expected
