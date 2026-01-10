"""Tests for citrees._.tree.py."""
from typing import Any, Dict

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
def test_node(kwargs: Dict[str, Any]) -> None:
    """Test Node functionality."""
    node = Node(**kwargs)

    for key, value in node.items():
        assert value == kwargs[key]


def test_base_conditional_inference_tree_parameters():
    """Test BaseConditionalInferenceTreeParameters functionality."""
    # Failure
    with pytest.raises(ValidationError) as e:
        BaseConditionalInferenceTreeParameters()
    assert e.type is ValidationError, f"Wrong exception, got ({e.type}) but expected ({ValidationError})"

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
        early_stopping_selector=True,
        early_stopping_splitter=True,
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
    assert (
        type(params) is BaseConditionalInferenceTreeParameters
    ), f"Wrong class, got ({type(params)}) but expected ({BaseConditionalInferenceTreeParameters})"


class TestHonestEstimation:
    """Tests for honest estimation feature."""

    def test_classifier_honest_basic(self):
        """Test honest classifier can fit and predict."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        clf = ConditionalInferenceTreeClassifier(
            honesty=True,
            honesty_fraction=0.5,
            random_state=42,
            verbose=0,
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
            random_state=42,
            verbose=0,
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
            random_state=42,
            verbose=0,
        )
        clf_regular.fit(X, y)

        # Honest tree
        clf_honest = ConditionalInferenceTreeClassifier(
            honesty=True,
            honesty_fraction=0.5,
            random_state=42,
            verbose=0,
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
