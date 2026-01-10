"""Tests for citrees._.tree.py."""
from typing import Any, Dict

import numpy as np
import pytest
from pydantic import ValidationError

from citrees._tree import BaseConditionalInferenceTreeParameters, Node

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
        random_state=None,
        verbose=1,
        check_for_unused_parameters=False,
    )
    assert (
        type(params) is BaseConditionalInferenceTreeParameters
    ), f"Wrong class, got ({type(params)}) but expected ({BaseConditionalInferenceTreeParameters})"
