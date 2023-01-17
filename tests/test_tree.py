"""Tests related to citrees._tree.py file."""
from typing import Any, Dict

import numpy as np
import pytest
from sklearn import datasets

from citrees._tree import ConditionalInferenceTreeClassifier, ConditionalInferenceTreeRegressor, Node


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


class TestConditionalInferenceTreeClassifier:
    """Test ConditionalInferenceTreeClassifier functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Initialize tests."""
        self.X, self.y = datasets.load_breast_cancer(return_X_y=True, as_frame=False)

    def test_fit(self) -> None:
        """Test fit method."""
        # TODO:
        import pdb

        pdb.set_trace()
        _ = ConditionalInferenceTreeClassifier().fit(self.X, self.y)
        assert 0


class TestConditionalInferenceTreeRegressor:
    """Test ConditionalInferenceTreeRegressor functionality."""

    # @pytest.fixture(autouse=True)
    # def setup(self) -> None:
    #     """Initialize tests."""
    #     self.X, self.y = datasets.load_breast_cancer(return_X_y=True, as_frame=False)

    # def test_fit(self) -> None:
    #     """Test fit method."""
    #     # TODO:
    #     _ = ConditionalInferenceTreeRegressor().fit(self.X, self.y)
    #     import pdb

    #     pdb.set_trace()
