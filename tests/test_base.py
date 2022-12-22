"""Tests related to citrees._base.py file."""
from typing import Any, Dict

import pytest
import numpy as np

from citrees._base import Node


@pytest.mark.parametrize("kwargs", [
    {
        "feature": 0,
        "feature_pval": 0.5,
        "split_pval": 0.5,
        "threshold": 2.5,
        "impurity": 1.0,
        "left_child": (np.ones(6).reshape(3, 2), np.zeros(6)),
        "right_child": (np.ones(6).reshape(3, 2), np.zeros(6)),
    },
    {"value": 0.5},
])
def test_node(kwargs: Dict[str, Any]) -> None:
    """Test Node functionality."""
    node = Node(**kwargs)

    for key, value in node.items():
        assert value == kwargs[key]
