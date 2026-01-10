"""Tests for citrees._utils.py."""
from typing import Any, Dict

import numpy as np
import pytest

from citrees._utils import estimate_mean, estimate_proba

pytestmark = pytest.mark.other


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"y": np.array([0, 0, 1, 1]), "n_classes": 1}, np.array([0.5])),
        ({"y": np.array([0, 0, 1, 1]), "n_classes": 2}, np.array([0.5, 0.5])),
        ({"y": np.array([0, 0, 1, 1]), "n_classes": 3}, np.array([0.5, 0.5, 0.0])),
        ({"y": np.array([1, 1, 2, 2]), "n_classes": 3}, np.array([0.0, 0.5, 0.5])),
    ],
)
def test_estimate_proba(kwargs: Dict[str, Any], expected: float) -> None:
    """Test estimate_proba function."""
    proba = estimate_proba(**kwargs)
    assert np.all(proba == expected)


def test_estimate_mean() -> None:
    """Test estimate_mean function."""
    for test in range(1, 4):
        np.random.seed(test)
        x = np.random.normal(0, 1, 100)
        mean = estimate_mean(x)
        assert np.allclose(mean, np.mean(x))
