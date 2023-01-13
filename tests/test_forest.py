"""Tests related to citrees._forest.py file."""
from typing import Any, Dict

import numpy as np
import pytest
from sklearn import datasets

from citrees._forest import ConditionalInferenceForestClassifier


class TestConditionalInferenceForestClassifier:
    """Test ConditionalInferenceForestClassifier functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Initialize tests."""
        self.X, self.y = datasets.load_breast_cancer(return_X_y=True, as_frame=False)

    def test_fit(self) -> None:
        """Test fit method."""
        # TODO:
        _ = ConditionalInferenceForestClassifier(verbose=3, n_jobs=-1).fit(self.X, self.y)
        import pdb; pdb.set_trace()
        assert 0
