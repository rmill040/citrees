"""Tests related to citrees._tree.py file."""
import pytest
from sklearn import datasets

from citrees._tree import ConditionalInferenceTreeClassifier


class TestConditionalInferenceTreeClassifier:
    """Test ConditionalInferenceTreeClassifier functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Initialize tests."""
        self.X, self.y = datasets.load_breast_cancer(return_X_y=True, as_frame=False)

    def test_fit(self) -> None:
        """Test fit method."""
        clf = ConditionalInferenceTreeClassifier().fit(self.X.tolist(), self.y.tolist())
