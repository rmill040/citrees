"""Tests for citrees._utils.py."""

from typing import Any

import numpy as np
import pytest

from citrees._utils import (
    balanced_bootstrap_sample,
    balanced_bootstrap_unsampled_idx,
    bayesian_bootstrap_proba,
    calculate_max_value,
    classic_bootstrap_sample,
    classic_bootstrap_unsampled_idx,
    estimate_mean,
    estimate_proba,
    split_data,
    stratified_bootstrap_sample,
    stratified_bootstrap_unsampled_idx,
)

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
def test_estimate_proba(kwargs: dict[str, Any], expected: float) -> None:
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


class TestCalculateMaxValue:
    """Tests for calculate_max_value function."""

    def test_int_value(self):
        """Test with integer desired_max."""
        result = calculate_max_value(n_values=100, desired_max=50)
        assert result == 50

    def test_int_value_capped(self):
        """Test int is capped by n_values."""
        result = calculate_max_value(n_values=30, desired_max=50)
        assert result == 30

    def test_sqrt(self):
        """Test with 'sqrt' desired_max."""
        result = calculate_max_value(n_values=100, desired_max="sqrt")
        assert result == 10  # sqrt(100) = 10

    def test_log2(self):
        """Test with 'log2' desired_max."""
        result = calculate_max_value(n_values=64, desired_max="log2")
        assert result == 6  # log2(64) = 6

    def test_float_fraction(self):
        """Test with float desired_max."""
        result = calculate_max_value(n_values=100, desired_max=0.5)
        assert result == 50

    def test_none_returns_all(self):
        """Test None returns all values."""
        result = calculate_max_value(n_values=100, desired_max=None)
        assert result == 100

    def test_float_capped(self):
        """Test float result is capped by n_values."""
        result = calculate_max_value(n_values=50, desired_max=0.8)
        assert result == 40


class TestSplitData:
    """Tests for split_data function."""

    def test_basic_split(self):
        """Test basic data splitting."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 0, 1, 1])
        threshold = 4.0  # Split on first feature

        X_left, y_left, X_right, y_right = split_data(
            X=X, y=y, feature=0, threshold=threshold
        )

        # First two samples should be left (x <= 4)
        assert len(X_left) == 2
        assert len(X_right) == 2
        assert np.array_equal(y_left, [0, 0])
        assert np.array_equal(y_right, [1, 1])

    def test_all_left(self):
        """Test all samples go left."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 2])

        X_left, y_left, X_right, y_right = split_data(
            X=X, y=y, feature=0, threshold=10.0
        )

        assert len(X_left) == 3
        assert len(X_right) == 0

    def test_all_right(self):
        """Test all samples go right."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 2])

        X_left, y_left, X_right, y_right = split_data(
            X=X, y=y, feature=0, threshold=0.0
        )

        assert len(X_left) == 0
        assert len(X_right) == 3


class TestBayesianBootstrapProba:
    """Tests for bayesian_bootstrap_proba function."""

    def test_returns_probabilities(self):
        """Test that output sums to 1."""
        proba = bayesian_bootstrap_proba(n=100, random_state=42)
        assert proba.shape == (100,)
        assert np.isclose(proba.sum(), 1.0)

    def test_all_positive(self):
        """Test all probabilities are positive."""
        proba = bayesian_bootstrap_proba(n=50, random_state=42)
        assert (proba > 0).all()

    def test_reproducible(self):
        """Test reproducibility with same seed."""
        p1 = bayesian_bootstrap_proba(n=100, random_state=42)
        p2 = bayesian_bootstrap_proba(n=100, random_state=42)
        assert np.allclose(p1, p2)

    def test_different_seeds(self):
        """Test different seeds give different results."""
        p1 = bayesian_bootstrap_proba(n=100, random_state=42)
        p2 = bayesian_bootstrap_proba(n=100, random_state=123)
        assert not np.allclose(p1, p2)


class TestClassicBootstrapSample:
    """Tests for classic_bootstrap_sample function."""

    def test_returns_correct_size(self):
        """Test returns correct sample size."""
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        idx = classic_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        assert len(idx) == len(y)

    def test_with_max_samples(self):
        """Test with max_samples constraint."""
        y = np.array([0, 0, 0, 1, 1, 1])
        idx = classic_bootstrap_sample(
            y=y, max_samples=3, bayesian_bootstrap=False, random_state=42
        )
        assert len(idx) == 3

    def test_bayesian_mode(self):
        """Test Bayesian bootstrap mode."""
        y = np.array([0, 0, 0, 1, 1, 1])
        idx = classic_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=True, random_state=42
        )
        assert len(idx) == len(y)

    def test_reproducible(self):
        """Test reproducibility."""
        y = np.array([0, 0, 0, 1, 1, 1])
        idx1 = classic_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        idx2 = classic_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        assert np.array_equal(idx1, idx2)


class TestClassicBootstrapUnsampledIdx:
    """Tests for classic_bootstrap_unsampled_idx function."""

    def test_returns_unsampled(self):
        """Test returns unsampled indices."""
        y = np.array([0, 0, 0, 1, 1, 1])
        idx_unsampled = classic_bootstrap_unsampled_idx(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        idx_sampled = classic_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        # Unsampled indices should not be in sampled indices
        for i in idx_unsampled:
            assert i not in idx_sampled


class TestStratifiedBootstrapSample:
    """Tests for stratified_bootstrap_sample function."""

    def test_maintains_class_proportions(self):
        """Test that class proportions are maintained."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        idx = stratified_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        y_sampled = y[idx]
        # Both classes should be represented
        assert (y_sampled == 0).sum() > 0
        assert (y_sampled == 1).sum() > 0

    def test_with_max_samples(self):
        """Test with max_samples constraint."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        idx = stratified_bootstrap_sample(
            y=y, max_samples=4, bayesian_bootstrap=False, random_state=42
        )
        assert len(idx) <= 4

    def test_multiclass(self):
        """Test with multiclass data."""
        y = np.array([0, 0, 1, 1, 2, 2])
        idx = stratified_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        y_sampled = y[idx]
        # All classes should be represented
        assert (y_sampled == 0).sum() > 0
        assert (y_sampled == 1).sum() > 0
        assert (y_sampled == 2).sum() > 0


class TestStratifiedBootstrapUnsampledIdx:
    """Tests for stratified_bootstrap_unsampled_idx function."""

    def test_returns_unsampled(self):
        """Test returns unsampled indices."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        idx_unsampled = stratified_bootstrap_unsampled_idx(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        idx_sampled = stratified_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        for i in idx_unsampled:
            assert i not in idx_sampled


class TestBalancedBootstrapSample:
    """Tests for balanced_bootstrap_sample function."""

    def test_balanced_classes(self):
        """Test that classes are balanced."""
        y = np.array([0, 0, 0, 0, 0, 1, 1])  # Imbalanced
        idx = balanced_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        y_sampled = y[idx]
        # Should be balanced now
        assert (y_sampled == 0).sum() == (y_sampled == 1).sum()

    def test_with_max_samples(self):
        """Test with max_samples constraint."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        idx = balanced_bootstrap_sample(
            y=y, max_samples=4, bayesian_bootstrap=False, random_state=42
        )
        assert len(idx) <= 4

    def test_multiclass_balanced(self):
        """Test multiclass balancing."""
        y = np.array([0, 0, 0, 0, 1, 1, 2])  # Imbalanced
        idx = balanced_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        y_sampled = y[idx]
        # Each class should have min_count samples
        counts = [np.sum(y_sampled == c) for c in range(3)]
        assert counts[0] == counts[1] == counts[2]


class TestBalancedBootstrapUnsampledIdx:
    """Tests for balanced_bootstrap_unsampled_idx function."""

    def test_returns_unsampled(self):
        """Test returns unsampled indices."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        idx_unsampled = balanced_bootstrap_unsampled_idx(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        idx_sampled = balanced_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        for i in idx_unsampled:
            assert i not in idx_sampled
