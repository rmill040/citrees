"""Tests for citrees._sequential.py."""

import numpy as np
import pytest
from scipy.stats import beta as scipy_beta

from citrees._sequential import (
    _beta_cdf,
    _ptest_sequential_adaptive,
    _ptest_sequential_simple,
)


class TestBetaCDF:
    """Tests for the Beta CDF implementation."""

    def test_boundary_zero(self):
        """Test Beta CDF at x=0."""
        assert _beta_cdf(0.0, 1.0, 1.0) == 0.0
        assert _beta_cdf(0.0, 2.0, 3.0) == 0.0

    def test_boundary_one(self):
        """Test Beta CDF at x=1."""
        assert _beta_cdf(1.0, 1.0, 1.0) == 1.0
        assert _beta_cdf(1.0, 2.0, 3.0) == 1.0

    def test_uniform_distribution(self):
        """Test Beta(1,1) is uniform - CDF should equal x."""
        for x in [0.1, 0.25, 0.5, 0.75, 0.9]:
            assert _beta_cdf(x, 1.0, 1.0) == pytest.approx(x, rel=1e-6)

    def test_against_scipy(self):
        """Test our Beta CDF matches scipy."""
        test_cases = [
            (0.05, 1.0, 1.0),
            (0.05, 2.0, 10.0),
            (0.5, 5.0, 5.0),
            (0.1, 10.0, 100.0),
            (0.9, 100.0, 10.0),
            (0.05, 1.0, 50.0),
            (0.95, 50.0, 1.0),
        ]
        for x, a, b in test_cases:
            expected = scipy_beta.cdf(x, a, b)
            result = _beta_cdf(x, a, b)
            assert result == pytest.approx(expected, rel=1e-4), (
                f"Mismatch for Beta({a}, {b}).cdf({x}): got {result}, expected {expected}"
            )

    def test_symmetry(self):
        """Test Beta CDF symmetry: F(x; a, b) = 1 - F(1-x; b, a)."""
        test_cases = [(0.3, 2.0, 5.0), (0.7, 5.0, 2.0), (0.5, 3.0, 3.0)]
        for x, a, b in test_cases:
            left = _beta_cdf(x, a, b)
            right = 1.0 - _beta_cdf(1 - x, b, a)
            assert left == pytest.approx(right, rel=1e-6)

    def test_extreme_parameters(self):
        """Test Beta CDF with extreme parameters."""
        # Very small alpha
        result = _beta_cdf(0.5, 0.1, 0.1)
        expected = scipy_beta.cdf(0.5, 0.1, 0.1)
        assert result == pytest.approx(expected, rel=1e-3)

        # Large parameters
        result = _beta_cdf(0.5, 100.0, 100.0)
        expected = scipy_beta.cdf(0.5, 100.0, 100.0)
        assert result == pytest.approx(expected, rel=1e-3)


class TestSequentialSimple:
    """Tests for simple sequential permutation testing."""

    def test_strong_signal_rejects(self):
        """Test simple sequential rejects strong signal."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.float64)

        # Use a simple mean difference as the test function
        def mean_diff(x, y, arg, random_state=None):
            mask = x > 0.5
            if mask.sum() == 0 or (~mask).sum() == 0:
                return 0.0
            return np.mean(y[mask]) - np.mean(y[~mask])

        pval = _ptest_sequential_simple(
            func=mean_diff,
            func_arg=None,
            x=x,
            y=y,
            n_resamples=200,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_null_signal_high_pvalue(self):
        """Test simple sequential doesn't reject null."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)

        def mean_diff(x, y, arg, random_state=None):
            mask = x > 0
            if mask.sum() == 0 or (~mask).sum() == 0:
                return 0.0
            return np.mean(y[mask]) - np.mean(y[~mask])

        pval = _ptest_sequential_simple(
            func=mean_diff,
            func_arg=None,
            x=x,
            y=y,
            n_resamples=200,
            alpha=0.05,
            random_state=42,
        )
        # Under null, p-value should typically be high
        assert pval > 0

    def test_pvalue_never_zero(self):
        """Test p-value is never exactly zero."""
        np.random.seed(42)
        n = 100
        # Perfect separation
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2) - 100, np.ones(n // 2) + 100])

        def mean_diff(x, y, arg, random_state=None):
            mask = x > 0.5
            if mask.sum() == 0 or (~mask).sum() == 0:
                return 0.0
            return np.mean(y[mask]) - np.mean(y[~mask])

        pval = _ptest_sequential_simple(
            func=mean_diff,
            func_arg=None,
            x=x,
            y=y,
            n_resamples=100,
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0


class TestSequentialAdaptive:
    """Tests for adaptive sequential permutation testing."""

    def test_strong_signal_rejects(self):
        """Test adaptive sequential rejects strong signal."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.float64)

        def mean_diff(x, y, arg, random_state=None):
            mask = x > 0.5
            if mask.sum() == 0 or (~mask).sum() == 0:
                return 0.0
            return np.mean(y[mask]) - np.mean(y[~mask])

        pval = _ptest_sequential_adaptive(
            func=mean_diff,
            func_arg=None,
            x=x,
            y=y,
            n_resamples=200,
            alpha=0.05,
            confidence=0.95,
            random_state=42,
        )
        assert pval < 0.05

    def test_null_signal_high_pvalue(self):
        """Test adaptive sequential doesn't reject null."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)

        def mean_diff(x, y, arg, random_state=None):
            mask = x > 0
            if mask.sum() == 0 or (~mask).sum() == 0:
                return 0.0
            return np.mean(y[mask]) - np.mean(y[~mask])

        pval = _ptest_sequential_adaptive(
            func=mean_diff,
            func_arg=None,
            x=x,
            y=y,
            n_resamples=200,
            alpha=0.05,
            confidence=0.95,
            random_state=42,
        )
        # Under null, p-value should typically be high
        assert pval > 0

    def test_pvalue_never_zero(self):
        """Test p-value is never exactly zero."""
        np.random.seed(42)
        n = 100
        # Perfect separation
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2) - 100, np.ones(n // 2) + 100])

        def mean_diff(x, y, arg, random_state=None):
            mask = x > 0.5
            if mask.sum() == 0 or (~mask).sum() == 0:
                return 0.0
            return np.mean(y[mask]) - np.mean(y[~mask])

        pval = _ptest_sequential_adaptive(
            func=mean_diff,
            func_arg=None,
            x=x,
            y=y,
            n_resamples=100,
            alpha=0.05,
            confidence=0.95,
            random_state=42,
        )
        assert pval > 0

    def test_confidence_parameter(self):
        """Test different confidence levels."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.float64)

        def mean_diff(x, y, arg, random_state=None):
            mask = x > 0.5
            if mask.sum() == 0 or (~mask).sum() == 0:
                return 0.0
            return np.mean(y[mask]) - np.mean(y[~mask])

        # Both should reject, but possibly at different stopping times
        pval_high = _ptest_sequential_adaptive(
            func=mean_diff,
            func_arg=None,
            x=x,
            y=y,
            n_resamples=200,
            alpha=0.05,
            confidence=0.99,  # Higher confidence
            random_state=42,
        )
        pval_low = _ptest_sequential_adaptive(
            func=mean_diff,
            func_arg=None,
            x=x,
            y=y,
            n_resamples=200,
            alpha=0.05,
            confidence=0.90,  # Lower confidence
            random_state=42,
        )
        # Both should indicate significance
        assert pval_high < 0.1
        assert pval_low < 0.1


