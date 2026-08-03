"""Tests for citrees._selector.py."""

import os
from collections.abc import Callable

import numba
import numpy as np
import pytest
from scipy.stats import kstest

from citrees import _selector
from citrees._selector import (
    _RDC_K,
    _RDC_S,
    _correlation,
    _covariance,
    _ptest_multi,
    _rdc,
    _rdc_cancor,
    _rdc_ecdf,
    _rdc_features,
    dc,
    mc,
    mi,
    pc,
    ptest_dc,
    ptest_mc,
    ptest_mi,
    ptest_pc,
    ptest_rdc_classifier,
    ptest_rdc_regressor,
    rdc_classifier,
    rdc_regressor,
)


class TestMultipleCorrelation:
    """Tests for mc (multiple correlation) selector."""

    def test_perfect_separation(self):
        """Test mc returns 1.0 for perfect class separation."""
        x = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        result = mc(x, y, n_classes=2)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_no_separation(self):
        """Test mc returns ~0 for random/no separation."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        result = mc(x, y, n_classes=2)
        # Should be low for random data
        assert result < 0.3

    def test_multiclass(self):
        """Test mc works with multiple classes."""
        x = np.array([0.0, 0.0, 5.0, 5.0, 10.0, 10.0])
        y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        result = mc(x, y, n_classes=3)
        assert 0 <= result <= 1

    def test_output_range(self):
        """Test mc returns values in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(50)
            y = np.random.randint(0, 2, 50).astype(np.int64)
            result = mc(x, y, n_classes=2)
            assert 0 <= result <= 1


class TestMutualInformation:
    """Tests for mi (mutual information) selector."""

    def test_perfect_dependence(self):
        """Test mi is high for perfectly dependent features."""
        x = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        result = mi(x, y, n_classes=2, random_state=42)
        assert result > 0.5

    def test_no_dependence(self):
        """Test mi is low for independent features."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randint(0, 2, 200).astype(np.int64)
        result = mi(x, y, n_classes=2, random_state=42)
        assert result < 0.1

    def test_output_non_negative(self):
        """Test mi returns non-negative values."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randint(0, 3, 100).astype(np.int64)
        result = mi(x, y, n_classes=3, random_state=42)
        assert result >= 0


class TestPearsonCorrelation:
    """Tests for pc (Pearson correlation) selector."""

    def test_perfect_positive_correlation(self):
        """Test pc returns 1.0 for perfect positive correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pc(x, y, standardize=True)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_perfect_negative_correlation(self):
        """Test pc returns -1.0 for perfect negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = pc(x, y, standardize=True)
        assert result == pytest.approx(-1.0, rel=0.01)

    def test_no_correlation(self):
        """Test pc returns ~0 for uncorrelated data."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)
        result = pc(x, y, standardize=True)
        assert abs(result) < 0.2

    def test_covariance_mode(self):
        """Test pc returns covariance when standardize=False."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        cov = pc(x, y, standardize=False)
        corr = pc(x, y, standardize=True)
        # Covariance should be positive for positive relationship
        assert cov > 0
        # Correlation should be 1.0
        assert corr == pytest.approx(1.0, rel=0.01)


class TestDistanceCorrelation:
    """Tests for dc (distance correlation) selector."""

    def test_perfect_linear(self):
        """Test dc detects linear relationship."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = dc(x, y, standardize=True)
        assert result > 0.9

    def test_no_dependence(self):
        """Test dc is low for independent data."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        result = dc(x, y, standardize=True)
        assert result < 0.3

    def test_nonlinear_dependence(self):
        """Test dc detects nonlinear relationships."""
        x = np.linspace(-3, 3, 100)
        y = x**2  # Parabolic relationship
        result = dc(x, y, standardize=True)
        # DC should detect the nonlinear relationship
        assert result > 0.4

    def test_output_range(self):
        """Test dc returns values in [0, 1]."""
        np.random.seed(42)
        for _ in range(5):
            x = np.random.randn(50)
            y = np.random.randn(50)
            result = dc(x, y, standardize=True)
            assert 0 <= result <= 1


class TestRDC:
    """Tests for RDC (Randomized Dependence Coefficient) selector."""

    def test_rdc_classifier_strong_signal(self):
        """Test RDC classifier with strong signal."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        result = rdc_classifier(x, y, n_classes=2, random_state=42)
        assert result > 0.5

    def test_rdc_classifier_no_signal(self):
        """Test RDC classifier with no signal."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        result = rdc_classifier(x, y, n_classes=2, random_state=42)
        assert result < 0.5

    def test_rdc_classifier_multiclass(self):
        """Test RDC classifier with multiclass."""
        x = np.concatenate([np.zeros(30) + i for i in range(3)])
        y = np.concatenate([np.full(30, i, dtype=np.int64) for i in range(3)])
        result = rdc_classifier(x, y, n_classes=3, random_state=42)
        assert result > 0.5

    def test_rdc_regressor_linear(self):
        """Test RDC regressor with linear relationship."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1
        result = rdc_regressor(x, y, standardize=True, random_state=42)
        assert result > 0.7

    def test_rdc_regressor_nonlinear(self):
        """Test RDC regressor with nonlinear relationship."""
        x = np.linspace(-3, 3, 100)
        y = x**2
        result = rdc_regressor(x, y, standardize=True, random_state=42)
        assert result > 0.5

    def test_rdc_output_range(self):
        """Test RDC returns values in [0, 1]."""
        np.random.seed(42)
        for _ in range(5):
            x = np.random.randn(50)
            y = np.random.randn(50)
            result = rdc_regressor(x, y, standardize=True, random_state=42)
            assert 0 <= result <= 1

    def test_internal_rdc_function(self):
        """Test internal _rdc function."""
        x = np.linspace(0, 10, 50)
        y = x * 2
        result = _rdc(x, y, k=_RDC_K, s=_RDC_S, seed=42)
        assert result > 0.5

    def test_rdc_constant_input(self):
        """Test RDC handles constant input."""
        x = np.ones(50)
        y = np.random.randn(50)
        result = _rdc(x, y, k=_RDC_K, s=_RDC_S, seed=42)
        assert result == 0.0

    def test_rdc_small_sample(self):
        """Test RDC handles very small samples."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        result = _rdc(x, y, k=_RDC_K, s=_RDC_S, seed=42)
        assert result == 0.0  # n < 3 returns 0


class TestCorrelationHelpers:
    """Tests for correlation helper functions."""

    def test_correlation_function(self):
        """Test _correlation helper."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _correlation(x, y)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_covariance_function(self):
        """Test _covariance helper."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _covariance(x, y)
        # Covariance of x with 2x should be 2 * var(x)
        assert result > 0


class TestPtestMC:
    """Tests for ptest_mc permutation test."""

    def test_strong_signal_low_pvalue(self):
        """Test ptest_mc gives low p-value for strong signal."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_parallel_version(self):
        """Test parallel version is triggered with enough resamples."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        # n_resamples >= 200 triggers parallel version
        pval = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05


def test_pvalue_uniform_under_null():
    """P-values should be approximately uniform when no signal exists.

    Under the null hypothesis (no relationship between X and y), p-values
    from a valid permutation test should be uniformly distributed on [0, 1].

    This test verifies:
    1. P-values follow a uniform distribution (KS test)
    2. False positive rate is near the nominal alpha level
    """
    n_trials = 500
    pvalues = []

    for seed in range(n_trials):
        rng = np.random.RandomState(seed)
        x = rng.randn(200)  # Single feature, pure noise
        y = rng.randint(0, 2, 200).astype(np.int64)  # Random labels

        pval = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=seed,
        )
        pvalues.append(pval)

    pvalues = np.array(pvalues)

    # KS test for uniformity
    # Note: With +1 correction, p-values are discrete (1/101, 2/101, ...)
    # so we use a conservative threshold
    stat, p = kstest(pvalues, "uniform")
    assert p > 0.001, f"P-values not uniform under null: KS stat={stat:.4f}, p={p:.4f}"

    # Check false positive rate is near nominal alpha=0.05
    # Allow range [0.02, 0.10] for sampling variability
    fp_rate = np.mean(pvalues < 0.05)
    assert 0.02 < fp_rate < 0.10, (
        f"False positive rate {fp_rate:.3f} outside expected range [0.02, 0.10]"
    )


def test_pvalue_never_zero():
    """P-values should never be exactly zero (Phipson & Smyth 2010 correction).

    With the +1 correction, minimum p-value is 1/(n_resamples+1).
    """
    rng = np.random.RandomState(42)

    # Create data with VERY strong signal - should give minimum possible p-value
    n = 200
    x = np.concatenate([rng.randn(n // 2) - 10, rng.randn(n // 2) + 10])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)

    n_resamples = 100
    pval = ptest_mc(
        x=x,
        y=y,
        n_classes=2,
        n_resamples=n_resamples,
        early_stopping=None,
        alpha=0.05,
        random_state=42,
    )

    # P-value should be 1/(n_resamples+1) = 1/101 ≈ 0.0099, never 0
    expected_min = 1 / (n_resamples + 1)
    assert pval > 0, "P-value should never be exactly zero"
    assert pval == pytest.approx(expected_min, rel=0.01), (
        f"With strong signal, p-value should be minimum possible: {expected_min:.4f}, got {pval:.4f}"
    )


class TestPtestMI:
    """Tests for ptest_mi permutation test."""

    def test_strong_signal_low_pvalue(self):
        """Test ptest_mi gives low p-value for strong signal."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval = ptest_mi(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.1

    def test_no_signal_high_pvalue(self):
        """Test ptest_mi gives reasonable p-value for no signal."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        pval = ptest_mi(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert 0 < pval <= 1


class TestPtestPC:
    """Tests for ptest_pc permutation test."""

    def test_strong_signal_low_pvalue(self):
        """Test ptest_pc gives low p-value for strong signal."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        pval = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_no_signal_high_pvalue(self):
        """Test ptest_pc gives reasonable p-value for no signal."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        pval = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert 0 < pval <= 1

    def test_parallel_version(self):
        """Test parallel version is triggered with enough resamples."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        # n_resamples >= 200 triggers parallel version
        pval = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05


class TestPtestDC:
    """Tests for ptest_dc permutation test."""

    def test_strong_linear_signal(self):
        """Test ptest_dc gives low p-value for linear signal."""
        x = np.linspace(0, 10, 50)
        y = 2 * x + np.random.randn(50) * 0.1
        pval = ptest_dc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.1

    def test_nonlinear_signal(self):
        """Test ptest_dc detects nonlinear relationships."""
        x = np.linspace(-3, 3, 50)
        y = x**2 + np.random.randn(50) * 0.1
        pval = ptest_dc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.1


class TestPtestRDC:
    """Tests for ptest_rdc permutation tests."""

    def test_classifier_strong_signal(self):
        """Test ptest_rdc_classifier gives low p-value for strong signal."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.1

    def test_regressor_strong_signal(self):
        """Test ptest_rdc_regressor gives low p-value for strong signal."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        pval = ptest_rdc_regressor(
            x=x,
            y=y,
            standardize=True,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.1

    def test_classifier_no_signal(self):
        """Test ptest_rdc_classifier gives reasonable p-value for no signal."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        pval = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert 0 < pval <= 1

    def test_regressor_parallel_strong_signal(self):
        """Test parallel RDC regressor gives low p-value for strong signal."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        # n_resamples >= 200 triggers parallel path
        pval = ptest_rdc_regressor(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_regressor_parallel_batched_strong_signal(self):
        """Test parallel batched RDC regressor gives low p-value for strong signal."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        pval = ptest_rdc_regressor(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_classifier_parallel_binary_strong_signal(self):
        """Test parallel RDC classifier (binary) gives low p-value for strong signal."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_classifier_parallel_batched_binary(self):
        """Test parallel batched RDC classifier (binary) gives low p-value."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_classifier_parallel_multiclass(self):
        """Test parallel RDC classifier (multiclass) gives low p-value."""
        x = np.concatenate([np.zeros(30) + i for i in range(3)])
        y = np.concatenate([np.full(30, i, dtype=np.int64) for i in range(3)])
        pval = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=3,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_classifier_parallel_batched_multiclass(self):
        """Test parallel batched RDC classifier (multiclass) gives low p-value."""
        x = np.concatenate([np.zeros(30) + i for i in range(3)])
        y = np.concatenate([np.full(30, i, dtype=np.int64) for i in range(3)])
        pval = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=3,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_regressor_parallel_null_feature(self):
        """Test parallel RDC regressor gives large p-value for null feature."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        pval = ptest_rdc_regressor(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0.05

    def test_classifier_parallel_null_feature(self):
        """Test parallel RDC classifier gives large p-value for null feature."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        y = rng.integers(0, 2, size=100).astype(np.int64)
        pval = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0.05

    def test_regressor_parallel_reproducible(self):
        """Test parallel RDC regressor is reproducible with same seed."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1.0
        pval1 = ptest_rdc_regressor(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        pval2 = ptest_rdc_regressor(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval1 == pval2

    def test_classifier_parallel_reproducible(self):
        """Test parallel RDC classifier is reproducible with same seed."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval1 = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        pval2 = ptest_rdc_classifier(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval1 == pval2

    def test_regressor_parallel_pvalue_never_zero(self):
        """Test parallel RDC regressor p-value is never exactly zero."""
        x = np.linspace(0, 10, 200)
        y = 100 * x  # extreme signal
        pval = ptest_rdc_regressor(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0


class TestEarlyStopping:
    """Tests for early stopping in permutation tests."""

    def test_early_stopping_mc(self):
        """Test early stopping for ptest_mc."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=1000,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_early_stopping_pc(self):
        """Test early stopping for ptest_pc."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        pval = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=1000,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_batched_parallel_mc_strong_signal(self):
        """Test batched parallel path is taken for mc with adaptive + n_resamples >= 200."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        # n_resamples >= 200 + adaptive triggers batched parallel
        pval = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_batched_parallel_pc_strong_signal(self):
        """Test batched parallel path is taken for pc with adaptive + n_resamples >= 200."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        # n_resamples >= 200 + adaptive triggers batched parallel
        pval = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_batched_parallel_mc_no_signal(self):
        """Test batched parallel mc doesn't reject null."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        pval = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0

    def test_batched_parallel_pc_no_signal(self):
        """Test batched parallel pc doesn't reject null."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        pval = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0

    def test_batched_parallel_mc_pvalue_never_zero(self):
        """Test batched parallel mc p-value is never exactly zero."""
        rng = np.random.RandomState(42)
        n = 200
        x = np.concatenate([rng.randn(n // 2) - 10, rng.randn(n // 2) + 10])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)
        pval = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0

    def test_batched_parallel_mc_reproducible(self):
        """Test batched parallel mc gives same result with same seed."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval1 = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        pval2 = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval1 == pval2

    def test_batched_parallel_pc_reproducible(self):
        """Test batched parallel pc gives same result with same seed."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1.0
        pval1 = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        pval2 = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval1 == pval2

    def test_batched_parallel_pc_pvalue_never_zero(self):
        """Test batched parallel pc p-value is never exactly zero."""
        x = np.linspace(0, 10, 200)
        y = 100 * x  # extreme signal
        pval = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=250,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0

    def test_ptest_adaptive_small_resamples_mc(self):
        """Test the serial adaptive branch with n_resamples below the parallel threshold."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        # n_resamples < 200 routes to the serial Python adaptive branch.
        pval = ptest_mc(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=100,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_ptest_adaptive_small_resamples_pc(self):
        """Test the serial PC adaptive branch below the parallel threshold."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.randn(100) * 0.1
        # n_resamples < 200 routes to the serial Python adaptive branch.
        pval = ptest_pc(
            x=x,
            y=y,
            standardize=True,
            n_resamples=100,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_ptest_multi_adaptive_batched(self):
        """Test _ptest_multi() adaptive batched branch."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        pval = _ptest_multi(
            funcs=[mc],
            func_args=[2],
            take_abs=[True],
            x=x,
            y=y,
            n_resamples=100,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_adaptive_stopping_is_host_cpu_invariant(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A host's reported CPU count must not change a seeded p-value."""
        rng = np.random.default_rng(1718)
        x = rng.normal(size=80)
        y = 0.15 * x + rng.normal(size=80)
        kwargs = {
            "func": pc,
            "func_arg": True,
            "x": x,
            "y": y,
            "n_resamples": 100,
            "early_stopping": "adaptive",
            "alpha": 0.05,
            "random_state": 1718,
        }

        monkeypatch.setattr(os, "cpu_count", lambda: 1)
        one_cpu = _selector._ptest_result(**kwargs)
        monkeypatch.setattr(os, "cpu_count", lambda: 64)
        many_cpus = _selector._ptest_result(**kwargs)

        assert one_cpu == many_cpus

    def test_multi_selector_stopping_is_host_cpu_invariant(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A host's reported CPU count must not change a seeded max-T p-value."""
        rng = np.random.default_rng(1718)
        x = rng.normal(size=80)
        y = (x + rng.normal(size=80) > 0).astype(np.int64)
        kwargs = {
            "funcs": [mc],
            "func_args": [2],
            "take_abs": [True],
            "x": x,
            "y": y,
            "n_resamples": 100,
            "early_stopping": "adaptive",
            "alpha": 0.05,
            "random_state": 1718,
        }

        monkeypatch.setattr(os, "cpu_count", lambda: 1)
        one_cpu = _ptest_multi(**kwargs)
        monkeypatch.setattr(os, "cpu_count", lambda: 64)
        many_cpus = _ptest_multi(**kwargs)

        assert one_cpu == many_cpus

    def test_minimum_budget_is_batch_boundary_invariant(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The minimum budget permits only the final adaptive decision check."""
        rng = np.random.default_rng(1718)
        x = rng.normal(size=80)
        y = rng.normal(size=80)
        kwargs = {
            "func": pc,
            "func_arg": True,
            "x": x,
            "y": y,
            "n_resamples": 20,
            "early_stopping": "adaptive",
            "alpha": 0.05,
            "random_state": 1718,
        }
        observed = []
        for batch_size in (1, 8, 32, 64):
            monkeypatch.setattr(_selector, "_ADAPTIVE_BATCH_SIZE", batch_size)
            observed.append(_selector._ptest_result(**kwargs))

        assert observed == [observed[0]] * len(observed)


class TestSelectorDirect:
    """Direct tests for selector functions (JIT disabled via NUMBA_DISABLE_JIT=1)."""

    def test_mc_perfect_separation(self):
        """Test mc with perfect class separation."""
        x = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        result = mc(x, y, n_classes=2)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_mc_no_separation(self):
        """Test mc with random data."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        result = mc(x, y, n_classes=2)
        assert result < 0.3

    def test_mc_constant_feature_returns_zero(self):
        """mc should return 0.0 for constant feature."""
        x = np.ones(50)
        y = np.array([0, 1] * 25, dtype=np.int64)
        result = mc(x, y, n_classes=2)
        assert result == 0.0

    def test_pc_perfect_positive(self):
        """Test pc with perfect positive correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pc(x, y, standardize=True)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_pc_perfect_negative(self):
        """Test pc with perfect negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = pc(x, y, standardize=True)
        assert result == pytest.approx(-1.0, rel=0.01)

    def test_covariance_positive(self):
        """Test _covariance with positive covariance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _covariance(x, y)
        assert result > 0

    def test_correlation_perfect(self):
        """Test _correlation with perfect correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = _correlation(x, y)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_rdc_ecdf(self):
        """Test _rdc_ecdf computes ECDF."""
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        result = _rdc_ecdf(x)
        assert np.all(result > 0)
        assert np.all(result <= 1)

    def test_rdc_linear(self):
        """Test _rdc with linear relationship."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1
        result = _rdc(x, y, k=_RDC_K, s=_RDC_S, seed=42)
        assert result > 0.7

    def test_rdc_constant(self):
        """Test _rdc with constant input."""
        x = np.ones(50)
        y = np.random.randn(50)
        result = _rdc(x, y, k=_RDC_K, s=_RDC_S, seed=42)
        assert result == 0.0

    def test_mc_multiclass(self):
        """Test mc with multiclass."""
        x = np.array([0.0, 0.0, 5.0, 5.0, 10.0, 10.0])
        y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        result = mc(x, y, n_classes=3)
        assert 0 <= result <= 1

    def test_pc_no_correlation(self):
        """Test pc with no correlation."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)
        result = pc(x, y, standardize=True)
        assert abs(result) < 0.2

    def test_pc_covariance_mode(self):
        """Test pc returns covariance when standardize=False."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        cov = pc(x, y, standardize=False)
        assert cov > 0

    def test_covariance_negative(self):
        """Test _covariance with negative covariance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        result = _covariance(x, y)
        assert result < 0

    def test_correlation_zero(self):
        """Test _correlation with uncorrelated data."""
        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)
        result = _correlation(x, y)
        assert abs(result) < 0.2

    def test_correlation_constant_inputs_returns_zero(self):
        """_correlation should return 0.0 for constant inputs."""
        x = np.ones(50)
        y = np.ones(50)
        result = _correlation(x, y)
        assert result == 0.0

    def test_rdc_features(self):
        """Test _rdc_features computes random features."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rdc_features(x, k=5, s=1.0 / 6.0, seed=42)
        assert result.shape == (len(x), 10)  # k features + k bias terms

    def test_rdc_cancor(self):
        """Test _rdc_cancor computes canonical correlation."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        Y = np.random.randn(50, 5)
        result = _rdc_cancor(X, Y)
        assert 0 <= result <= 1

    def test_rdc_no_relationship(self):
        """Test _rdc with no relationship."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        result = _rdc(x, y, k=_RDC_K, s=_RDC_S, seed=42)
        assert result < 0.5

    def test_rdc_small_sample(self):
        """Test _rdc with very small sample."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        result = _rdc(x, y, k=_RDC_K, s=_RDC_S, seed=42)
        assert result == 0.0  # n < 3 returns 0


# =============================================================================
# RNG REPRODUCIBILITY TESTS
# =============================================================================


class TestSelectorRNGReproducibility:
    """Test RNG reproducibility for selector permutation tests."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        np.random.seed(42)
        x = np.random.randn(100).astype(np.float64)
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        return x, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        np.random.seed(42)
        x = np.random.randn(100).astype(np.float64)
        y = np.random.randn(100).astype(np.float64)
        return x, y

    def test_ptest_same_seed_same_result(self, classification_data):
        """Same random_state should produce identical p-values."""
        from citrees._selector import _ptest_result

        x, y = classification_data

        result1 = _ptest_result(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        result2 = _ptest_result(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        assert result1 == result2, f"Same seed should give same result: {result1} != {result2}"

    def test_ptest_different_seed_different_result(self, classification_data):
        """Different random_state should produce different p-values."""
        from citrees._selector import _ptest_result

        x, y = classification_data

        result1 = _ptest_result(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        result2 = _ptest_result(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=99,
        )

        assert result1 != result2, (
            f"Different seeds should give different results: {result1} == {result2}"
        )

    def test_ptest_no_global_state_contamination(self, classification_data):
        """The serial result core should not contaminate global RNG state."""
        from citrees._selector import _ptest_result

        x, y = classification_data

        np.random.seed(123)
        before = np.random.random()

        np.random.seed(123)
        _ptest_result(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=999,
        )
        after = np.random.random()

        assert before == after, f"_ptest_result contaminated global state: {before} != {after}"

    def test_ptest_multi_same_seed_same_result(self, classification_data):
        """_ptest_multi with same seed should produce identical results."""
        from citrees._selector import _ptest_multi

        x, y = classification_data

        pval1 = _ptest_multi(
            funcs=[mc],
            func_args=[2],
            take_abs=[True],
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        pval2 = _ptest_multi(
            funcs=[mc],
            func_args=[2],
            take_abs=[True],
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"

    def test_ptest_multi_no_global_state_contamination(self, classification_data):
        """_ptest_multi should not contaminate global RNG state."""
        from citrees._selector import _ptest_multi

        x, y = classification_data

        np.random.seed(123)
        before = np.random.random()

        np.random.seed(123)
        _ptest_multi(
            funcs=[mc],
            func_args=[2],
            take_abs=[True],
            x=x,
            y=y,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=999,
        )
        after = np.random.random()

        assert before == after, f"_ptest_multi contaminated global state: {before} != {after}"


class TestParallelSelectorRNGReproducibility:
    """Test RNG reproducibility for parallel selector permutation tests."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification test data."""
        np.random.seed(42)
        x = np.random.randn(100).astype(np.float64)
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        return x, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression test data."""
        np.random.seed(42)
        x = np.random.randn(100).astype(np.float64)
        y = np.random.randn(100).astype(np.float64)
        return x, y

    def test_ptest_mc_parallel_same_seed_same_result(self, classification_data):
        """Parallel MC test with same seed should produce identical results."""
        from citrees._selector import _ptest_mc_parallel_result

        x, y = classification_data

        result1 = _ptest_mc_parallel_result(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)
        result2 = _ptest_mc_parallel_result(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)

        assert result1 == result2, f"Same seed should give same result: {result1} != {result2}"

    def test_ptest_mc_parallel_different_seed_different_result(self, classification_data):
        """Parallel MC test with different seed should produce different results."""
        from citrees._selector import _ptest_mc_parallel_result

        x, y = classification_data

        result1 = _ptest_mc_parallel_result(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)
        result2 = _ptest_mc_parallel_result(x=x, y=y, n_classes=2, n_resamples=500, random_state=99)

        assert result1 != result2, (
            f"Different seeds should give different results: {result1} == {result2}"
        )

    def test_ptest_pc_parallel_same_seed_same_result(self, regression_data):
        """Parallel PC test with same seed should produce identical results."""
        from citrees._selector import _ptest_pc_parallel_result

        x, y = regression_data

        result1 = _ptest_pc_parallel_result(x=x, y=y, n_resamples=500, random_state=42)
        result2 = _ptest_pc_parallel_result(x=x, y=y, n_resamples=500, random_state=42)

        assert result1 == result2, f"Same seed should give same result: {result1} != {result2}"


# =============================================================================
# JIT / py_func PARITY
# =============================================================================


@pytest.mark.skipif(numba.config.DISABLE_JIT, reason="JIT disabled: no compiled kernel to compare")
class TestJitParity:
    """Compiled Numba kernels must match their pure-Python source."""

    _RNG = np.random.default_rng(1718)
    _X = _RNG.standard_normal(120)
    _Y_CLF = (_X + _RNG.standard_normal(120) * 0.5 > 0).astype(np.int64)
    _Y_REG = 2.0 * _X + _RNG.standard_normal(120) * 0.5
    _X_2D = _RNG.standard_normal((60, 3))
    _Y_2D = _RNG.standard_normal((60, 3))

    KERNELS = {
        "mc": (_selector.mc, (_X, _Y_CLF, 2, 1718)),
        "pc": (_selector.pc, (_X, _Y_REG, True, 1718)),
        "_correlation": (_selector._correlation, (_X, _Y_REG)),
        "_covariance": (_selector._covariance, (_X, _Y_REG)),
        "_rdc": (_selector._rdc, (_X, _Y_REG, _RDC_K, _RDC_S, 1718)),
        "_rdc_cancor": (_selector._rdc_cancor, (_X_2D, _Y_2D)),
        "_rdc_ecdf": (_selector._rdc_ecdf, (_X,)),
        "_rdc_features": (_selector._rdc_features, (_X, _RDC_K, _RDC_S, 1718)),
        "_beta_cdf": (_selector._beta_cdf, (0.3, 2.0, 5.0)),
        "_ptest_mc_parallel_result": (
            _selector._ptest_mc_parallel_result,
            (_X, _Y_CLF, 2, 250, 1718),
        ),
        "_ptest_mc_parallel_batched_result": (
            _selector._ptest_mc_parallel_batched_result,
            (_X, _Y_CLF, 2, 250, 1718, 0.05, 0.95),
        ),
        "_ptest_pc_parallel_result": (
            _selector._ptest_pc_parallel_result,
            (_X, _Y_REG, 250, 1718),
        ),
        "_ptest_pc_parallel_batched_result": (
            _selector._ptest_pc_parallel_batched_result,
            (_X, _Y_REG, 250, 1718, 0.05, 0.95),
        ),
        "_ptest_rdc_classifier_parallel_result": (
            _selector._ptest_rdc_classifier_parallel_result,
            (_X, _Y_CLF, 2, _RDC_K, _RDC_S, 1718, 250, 1718),
        ),
        "_ptest_rdc_classifier_parallel_batched_result": (
            _selector._ptest_rdc_classifier_parallel_batched_result,
            (_X, _Y_CLF, 2, _RDC_K, _RDC_S, 1718, 250, 1718, 0.05, 0.95),
        ),
        "_ptest_rdc_regressor_parallel_result": (
            _selector._ptest_rdc_regressor_parallel_result,
            (_X, _Y_REG, _RDC_K, _RDC_S, 1718, 250, 1718),
        ),
        "_ptest_rdc_regressor_parallel_batched_result": (
            _selector._ptest_rdc_regressor_parallel_batched_result,
            (_X, _Y_REG, _RDC_K, _RDC_S, 1718, 250, 1718, 0.05, 0.95),
        ),
    }

    @pytest.mark.parametrize("name", sorted(KERNELS))
    def test_jit_matches_py_func(self, name, assert_jit_parity):
        """The compiled kernel and its Python source must return identical values."""
        fn, args = self.KERNELS[name]
        assert_jit_parity(fn, *args)

    def test_every_kernel_has_a_parity_case(self, assert_all_kernels_covered):
        """Fail when a Numba kernel is added to _selector without a parity case."""
        assert_all_kernels_covered(_selector, {fn for fn, _ in self.KERNELS.values()})


# =============================================================================
# STATISTICAL VALIDITY OF THE PERMUTATION TESTS
# =============================================================================
# The library's central claim is that its permutation tests produce valid
# p-values and that EarlyStopping.ADAPTIVE preserves that validity. These
# properties are asserted directly here rather than inferred from downstream
# accuracy. All tests are marked slow: each runs hundreds of permutation tests.


N_TRIALS = 300
N_SAMPLES = 150
N_RESAMPLES = 100
ALPHA = 0.05


def _null_classification(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Feature and labels drawn independently, so the null holds by construction."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N_SAMPLES)
    y = rng.integers(0, 2, N_SAMPLES).astype(np.int64)
    return x, y


def _null_regression(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Feature and target drawn independently, so the null holds by construction."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N_SAMPLES)
    y = rng.standard_normal(N_SAMPLES)
    return x, y


def _clf_selector(fn: Callable[..., float]) -> Callable[[int], float]:
    def run(seed: int) -> float:
        x, y = _null_classification(seed)
        return fn(
            x=x,
            y=y,
            n_classes=2,
            n_resamples=N_RESAMPLES,
            early_stopping=None,
            alpha=ALPHA,
            random_state=seed,
        )

    return run


def _reg_selector(fn: Callable[..., float]) -> Callable[[int], float]:
    def run(seed: int) -> float:
        x, y = _null_regression(seed)
        return fn(
            x=x,
            y=y,
            standardize=True,
            n_resamples=N_RESAMPLES,
            early_stopping=None,
            alpha=ALPHA,
            random_state=seed,
        )

    return run


NULL_SELECTORS = {
    "mc": _clf_selector(ptest_mc),
    "mi": _clf_selector(ptest_mi),
    "rdc_classifier": _clf_selector(ptest_rdc_classifier),
    "pc": _reg_selector(ptest_pc),
    "dc": _reg_selector(ptest_dc),
    "rdc_regressor": _reg_selector(ptest_rdc_regressor),
}


def _null_pvalues(run: Callable[[int], float], n_trials: int = N_TRIALS) -> np.ndarray:
    return np.array([run(seed) for seed in range(n_trials)])


# ``mi`` is excluded from the uniformity check by construction, not by
# convenience. It delegates to sklearn's ``mutual_info_classif``, whose kNN
# estimator clips negative estimates to exactly 0.0; under the null that happens
# for roughly half of all datasets. A zero statistic is tied by every
# permutation, so those trials return p = 1.0 and the null distribution carries
# a large atom at 1. That makes the test conservative, not invalid, which
# ``test_no_anti_conservatism_under_null`` verifies for every selector.
CONTINUOUS_STATISTIC_SELECTORS = sorted(set(NULL_SELECTORS) - {"mi"})


@pytest.mark.slow
@pytest.mark.parametrize("name", CONTINUOUS_STATISTIC_SELECTORS)
def test_pvalues_uniform_under_null(name: str) -> None:
    """Selectors with a continuous statistic produce uniform p-values under the null."""
    pvalues = _null_pvalues(NULL_SELECTORS[name])

    # Permutation p-values are discrete on multiples of 1/(n_resamples+1), so a
    # strict KS threshold would reject on discreteness alone.
    stat, p = kstest(pvalues, "uniform")
    assert p > 0.001, f"{name}: p-values not uniform under null (KS stat={stat:.4f}, p={p:.4g})"


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(NULL_SELECTORS))
def test_no_anti_conservatism_under_null(name: str) -> None:
    """P(p <= t) must not exceed t under the null, for every selector.

    This is the validity condition a permutation test has to satisfy. It holds
    for conservative discrete tests such as ``mi``, where strict uniformity does
    not, and it is what guarantees the nominal alpha is an upper bound on the
    false positive rate.
    """
    pvalues = _null_pvalues(NULL_SELECTORS[name])

    # Tolerance is ~3 binomial standard errors at the worst-case p=0.5.
    tolerance = 3.0 * np.sqrt(0.25 / N_TRIALS)
    for threshold in (0.01, 0.05, 0.10, 0.25, 0.50):
        rate = float(np.mean(pvalues <= threshold))
        assert rate <= threshold + tolerance, (
            f"{name}: anti-conservative at t={threshold}: P(p<=t)={rate:.3f} exceeds {threshold + tolerance:.3f}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(NULL_SELECTORS))
def test_type_i_error_controlled_under_null(name: str) -> None:
    """Rejection rate under the null must stay near the nominal alpha."""
    pvalues = _null_pvalues(NULL_SELECTORS[name])
    rate = float(np.mean(pvalues < ALPHA))

    # Binomial standard error at alpha=0.05 over N_TRIALS draws is ~0.013, so a
    # 0.02-0.10 band is roughly +/- 4 SE and fails on real inflation, not noise.
    assert 0.02 < rate < 0.10, (
        f"{name}: Type I error {rate:.3f} outside [0.02, 0.10] at alpha={ALPHA}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(NULL_SELECTORS))
def test_pvalues_never_zero_and_bounded(name: str) -> None:
    """The Phipson & Smyth +1 correction bounds p-values away from zero."""
    pvalues = _null_pvalues(NULL_SELECTORS[name], n_trials=50)

    assert pvalues.min() >= 1.0 / (N_RESAMPLES + 1) - 1e-12, (
        f"{name}: p-value below the +1 correction floor"
    )
    assert pvalues.max() <= 1.0, f"{name}: p-value above 1"


@pytest.mark.slow
def test_adaptive_early_stopping_preserves_type_i_error() -> None:
    """EarlyStopping.ADAPTIVE must keep Type I error at roughly the nominal alpha.

    This is the property that makes ADAPTIVE the default: it is ~95% faster than
    the full permutation test without inflating the false positive rate.
    """
    pvalues = []
    for seed in range(N_TRIALS):
        x, y = _null_classification(seed)
        pvalues.append(
            ptest_mc(
                x=x,
                y=y,
                n_classes=2,
                n_resamples=N_RESAMPLES,
                early_stopping="adaptive",
                alpha=ALPHA,
                random_state=seed,
            )
        )

    rate = float(np.mean(np.array(pvalues) < ALPHA))
    assert rate < 0.10, (
        f"adaptive early stopping inflated Type I error to {rate:.3f} at alpha={ALPHA}"
    )


@pytest.mark.slow
def test_adaptive_early_stopping_retains_power() -> None:
    """ADAPTIVE must still reject when a strong signal is present."""
    rng = np.random.default_rng(1718)
    x = np.concatenate([rng.standard_normal(75) - 3, rng.standard_normal(75) + 3])
    y = np.concatenate([np.zeros(75), np.ones(75)]).astype(np.int64)

    pval = ptest_mc(
        x=x,
        y=y,
        n_classes=2,
        n_resamples=N_RESAMPLES,
        early_stopping="adaptive",
        alpha=ALPHA,
        random_state=1718,
    )
    assert pval < ALPHA, f"adaptive early stopping failed to detect a strong signal (p={pval:.4f})"
