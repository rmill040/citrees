"""Tests for citrees._selector.py."""

import numpy as np
import pytest
from scipy.stats import kstest

from citrees._selector import (
    _RDC_K,
    _RDC_S,
    _correlation,
    _covariance,
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
            x=x, y=y, n_classes=2, n_resamples=100, early_stopping=None, alpha=0.05, random_state=42
        )
        assert pval < 0.05

    def test_parallel_version(self):
        """Test parallel version is triggered with enough resamples."""
        x = np.concatenate([np.zeros(50), np.ones(50)])
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.int64)
        # n_resamples >= 200 triggers parallel version
        pval = ptest_mc(
            x=x, y=y, n_classes=2, n_resamples=250, early_stopping=None, alpha=0.05, random_state=42
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
            x=x, y=y, n_classes=2, n_resamples=50, early_stopping=None, alpha=0.05, random_state=42
        )
        assert pval < 0.1

    def test_no_signal_high_pvalue(self):
        """Test ptest_mi gives reasonable p-value for no signal."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randint(0, 2, 100).astype(np.int64)
        pval = ptest_mi(
            x=x, y=y, n_classes=2, n_resamples=50, early_stopping=None, alpha=0.05, random_state=42
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
            x=x, y=y, n_classes=2, n_resamples=50, early_stopping=None, alpha=0.05, random_state=42
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
            x=x, y=y, n_classes=2, n_resamples=50, early_stopping=None, alpha=0.05, random_state=42
        )
        assert 0 < pval <= 1


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
        from citrees._selector import _ptest

        x, y = classification_data

        pval1 = _ptest(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        pval2 = _ptest(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"

    def test_ptest_different_seed_different_result(self, classification_data):
        """Different random_state should produce different p-values."""
        from citrees._selector import _ptest

        x, y = classification_data

        pval1 = _ptest(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        pval2 = _ptest(
            func=mc,
            func_arg=2,
            x=x,
            y=y,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=99,
        )

        assert pval1 != pval2, f"Different seeds should give different results: {pval1} == {pval2}"

    def test_ptest_no_global_state_contamination(self, classification_data):
        """_ptest should not contaminate global RNG state."""
        from citrees._selector import _ptest

        x, y = classification_data

        np.random.seed(123)
        before = np.random.random()

        np.random.seed(123)
        _ptest(
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

        assert before == after, f"_ptest contaminated global state: {before} != {after}"

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
        from citrees._selector import _ptest_mc_parallel

        x, y = classification_data

        pval1 = _ptest_mc_parallel(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)
        pval2 = _ptest_mc_parallel(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"

    def test_ptest_mc_parallel_different_seed_different_result(self, classification_data):
        """Parallel MC test with different seed should produce different results."""
        from citrees._selector import _ptest_mc_parallel

        x, y = classification_data

        pval1 = _ptest_mc_parallel(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)
        pval2 = _ptest_mc_parallel(x=x, y=y, n_classes=2, n_resamples=500, random_state=99)

        assert pval1 != pval2, f"Different seeds should give different results: {pval1} == {pval2}"

    def test_ptest_pc_parallel_same_seed_same_result(self, regression_data):
        """Parallel PC test with same seed should produce identical results."""
        from citrees._selector import _ptest_pc_parallel

        x, y = regression_data

        pval1 = _ptest_pc_parallel(x=x, y=y, n_resamples=500, random_state=42)
        pval2 = _ptest_pc_parallel(x=x, y=y, n_resamples=500, random_state=42)

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"
