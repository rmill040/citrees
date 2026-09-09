"""Tests for citrees._splitter.py."""

import numba
import numpy as np
import pytest

from citrees import _splitter
from citrees._splitter import (
    _ptest_mse_parallel_result,
    _ptest_result,
    entropy,
    gini,
    mae,
    mse,
    ptest_entropy,
    ptest_gini,
    ptest_mae,
    ptest_mse,
)


def _child_impurity(func, y, idx, weighted):
    """Compute the split statistic used by Stage B tests."""
    if weighted:
        n = len(y)
        n_left = int(np.sum(idx))
        n_right = n - n_left
        return (n_left / n) * func(y[idx]) + (n_right / n) * func(y[~idx])
    return func(y[idx]) + func(y[~idx])


def _expected_splitter_ptest(func, x, y, threshold, n_resamples, random_state, weighted):
    idx = x <= threshold
    theta = _child_impurity(func, y, idx, weighted)
    y_perm = y.copy()
    rng = np.random.default_rng(random_state)

    extreme_count = 0
    for _ in range(n_resamples):
        rng.shuffle(y_perm)
        theta_perm = _child_impurity(func, y_perm, idx, weighted)
        if theta_perm <= theta:
            extreme_count += 1

    return (extreme_count + 1) / (n_resamples + 1)


def _expected_parallel_splitter_ptest(func, x, y, threshold, n_resamples, random_state, weighted):
    idx = x <= threshold
    theta = _child_impurity(func, y, idx, weighted)
    extreme_count = 0
    random_state_before = np.random.get_state()

    try:
        for i in range(n_resamples):
            np.random.seed(random_state + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)
            theta_perm = _child_impurity(func, y_perm, idx, weighted)
            if theta_perm <= theta:
                extreme_count += 1
    finally:
        np.random.set_state(random_state_before)

    return (extreme_count + 1) / (n_resamples + 1)


class TestGini:
    """Tests for the gini impurity function."""

    def test_pure_node_zero_impurity(self):
        """Test that a pure node has zero Gini impurity."""
        y = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        assert gini(y) == pytest.approx(0.0)

        y = np.array([1, 1, 1, 1], dtype=np.int64)
        assert gini(y) == pytest.approx(0.0)

    def test_balanced_binary_max_impurity(self):
        """Test that balanced binary classes have max Gini impurity of 0.5."""
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        assert gini(y) == pytest.approx(0.5)

    def test_imbalanced_classes(self):
        """Test Gini impurity for imbalanced classes."""
        # 75% class 0, 25% class 1: Gini = 1 - (0.75^2 + 0.25^2) = 1 - 0.625 = 0.375
        y = np.array([0, 0, 0, 1], dtype=np.int64)
        assert gini(y) == pytest.approx(0.375)

    def test_multiclass(self):
        """Test Gini impurity for multiclass."""
        # Equal 3-class: Gini = 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
        y = np.array([0, 1, 2], dtype=np.int64)
        assert gini(y) == pytest.approx(2 / 3)


class TestEntropy:
    """Tests for the entropy impurity function."""

    def test_pure_node_zero_entropy(self):
        """Test that a pure node has zero entropy."""
        y = np.array([0, 0, 0, 0], dtype=np.int64)
        assert entropy(y) == pytest.approx(0.0)

    def test_balanced_binary_max_entropy(self):
        """Test that balanced binary classes have max entropy of 1.0."""
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        assert entropy(y) == pytest.approx(1.0)

    def test_imbalanced_classes(self):
        """Test entropy for imbalanced classes."""
        # 75% class 0, 25% class 1
        y = np.array([0, 0, 0, 1], dtype=np.int64)
        expected = -0.75 * np.log2(0.75) - 0.25 * np.log2(0.25)
        assert entropy(y) == pytest.approx(expected)

    def test_multiclass(self):
        """Test entropy for multiclass."""
        # Equal 3-class: entropy = -3*(1/3)*log2(1/3) = log2(3)
        y = np.array([0, 1, 2], dtype=np.int64)
        assert entropy(y) == pytest.approx(np.log2(3))


class TestMSE:
    """Tests for the MSE impurity function."""

    def test_constant_values_zero_mse(self):
        """Test that constant values have zero MSE."""
        y = np.array([5.0, 5.0, 5.0, 5.0])
        assert mse(y) == pytest.approx(0.0)

    def test_simple_variance(self):
        """Test MSE equals variance."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.var(y)
        assert mse(y) == pytest.approx(expected)

    def test_symmetric_values(self):
        """Test MSE for symmetric values."""
        y = np.array([-1.0, 1.0])
        # Mean = 0, MSE = (1 + 1) / 2 = 1
        assert mse(y) == pytest.approx(1.0)


class TestMAE:
    """Tests for the MAE impurity function."""

    def test_constant_values_zero_mae(self):
        """Test that constant values have zero MAE."""
        y = np.array([5.0, 5.0, 5.0, 5.0])
        assert mae(y) == pytest.approx(0.0)

    def test_simple_deviation(self):
        """Test MAE equals mean absolute deviation."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mean = np.mean(y)
        expected = np.mean(np.abs(y - mean))
        assert mae(y) == pytest.approx(expected)

    def test_symmetric_values(self):
        """Test MAE for symmetric values."""
        y = np.array([-1.0, 1.0])
        # Mean = 0, MAE = (1 + 1) / 2 = 1
        assert mae(y) == pytest.approx(1.0)

    def test_skewed_uses_median(self):
        """Test MAE uses median (L1-optimal) not mean."""
        y = np.array([1.0, 2.0, 3.0, 100.0])  # Skewed data
        # Median = 2.5, Mean = 26.5
        median_val = np.median(y)  # 2.5
        mean_val = np.mean(y)  # 26.5
        expected = np.mean(np.abs(y - median_val))  # ~24.0
        not_expected = np.mean(np.abs(y - mean_val))  # ~36.5
        result = mae(y)
        assert result == pytest.approx(expected)
        assert result != pytest.approx(not_expected, rel=0.1)


class TestPtestGini:
    """Tests for ptest_gini permutation test."""

    def test_strong_signal_low_pvalue(self):
        """Test that strong signal gives low p-value."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)
        threshold = 0.5

        pval = ptest_gini(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_no_signal_high_pvalue(self):
        """Test that no signal gives high p-value (on average)."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randint(0, 2, n).astype(np.int64)
        threshold = 0.0

        pval = ptest_gini(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        # P-value should typically be > 0.05 for random data
        # We don't assert exact value due to randomness
        assert 0 < pval <= 1

    def test_pvalue_never_zero(self):
        """Test that p-value is never exactly zero."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)

        pval = ptest_gini(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval > 0


class TestPtestEntropy:
    """Tests for ptest_entropy permutation test."""

    def test_strong_signal_low_pvalue(self):
        """Test that strong signal gives low p-value."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)

        pval = ptest_entropy(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05


class TestPtestMSE:
    """Tests for ptest_mse permutation test."""

    def test_strong_signal_low_pvalue(self):
        """Test that strong signal gives low p-value."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2) + 0, np.ones(n // 2) + 10])

        pval = ptest_mse(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_no_signal_high_pvalue(self):
        """Test that no signal gives reasonable p-value."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)

        pval = ptest_mse(
            x=x,
            y=y,
            threshold=0.0,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert 0 < pval <= 1


class TestStageBSplitStatistic:
    """Tests for the Stage B split-test statistic."""

    def test_generic_ptest_uses_weighted_child_impurity(self):
        x = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        y = np.array([0, 0, 10, 0, 1, 1, 1, 20], dtype=np.float64)

        pval = _ptest_result(
            func=mse,
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=10,
            early_stopping=None,
            alpha=0.05,
            random_state=2,
        )[0]

        weighted_expected = _expected_splitter_ptest(
            func=mse,
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=10,
            random_state=2,
            weighted=True,
        )
        plain_sum_expected = _expected_splitter_ptest(
            func=mse,
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=10,
            random_state=2,
            weighted=False,
        )

        assert pval == pytest.approx(weighted_expected)
        assert pval != pytest.approx(plain_sum_expected)

    def test_parallel_mse_ptest_uses_weighted_child_impurity(self):
        x = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64)
        y = np.array([0, 0, 10, 0, 1, 1, 1, 20], dtype=np.float64)

        pval = _ptest_mse_parallel_result(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=200,
            random_state=1,
        )[0]

        weighted_expected = _expected_parallel_splitter_ptest(
            func=mse,
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=200,
            random_state=1,
            weighted=True,
        )
        plain_sum_expected = _expected_parallel_splitter_ptest(
            func=mse,
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=200,
            random_state=1,
            weighted=False,
        )

        assert pval == pytest.approx(weighted_expected)
        assert pval != pytest.approx(plain_sum_expected)


class TestPtestMAE:
    """Tests for ptest_mae permutation test."""

    def test_strong_signal_low_pvalue(self):
        """Test that strong signal gives low p-value."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2) + 0, np.ones(n // 2) + 10])

        pval = ptest_mae(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05


class TestEarlyStopping:
    """Tests for early stopping behavior."""

    def test_early_stopping_produces_valid_pvalue(self):
        """Test that early stopping produces valid p-value."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)

        pval = ptest_gini(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=100,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        assert 0 < pval <= 1

    def test_early_stopping_faster_for_strong_signal(self):
        """Test that early stopping is effective for strong signals."""
        # This is more of a behavioral test - early stopping should
        # terminate quickly for very strong signals
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)

        pval = ptest_gini(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=1000,
            early_stopping="adaptive",
            alpha=0.05,
            random_state=42,
        )
        # Should get a low p-value
        assert pval < 0.05


class TestSplitterDirect:
    """Direct tests for splitter functions (JIT disabled via NUMBA_DISABLE_JIT=1)."""

    def test_gini_pure(self):
        """Test gini with pure node."""
        y = np.array([0, 0, 0, 0], dtype=np.int64)
        assert gini(y) == pytest.approx(0.0)

    def test_gini_balanced(self):
        """Test gini with balanced classes."""
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        assert gini(y) == pytest.approx(0.5)

    def test_gini_imbalanced(self):
        """Test gini with imbalanced classes."""
        y = np.array([0, 0, 0, 1], dtype=np.int64)
        assert gini(y) == pytest.approx(0.375)

    def test_gini_multiclass(self):
        """Test gini with multiclass."""
        y = np.array([0, 1, 2], dtype=np.int64)
        assert gini(y) == pytest.approx(2 / 3)

    def test_entropy_pure(self):
        """Test entropy with pure node."""
        y = np.array([0, 0, 0, 0], dtype=np.int64)
        assert entropy(y) == pytest.approx(0.0)

    def test_entropy_balanced(self):
        """Test entropy with balanced classes."""
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        assert entropy(y) == pytest.approx(1.0)

    def test_entropy_multiclass(self):
        """Test entropy with multiclass."""
        y = np.array([0, 1, 2], dtype=np.int64)
        assert entropy(y) == pytest.approx(np.log2(3))

    def test_mse_constant(self):
        """Test mse with constant values."""
        y = np.array([5.0, 5.0, 5.0, 5.0])
        assert mse(y) == pytest.approx(0.0)

    def test_mse_variance(self):
        """Test mse equals variance."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        expected = np.var(y)
        assert mse(y) == pytest.approx(expected)

    def test_mae_constant(self):
        """Test mae with constant values."""
        y = np.array([5.0, 5.0, 5.0, 5.0])
        assert mae(y) == pytest.approx(0.0)

    def test_mae_deviation(self):
        """Test mae equals mean absolute deviation."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mean = np.mean(y)
        expected = np.mean(np.abs(y - mean))
        assert mae(y) == pytest.approx(expected)


class TestParallelPtests:
    """Tests for parallel versions of permutation tests (n_resamples >= 200)."""

    def test_ptest_gini_parallel(self):
        """Test parallel version of ptest_gini."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)

        # n_resamples >= 200 triggers parallel version when early_stopping=None
        pval = ptest_gini(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_ptest_entropy_parallel(self):
        """Test parallel version of ptest_entropy."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(np.int64)

        pval = ptest_entropy(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_ptest_mse_parallel(self):
        """Test parallel version of ptest_mse."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2) + 0, np.ones(n // 2) + 10])

        pval = ptest_mse(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_ptest_mae_parallel(self):
        """Test parallel version of ptest_mae."""
        np.random.seed(42)
        n = 100
        x = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        y = np.concatenate([np.zeros(n // 2) + 0, np.ones(n // 2) + 10])

        pval = ptest_mae(
            x=x,
            y=y,
            threshold=0.5,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        assert pval < 0.05

    def test_parallel_no_signal(self):
        """Test parallel version with no signal."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.random.randn(n)

        pval = ptest_mse(
            x=x,
            y=y,
            threshold=0.0,
            n_resamples=250,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )
        # Should give reasonable p-value for random data
        assert 0 < pval <= 1


# =============================================================================
# RNG REPRODUCIBILITY TESTS
# =============================================================================


class TestSplitterRNGReproducibility:
    """Test RNG reproducibility for splitter permutation tests."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification test data with a threshold."""
        np.random.seed(42)
        x = np.random.randn(100).astype(np.float64)
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        threshold = 0.0
        return x, y, threshold

    @pytest.fixture
    def regression_data(self):
        """Generate regression test data with a threshold."""
        np.random.seed(42)
        x = np.random.randn(100).astype(np.float64)
        y = np.random.randn(100).astype(np.float64)
        threshold = 0.0
        return x, y, threshold

    def test_splitter_ptest_same_seed_same_result(self, classification_data):
        """The serial splitter result core is reproducible for a fixed seed."""
        from citrees._splitter import _ptest_result as _ptest_splitter_result

        x, y, threshold = classification_data

        result1 = _ptest_splitter_result(
            func=gini,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        result2 = _ptest_splitter_result(
            func=gini,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        assert result1 == result2, f"Same seed should give same result: {result1} != {result2}"

    def test_splitter_ptest_no_global_state_contamination(self, classification_data):
        """The serial splitter result core should not contaminate global RNG state."""
        from citrees._splitter import _ptest_result as _ptest_splitter_result

        x, y, threshold = classification_data

        np.random.seed(123)
        before = np.random.random()

        np.random.seed(123)
        _ptest_splitter_result(
            func=gini,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=50,
            early_stopping=None,
            alpha=0.05,
            random_state=999,
        )
        after = np.random.random()

        assert before == after, (
            f"_ptest_splitter_result contaminated global state: {before} != {after}"
        )

    def test_ptest_gini_parallel_same_seed_same_result(self, classification_data):
        """Parallel Gini test with same seed should produce identical results."""
        from citrees._splitter import _ptest_gini_parallel_result

        x, y, threshold = classification_data

        result1 = _ptest_gini_parallel_result(
            x=x, y=y, threshold=threshold, n_resamples=500, random_state=42
        )
        result2 = _ptest_gini_parallel_result(
            x=x, y=y, threshold=threshold, n_resamples=500, random_state=42
        )

        assert result1 == result2, f"Same seed should give same result: {result1} != {result2}"

    def test_ptest_mse_parallel_same_seed_same_result(self, regression_data):
        """Parallel MSE test with same seed should produce identical results."""
        from citrees._splitter import _ptest_mse_parallel_result

        x, y, threshold = regression_data

        result1 = _ptest_mse_parallel_result(
            x=x, y=y, threshold=threshold, n_resamples=500, random_state=42
        )
        result2 = _ptest_mse_parallel_result(
            x=x, y=y, threshold=threshold, n_resamples=500, random_state=42
        )

        assert result1 == result2, f"Same seed should give same result: {result1} != {result2}"


@pytest.mark.skipif(numba.config.DISABLE_JIT, reason="JIT disabled: no compiled kernel to compare")
class TestJitParity:
    """Compiled Numba kernels must match their pure-Python source."""

    _RNG = np.random.default_rng(1718)
    _X = _RNG.standard_normal(120)
    _Y_CLF = (_X + _RNG.standard_normal(120) * 0.5 > 0).astype(np.int64)
    _Y_REG = 2.0 * _X + _RNG.standard_normal(120) * 0.5
    _Y_INT = np.array([0, 0, 1, 1, 2, 2, 0, 1], dtype=np.int64)
    _Y_FLOAT = np.array([0.5, 1.5, 2.5, 3.5, -1.0, 0.0], dtype=np.float64)
    _MASKS = np.vstack([_X <= -0.5, _X <= 0.0, _X <= 0.5])
    _N_LEFT = _MASKS.sum(axis=1).astype(np.int64)
    _OBS = np.array([0.30, 0.28, 0.31])
    _PERM = np.random.default_rng(3).uniform(0.25, 0.35, size=(40, 3))

    KERNELS = {
        "gini": (_splitter.gini, (_Y_INT,)),
        "entropy": (_splitter.entropy, (_Y_INT,)),
        "mse": (_splitter.mse, (_Y_FLOAT,)),
        "mae": (_splitter.mae, (_Y_FLOAT,)),
        "_ptest_gini_parallel_result": (
            _splitter._ptest_gini_parallel_result,
            (_X, _Y_CLF, 0.0, 250, 1718),
        ),
        "_ptest_entropy_parallel_result": (
            _splitter._ptest_entropy_parallel_result,
            (_X, _Y_CLF, 0.0, 250, 1718),
        ),
        "_ptest_mse_parallel_result": (
            _splitter._ptest_mse_parallel_result,
            (_X, _Y_REG, 0.0, 250, 1718),
        ),
        "_ptest_mae_parallel_result": (
            _splitter._ptest_mae_parallel_result,
            (_X, _Y_REG, 0.0, 250, 1718),
        ),
        "_beta_cdf": (_splitter._beta_cdf, (0.3, 2.0, 5.0)),
        "_gini_split_stat": (_splitter._gini_split_stat, (_Y_INT[:4], _Y_INT[4:], 4, 4, 0.5, 0.5)),
        "_entropy_split_stat": (
            _splitter._entropy_split_stat,
            (_Y_INT[:4], _Y_INT[4:], 4, 4, 0.5, 0.5),
        ),
        "_mse_split_stat": (_splitter._mse_split_stat, (_Y_FLOAT[:3], _Y_FLOAT[3:], 0.5, 0.5)),
        "_mae_split_stat": (_splitter._mae_split_stat, (_Y_FLOAT[:3], _Y_FLOAT[3:], 0.5, 0.5)),
        "_split_masks": (_splitter._split_masks, (_X, np.array([-0.5, 0.0, 0.5]))),
        "_clf_split_stats": (_splitter._clf_split_stats, (_Y_CLF, _MASKS, _N_LEFT, 0, np.empty(3))),
        "_reg_split_stats": (_splitter._reg_split_stats, (_Y_REG, _MASKS, _N_LEFT, 0, np.empty(3))),
        "_standardized_min_test": (_splitter._standardized_min_test, (_OBS, _PERM, 40)),
        "_ptest_maxt_clf_parallel_result": (
            _splitter._ptest_maxt_clf_parallel_result,
            (_Y_CLF, _MASKS, _N_LEFT, 0, 250, 1718),
        ),
        "_ptest_maxt_reg_parallel_result": (
            _splitter._ptest_maxt_reg_parallel_result,
            (_Y_REG, _MASKS, _N_LEFT, 0, 250, 1718),
        ),
        "_ptest_maxt_clf_parallel_batched_result": (
            _splitter._ptest_maxt_clf_parallel_batched_result,
            (_Y_CLF, _MASKS, _N_LEFT, 1, 250, 1718, 0.05, 0.95),
        ),
        "_ptest_maxt_reg_parallel_batched_result": (
            _splitter._ptest_maxt_reg_parallel_batched_result,
            (_Y_REG, _MASKS, _N_LEFT, 1, 250, 1718, 0.05, 0.95),
        ),
    }

    @pytest.mark.parametrize("name", sorted(KERNELS))
    def test_jit_matches_py_func(self, name, assert_jit_parity):
        """The compiled kernel and its Python source must return identical values."""
        fn, args = self.KERNELS[name]
        assert_jit_parity(fn, *args)

    def test_every_kernel_has_a_parity_case(self, assert_all_kernels_covered):
        """Fail when a Numba kernel is added to _splitter without a parity case."""
        assert_all_kernels_covered(_splitter, {fn for fn, _ in self.KERNELS.values()})


# =============================================================================
# MAX-TYPE THRESHOLD TEST
# =============================================================================


class TestMaxTypeThresholdTest:
    """One permutation test on the minimum impurity over all candidate thresholds."""

    @staticmethod
    def _data(seed: int, signal: float):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(300)
        y = (x * signal + rng.standard_normal(300) > 0).astype(np.int64)
        thresholds = np.quantile(x, np.linspace(0.05, 0.95, 16))
        return x, y, thresholds

    def test_returns_a_probability_and_a_candidate_threshold(self) -> None:
        x, y, thresholds = self._data(0, 1.0)
        p, best = _splitter.ptest_maxt(x, y, thresholds, "gini", 200, None, 0.05, 1718)
        assert 0.0 < p <= 1.0
        assert best in set(thresholds.tolist())

    def test_strong_signal_is_detected_and_best_threshold_is_central(self) -> None:
        x, y, thresholds = self._data(1, 3.0)
        p, best = _splitter.ptest_maxt(x, y, thresholds, "gini", 200, None, 0.05, 1718)
        assert p < 0.05
        # y flips sign at x = 0, so the impurity-minimizing threshold sits near 0.
        assert abs(best) < 0.5

    def test_best_threshold_is_the_standardized_minimizer(self) -> None:
        # The chosen threshold minimizes the observed impurity after each candidate
        # is standardized by the mean and standard deviation of its B + 1 values.
        x, y, thresholds = self._data(2, 2.0)
        _, best = _splitter.ptest_maxt(x, y, thresholds, "gini", 100, None, 0.05, 1718)
        masks, n_left = _splitter._split_masks(x, thresholds)
        observed = np.empty(len(thresholds))
        _splitter._clf_split_stats(y, masks, n_left, 0, observed)
        permuted = np.empty((100, len(thresholds)))
        for b in range(100):
            np.random.seed(1718 + b)
            y_perm = y.copy()
            np.random.shuffle(y_perm)
            _splitter._clf_split_stats(y_perm, masks, n_left, 0, permuted[b])
        stacked = np.vstack([observed, permuted])
        z = (observed - stacked.mean(axis=0)) / stacked.std(axis=0)
        assert best == thresholds[int(np.argmin(z))]

    def test_noisy_tail_candidate_does_not_win_by_variance_alone(self) -> None:
        # Heavy-tailed target: the raw-impurity minimum tends to isolate the
        # extreme observations at a tail threshold; the standardized statistic
        # should prefer the planted central step instead.
        rng = np.random.default_rng(11)
        x = rng.standard_normal(400)
        y = 1.0 * (x > 0.0) + rng.standard_t(1.5, size=400)
        thresholds = np.quantile(x, np.linspace(0.02, 0.98, 25))
        _, best = _splitter.ptest_maxt(x, y, thresholds, "mse", 200, None, 0.05, 1718)
        assert abs(best) < 0.6

    @pytest.mark.parametrize("splitter", ["gini", "entropy"])
    def test_classifier_splitters_agree_on_direction(self, splitter: str) -> None:
        x, y, thresholds = self._data(3, 2.5)
        p, _ = _splitter.ptest_maxt(x, y, thresholds, splitter, 200, None, 0.05, 1718)
        assert p < 0.05

    @pytest.mark.parametrize("splitter", ["mse", "mae"])
    def test_regressor_splitters_run(self, splitter: str) -> None:
        rng = np.random.default_rng(4)
        x = rng.standard_normal(300)
        y = 2.0 * (x > 0) + rng.standard_normal(300) * 0.5
        thresholds = np.quantile(x, np.linspace(0.05, 0.95, 16))
        p, best = _splitter.ptest_maxt(x, y, thresholds, splitter, 200, None, 0.05, 1718)
        assert p < 0.05 and abs(best) < 0.5

    def test_adaptive_and_full_agree_on_a_clear_case(self) -> None:
        x, y, thresholds = self._data(5, 3.0)
        p_full, best_full = _splitter.ptest_maxt(x, y, thresholds, "gini", 400, None, 0.05, 1718)
        p_adaptive, best_adaptive = _splitter.ptest_maxt(
            x, y, thresholds, "gini", 400, "adaptive", 0.05, 1718
        )
        assert best_full == best_adaptive
        assert p_full < 0.05 and p_adaptive < 0.05

    def test_degenerate_thresholds_return_one_and_no_threshold(self) -> None:
        x, y, _ = self._data(6, 1.0)
        # Every candidate puts all samples on one side, so there is no split to test
        # and no candidate attained the statistic.
        p, best = _splitter.ptest_maxt(
            x, y, np.array([x.max() + 1.0, x.max() + 2.0]), "gini", 100, None, 0.05, 1718
        )
        assert p == 1.0
        assert np.isnan(best)

    def test_empty_threshold_set_is_rejected(self) -> None:
        x, y, _ = self._data(6, 1.0)
        with pytest.raises(ValueError, match="at least one candidate threshold"):
            _splitter.ptest_maxt(x, y, np.array([]), "gini", 100, None, 0.05, 1718)

    def test_unknown_splitter_is_rejected(self) -> None:
        x, y, thresholds = self._data(7, 1.0)
        with pytest.raises(ValueError, match="no max-type split test"):
            _splitter.ptest_maxt(x, y, thresholds, "nonsense", 100, None, 0.05, 1718)

    def test_result_is_seed_deterministic(self) -> None:
        x, y, thresholds = self._data(8, 1.0)
        first = _splitter.ptest_maxt(x, y, thresholds, "gini", 300, None, 0.05, 42)
        second = _splitter.ptest_maxt(x, y, thresholds, "gini", 300, None, 0.05, 42)
        assert first == second

    @pytest.mark.slow
    def test_null_rejection_rate_is_controlled(self) -> None:
        # Response independent of x: the familywise rejection rate over 16
        # thresholds must stay at or below the nominal 5 percent (binomial slack).
        rng = np.random.default_rng(1718)
        rejections = 0
        trials = 200
        for trial in range(trials):
            x = rng.standard_normal(200)
            y = rng.integers(0, 2, 200).astype(np.int64)
            thresholds = np.quantile(x, np.linspace(0.05, 0.95, 16))
            p, _ = _splitter.ptest_maxt(x, y, thresholds, "gini", 200, None, 0.05, trial)
            rejections += p < 0.05
        assert rejections / trials < 0.09
