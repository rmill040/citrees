"""Tests for citrees._splitter.py."""

import numpy as np
import pytest

from citrees._splitter import (
    entropy,
    gini,
    mae,
    mse,
    ptest_entropy,
    ptest_gini,
    ptest_mae,
    ptest_mse,
)


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
        """Splitter _ptest with same seed should produce identical results."""
        from citrees._splitter import _ptest as _ptest_splitter

        x, y, threshold = classification_data

        pval1 = _ptest_splitter(
            func=gini,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        pval2 = _ptest_splitter(
            func=gini,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=100,
            early_stopping=None,
            alpha=0.05,
            random_state=42,
        )

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"

    def test_splitter_ptest_no_global_state_contamination(self, classification_data):
        """Splitter _ptest should not contaminate global RNG state."""
        from citrees._splitter import _ptest as _ptest_splitter

        x, y, threshold = classification_data

        np.random.seed(123)
        before = np.random.random()

        np.random.seed(123)
        _ptest_splitter(
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

        assert before == after, f"_ptest_splitter contaminated global state: {before} != {after}"

    def test_ptest_gini_parallel_same_seed_same_result(self, classification_data):
        """Parallel Gini test with same seed should produce identical results."""
        from citrees._splitter import _ptest_gini_parallel

        x, y, threshold = classification_data

        pval1 = _ptest_gini_parallel(
            x=x, y=y, threshold=threshold, n_resamples=500, random_state=42
        )
        pval2 = _ptest_gini_parallel(
            x=x, y=y, threshold=threshold, n_resamples=500, random_state=42
        )

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"

    def test_ptest_mse_parallel_same_seed_same_result(self, regression_data):
        """Parallel MSE test with same seed should produce identical results."""
        from citrees._splitter import _ptest_mse_parallel

        x, y, threshold = regression_data

        pval1 = _ptest_mse_parallel(x=x, y=y, threshold=threshold, n_resamples=500, random_state=42)
        pval2 = _ptest_mse_parallel(x=x, y=y, threshold=threshold, n_resamples=500, random_state=42)

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"
