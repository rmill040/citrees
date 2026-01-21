"""Tests for RNG reproducibility in permutation tests.

Verifies that:
1. Same random_state produces identical results
2. Different random_state produces different results
3. Pure Python functions use isolated RNG (no global state contamination)
4. Parallel permutation tests are deterministic
"""

import numpy as np
import pytest

from citrees._selector import _ptest, _ptest_mc_parallel, _ptest_multi, _ptest_pc_parallel, mc
from citrees._splitter import _ptest as _ptest_splitter
from citrees._splitter import _ptest_gini_parallel, _ptest_mse_parallel, gini


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

        # Note: There's a small probability they could be equal by chance,
        # but with 100 resamples this is extremely unlikely
        assert pval1 != pval2, f"Different seeds should give different results: {pval1} == {pval2}"

    def test_ptest_no_global_state_contamination(self, classification_data):
        """_ptest should not contaminate global RNG state."""
        x, y = classification_data

        # Set global seed and get a random number
        np.random.seed(123)
        before = np.random.random()

        # Call _ptest (should use isolated RNG)
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


class TestParallelRNGReproducibility:
    """Test RNG reproducibility for parallel permutation tests."""

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
        x, y = classification_data

        pval1 = _ptest_mc_parallel(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)
        pval2 = _ptest_mc_parallel(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"

    def test_ptest_mc_parallel_different_seed_different_result(self, classification_data):
        """Parallel MC test with different seed should produce different results."""
        x, y = classification_data

        pval1 = _ptest_mc_parallel(x=x, y=y, n_classes=2, n_resamples=500, random_state=42)
        pval2 = _ptest_mc_parallel(x=x, y=y, n_classes=2, n_resamples=500, random_state=99)

        assert pval1 != pval2, f"Different seeds should give different results: {pval1} == {pval2}"

    def test_ptest_pc_parallel_same_seed_same_result(self, regression_data):
        """Parallel PC test with same seed should produce identical results."""
        x, y = regression_data

        pval1 = _ptest_pc_parallel(x=x, y=y, n_resamples=500, random_state=42)
        pval2 = _ptest_pc_parallel(x=x, y=y, n_resamples=500, random_state=42)

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"


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
        x, y, threshold = regression_data

        pval1 = _ptest_mse_parallel(x=x, y=y, threshold=threshold, n_resamples=500, random_state=42)
        pval2 = _ptest_mse_parallel(x=x, y=y, threshold=threshold, n_resamples=500, random_state=42)

        assert pval1 == pval2, f"Same seed should give same result: {pval1} != {pval2}"
