"""Tests for citrees._threshold_method.py."""

import numpy as np
import pytest

from citrees._threshold_method import exact, histogram, percentile, random


class TestExact:
    """Tests for the exact threshold method."""

    def test_returns_midpoints(self):
        """Test that exact returns midpoints between unique values."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        thresholds = exact(x)
        expected = np.array([1.5, 2.5, 3.5])
        assert np.allclose(thresholds, expected)

    def test_handles_duplicates(self):
        """Test that duplicates are handled correctly."""
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        thresholds = exact(x)
        expected = np.array([1.5, 2.5])
        assert np.allclose(thresholds, expected)

    def test_single_value_empty(self):
        """Test that single unique value returns empty array."""
        x = np.array([1.0, 1.0, 1.0])
        thresholds = exact(x)
        assert len(thresholds) == 0

    def test_two_values_one_threshold(self):
        """Test that two unique values return one threshold."""
        x = np.array([1.0, 5.0])
        thresholds = exact(x)
        assert len(thresholds) == 1
        assert thresholds[0] == pytest.approx(3.0)

    def test_ignores_max_thresholds(self):
        """Test that exact ignores max_thresholds parameter."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        thresholds = exact(x, max_thresholds=2)
        # Should still return all midpoints
        assert len(thresholds) == 4


class TestRandom:
    """Tests for the random threshold method."""

    def test_returns_subset_of_midpoints(self):
        """Test that random returns a subset of midpoints."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        thresholds = random(x, max_thresholds=3, random_state=42)
        assert len(thresholds) == 3
        # All returned values should be valid midpoints
        all_midpoints = exact(x)
        for t in thresholds:
            assert np.any(np.isclose(all_midpoints, t))

    def test_max_thresholds_respected(self):
        """Test that max_thresholds is respected."""
        x = np.arange(100, dtype=float)
        thresholds = random(x, max_thresholds=10, random_state=42)
        assert len(thresholds) == 10

    def test_max_thresholds_capped_by_data(self):
        """Test that max_thresholds is capped by available midpoints."""
        x = np.array([1.0, 2.0, 3.0])  # Only 2 midpoints possible
        thresholds = random(x, max_thresholds=10, random_state=42)
        assert len(thresholds) == 2

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with same seed."""
        x = np.arange(100, dtype=float)
        t1 = random(x, max_thresholds=5, random_state=42)
        t2 = random(x, max_thresholds=5, random_state=42)
        assert np.allclose(np.sort(t1), np.sort(t2))

    def test_different_seeds_different_results(self):
        """Test that different seeds give different results."""
        x = np.arange(100, dtype=float)
        t1 = random(x, max_thresholds=5, random_state=42)
        t2 = random(x, max_thresholds=5, random_state=123)
        # Very unlikely to be the same
        assert not np.allclose(np.sort(t1), np.sort(t2))


class TestPercentile:
    """Tests for the percentile threshold method."""

    def test_returns_percentiles(self):
        """Test that percentile returns percentile-based thresholds."""
        x = np.arange(101, dtype=float)  # 0 to 100
        thresholds = percentile(x, max_thresholds=5, random_state=None)
        # Should return values near 0%, 25%, 50%, 75%, 100% of midpoints
        assert len(thresholds) <= 5

    def test_max_thresholds_respected(self):
        """Test that max_thresholds is approximately respected."""
        x = np.arange(1000, dtype=float)
        thresholds = percentile(x, max_thresholds=10, random_state=None)
        # Due to uniqueness, might be slightly different
        assert len(thresholds) <= 10

    def test_unique_values_returned(self):
        """Test that returned thresholds are unique."""
        x = np.arange(100, dtype=float)
        thresholds = percentile(x, max_thresholds=10, random_state=None)
        assert len(thresholds) == len(np.unique(thresholds))

    def test_handles_small_data(self):
        """Test handling of small datasets."""
        x = np.array([1.0, 2.0, 3.0])
        thresholds = percentile(x, max_thresholds=10, random_state=None)
        assert len(thresholds) <= 2  # Only 2 midpoints available


class TestHistogram:
    """Tests for the histogram threshold method."""

    def test_returns_bin_edges(self):
        """Test that histogram returns bin edges."""
        x = np.arange(100, dtype=float)
        thresholds = histogram(x, max_thresholds=10, random_state=None)
        # Should return roughly max_thresholds + 1 bin edges
        assert len(thresholds) > 0

    def test_max_thresholds_affects_bins(self):
        """Test that max_thresholds affects number of bins."""
        x = np.arange(1000, dtype=float)
        t5 = histogram(x, max_thresholds=5, random_state=None)
        t20 = histogram(x, max_thresholds=20, random_state=None)
        # More bins should give more thresholds
        assert len(t20) >= len(t5)

    def test_unique_values_returned(self):
        """Test that returned thresholds are unique."""
        x = np.arange(100, dtype=float)
        thresholds = histogram(x, max_thresholds=10, random_state=None)
        assert len(thresholds) == len(np.unique(thresholds))

class TestThresholdMethodComparison:
    """Tests comparing different threshold methods."""

    def test_all_methods_return_valid_thresholds(self):
        """Test that all methods return values within data range."""
        np.random.seed(42)
        x = np.random.randn(100)
        x_min, x_max = x.min(), x.max()

        for method, kwargs in [
            (exact, {}),
            (random, {"max_thresholds": 10, "random_state": 42}),
            (percentile, {"max_thresholds": 10}),
            (histogram, {"max_thresholds": 10}),
        ]:
            thresholds = method(x, **kwargs)
            if len(thresholds) > 0:
                assert thresholds.min() >= x_min
                assert thresholds.max() <= x_max

    def test_exact_returns_most_thresholds(self):
        """Test that exact method returns the most thresholds."""
        x = np.arange(100, dtype=float)
        t_exact = exact(x)
        t_random = random(x, max_thresholds=10, random_state=42)
        t_percentile = percentile(x, max_thresholds=10)
        t_histogram = histogram(x, max_thresholds=10)

        assert len(t_exact) >= len(t_random)
        assert len(t_exact) >= len(t_percentile)
        assert len(t_exact) >= len(t_histogram)

    def test_all_methods_handle_edge_cases(self):
        """Test that all methods handle edge cases gracefully."""
        # Two values (single value is edge case that histogram can't handle)
        x_two = np.array([1.0, 2.0])
        assert len(exact(x_two)) == 1
        assert len(random(x_two, max_thresholds=5, random_state=42)) == 1
        assert len(percentile(x_two, max_thresholds=5)) == 1

    def test_exact_handles_single_value(self):
        """Test exact handles single unique value."""
        x_single = np.array([1.0, 1.0, 1.0])
        assert len(exact(x_single)) == 0

    def test_random_handles_single_value(self):
        """Test random handles single unique value."""
        x_single = np.array([1.0, 1.0, 1.0])
        assert len(random(x_single, max_thresholds=5, random_state=42)) == 0

    def test_percentile_handles_single_value(self):
        """Test percentile handles single unique value."""
        x_single = np.array([1.0, 1.0, 1.0])
        assert len(percentile(x_single, max_thresholds=5)) == 0


class TestThresholdMethodPyFunc:
    """Tests for threshold methods via .py_func for coverage."""

    # Exact py_func tests
    def test_exact_py_func_midpoints(self):
        """Test exact.py_func returns midpoints."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        thresholds = exact.py_func(x)
        expected = np.array([1.5, 2.5, 3.5])
        assert np.allclose(thresholds, expected)

    def test_exact_py_func_duplicates(self):
        """Test exact.py_func handles duplicates."""
        x = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
        thresholds = exact.py_func(x)
        expected = np.array([1.5, 2.5])
        assert np.allclose(thresholds, expected)

    def test_exact_py_func_single_value(self):
        """Test exact.py_func with single unique value."""
        x = np.array([1.0, 1.0, 1.0])
        thresholds = exact.py_func(x)
        assert len(thresholds) == 0

    def test_exact_py_func_two_values(self):
        """Test exact.py_func with two unique values."""
        x = np.array([1.0, 5.0])
        thresholds = exact.py_func(x)
        assert len(thresholds) == 1
        assert thresholds[0] == pytest.approx(3.0)

    def test_exact_consistency(self):
        """Verify exact JIT and py_func produce identical results."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(exact(x), exact.py_func(x))

    # Random py_func tests
    def test_random_py_func_subset(self):
        """Test random.py_func returns subset of midpoints."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        thresholds = random.py_func(x, max_thresholds=3, random_state=42)
        assert len(thresholds) == 3
        # All returned values should be valid midpoints
        all_midpoints = exact(x)
        for t in thresholds:
            assert np.any(np.isclose(all_midpoints, t))

    def test_random_py_func_max_thresholds(self):
        """Test random.py_func respects max_thresholds."""
        x = np.arange(100, dtype=float)
        thresholds = random.py_func(x, max_thresholds=10, random_state=42)
        assert len(thresholds) == 10

    def test_random_py_func_reproducible(self):
        """Test random.py_func is reproducible with same seed."""
        x = np.arange(100, dtype=float)
        t1 = random.py_func(x, max_thresholds=5, random_state=42)
        t2 = random.py_func(x, max_thresholds=5, random_state=42)
        assert np.allclose(np.sort(t1), np.sort(t2))

    def test_random_consistency(self):
        """Verify random JIT and py_func produce identical results."""
        x = np.arange(50, dtype=float)
        assert np.allclose(
            np.sort(random(x, max_thresholds=5, random_state=42)),
            np.sort(random.py_func(x, max_thresholds=5, random_state=42))
        )

    # Percentile py_func tests
    def test_percentile_py_func_returns_values(self):
        """Test percentile.py_func returns thresholds."""
        x = np.arange(101, dtype=float)
        thresholds = percentile.py_func(x, max_thresholds=5, random_state=None)
        assert len(thresholds) <= 5

    def test_percentile_py_func_unique(self):
        """Test percentile.py_func returns unique values."""
        x = np.arange(100, dtype=float)
        thresholds = percentile.py_func(x, max_thresholds=10, random_state=None)
        assert len(thresholds) == len(np.unique(thresholds))

    def test_percentile_py_func_small_data(self):
        """Test percentile.py_func with small dataset."""
        x = np.array([1.0, 2.0, 3.0])
        thresholds = percentile.py_func(x, max_thresholds=10, random_state=None)
        assert len(thresholds) <= 2

    def test_percentile_consistency(self):
        """Verify percentile JIT and py_func produce identical results."""
        x = np.arange(50, dtype=float)
        assert np.allclose(
            percentile(x, max_thresholds=5, random_state=None),
            percentile.py_func(x, max_thresholds=5, random_state=None)
        )

    # Histogram py_func tests
    def test_histogram_py_func_returns_edges(self):
        """Test histogram.py_func returns bin edges."""
        x = np.arange(100, dtype=float)
        thresholds = histogram.py_func(x, max_thresholds=10, random_state=None)
        assert len(thresholds) > 0

    def test_histogram_py_func_max_thresholds(self):
        """Test histogram.py_func respects max_thresholds."""
        x = np.arange(1000, dtype=float)
        t5 = histogram.py_func(x, max_thresholds=5, random_state=None)
        t20 = histogram.py_func(x, max_thresholds=20, random_state=None)
        assert len(t20) >= len(t5)

    def test_histogram_py_func_unique(self):
        """Test histogram.py_func returns unique values."""
        x = np.arange(100, dtype=float)
        thresholds = histogram.py_func(x, max_thresholds=10, random_state=None)
        assert len(thresholds) == len(np.unique(thresholds))

    def test_histogram_consistency(self):
        """Verify histogram JIT and py_func produce identical results."""
        x = np.arange(100, dtype=float)
        assert np.allclose(
            histogram(x, max_thresholds=10, random_state=None),
            histogram.py_func(x, max_thresholds=10, random_state=None)
        )
