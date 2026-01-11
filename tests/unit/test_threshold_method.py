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
