"""Tests for citrees._utils.py."""

from typing import Any

import numpy as np
import pytest

from citrees._utils import (
    bayesian_bootstrap_proba,
    calculate_max_value,
    classic_bootstrap_sample,
    classic_bootstrap_unsampled_idx,
    estimate_mean,
    estimate_proba,
    oversample_bootstrap_sample,
    oversample_bootstrap_unsampled_idx,
    split_data,
    stratified_bootstrap_sample,
    stratified_bootstrap_unsampled_idx,
    undersample_bootstrap_sample,
    undersample_bootstrap_unsampled_idx,
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

    def test_numpy_floating(self):
        """Test with numpy floating types (e.g., np.float64)."""
        result = calculate_max_value(n_values=100, desired_max=np.float64(0.5))
        assert result == 50
        result = calculate_max_value(n_values=100, desired_max=np.float32(0.25))
        assert result == 25


class TestSplitData:
    """Tests for split_data function."""

    def test_basic_split(self):
        """Test basic data splitting."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 0, 1, 1])
        threshold = 4.0  # Split on first feature

        X_left, y_left, X_right, y_right = split_data(X=X, y=y, feature=0, threshold=threshold)

        # First two samples should be left (x <= 4)
        assert len(X_left) == 2
        assert len(X_right) == 2
        assert np.array_equal(y_left, [0, 0])
        assert np.array_equal(y_right, [1, 1])

    def test_all_left(self):
        """Test all samples go left."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 2])

        X_left, y_left, X_right, y_right = split_data(X=X, y=y, feature=0, threshold=10.0)

        assert len(X_left) == 3
        assert len(X_right) == 0

    def test_all_right(self):
        """Test all samples go right."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 2])

        X_left, y_left, X_right, y_right = split_data(X=X, y=y, feature=0, threshold=0.0)

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


class TestUndersampleBootstrapSample:
    """Tests for undersample_bootstrap_sample function."""

    def test_balances_to_minority_count(self):
        """Each class should have exactly n_min samples when not truncated."""
        y = np.array([0, 0, 0, 0, 0, 1, 1])  # n_min = 2
        idx = undersample_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        assert len(idx) == 4
        y_sampled = y[idx]
        assert (y_sampled == 0).sum() == (y_sampled == 1).sum() == 2

    def test_truncates_to_max_samples(self):
        """When max_samples < K*n_min, output size is exactly max_samples."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # n_min = 4, K*n_min = 8
        idx = undersample_bootstrap_sample(y=y, max_samples=5, bayesian_bootstrap=False, random_state=42)
        assert len(idx) == 5
        y_sampled = y[idx]
        assert abs(int((y_sampled == 0).sum()) - int((y_sampled == 1).sum())) <= 1


class TestUndersampleBootstrapUnsampledIdx:
    """Tests for undersample_bootstrap_unsampled_idx function."""

    def test_returns_unsampled(self):
        """Returned indices should not appear in the sampled multiset."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        idx_unsampled = undersample_bootstrap_unsampled_idx(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        idx_sampled = undersample_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        for i in idx_unsampled:
            assert i not in idx_sampled


class TestOversampleBootstrapSample:
    """Tests for oversample_bootstrap_sample function."""

    def test_fixed_size_balanced_classes(self):
        """Output size should be max_samples and class counts differ by at most 1."""
        y = np.array([0, 0, 0, 0, 0, 1, 1])  # imbalanced
        idx = oversample_bootstrap_sample(y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42)
        assert len(idx) == len(y)
        y_sampled = y[idx]
        assert abs(int((y_sampled == 0).sum()) - int((y_sampled == 1).sum())) <= 1
        # Minority class should be oversampled relative to its original count here (2 -> 3 or 4)
        assert int((y_sampled == 1).sum()) >= 3

    def test_multiclass_balancing(self):
        """Multiclass counts should be as equal as possible (diff <= 1)."""
        y = np.array([0] * 10 + [1] * 2 + [2] * 5)
        idx = oversample_bootstrap_sample(y=y, max_samples=9, bayesian_bootstrap=False, random_state=42)
        y_sampled = y[idx]
        counts = np.array([(y_sampled == c).sum() for c in [0, 1, 2]], dtype=int)
        assert counts.sum() == 9
        assert counts.max() - counts.min() <= 1


class TestOversampleBootstrapUnsampledIdx:
    """Tests for oversample_bootstrap_unsampled_idx function."""

    def test_returns_unsampled(self):
        """Returned indices should not appear in the sampled multiset."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        idx_unsampled = oversample_bootstrap_unsampled_idx(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        idx_sampled = oversample_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=False, random_state=42
        )
        for i in idx_unsampled:
            assert i not in idx_sampled


class TestUtilsPyFunc:
    """Tests for utility functions via  for coverage.

    Numba @njit decorated functions compile to machine code, so pytest-cov
    cannot track line coverage. Using  accesses the original Python
    function, enabling coverage tracking.
    """

    # estimate_proba py_func tests
    def test_estimate_proba_py_func_binary(self):
        """Test estimate_proba with binary classes."""
        y = np.array([0, 0, 1, 1])
        result = estimate_proba(y=y, n_classes=2)
        expected = np.array([0.5, 0.5])
        assert np.allclose(result, expected)

    def test_estimate_proba_py_func_multiclass(self):
        """Test estimate_proba with multiple classes."""
        y = np.array([0, 1, 2])
        result = estimate_proba(y=y, n_classes=3)
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        assert np.allclose(result, expected)

    def test_estimate_proba_py_func_missing_class(self):
        """Test estimate_proba with missing class."""
        y = np.array([0, 0, 1, 1])
        result = estimate_proba(y=y, n_classes=3)
        expected = np.array([0.5, 0.5, 0.0])
        assert np.allclose(result, expected)

    def test_estimate_proba_consistency(self):
        """Verify estimate_proba JIT and py_func produce identical results."""
        y = np.array([0, 0, 1, 1, 2])
        jit_result = estimate_proba(y=y, n_classes=3)
        py_result = estimate_proba(y=y, n_classes=3)
        assert np.allclose(jit_result, py_result)

    # estimate_mean py_func tests
    def test_estimate_mean_py_func(self):
        """Test estimate_mean."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = estimate_mean(y)
        assert result == pytest.approx(3.0)

    def test_estimate_mean_py_func_single(self):
        """Test estimate_mean with single value."""
        y = np.array([5.0])
        result = estimate_mean(y)
        assert result == pytest.approx(5.0)

    def test_estimate_mean_consistency(self):
        """Verify estimate_mean JIT and py_func produce identical results."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert estimate_mean(y) == pytest.approx(estimate_mean(y))

    # split_data py_func tests
    def test_split_data_py_func_basic(self):
        """Test split_data basic split."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 0, 1, 1])
        X_left, y_left, X_right, y_right = split_data(X=X, y=y, feature=0, threshold=4.0)
        assert len(X_left) == 2
        assert len(X_right) == 2

    def test_split_data_py_func_all_left(self):
        """Test split_data all samples go left."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 2])
        X_left, y_left, X_right, y_right = split_data(X=X, y=y, feature=0, threshold=10.0)
        assert len(X_left) == 3
        assert len(X_right) == 0

    def test_split_data_py_func_all_right(self):
        """Test split_data all samples go right."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 1, 2])
        X_left, y_left, X_right, y_right = split_data(X=X, y=y, feature=0, threshold=0.0)
        assert len(X_left) == 0
        assert len(X_right) == 3

    def test_split_data_consistency(self):
        """Verify split_data JIT and py_func produce identical results."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 2])
        jit_result = split_data(X=X, y=y, feature=0, threshold=2.5)
        py_result = split_data(X=X, y=y, feature=0, threshold=2.5)
        for jit_arr, py_arr in zip(jit_result, py_result, strict=False):
            assert np.allclose(jit_arr, py_arr)

    # bayesian_bootstrap_proba py_func tests
    def test_bayesian_bootstrap_proba_py_func_sums_to_one(self):
        """Test bayesian_bootstrap_proba sums to 1."""
        proba = bayesian_bootstrap_proba(n=100, random_state=42)
        assert np.isclose(proba.sum(), 1.0)

    def test_bayesian_bootstrap_proba_py_func_all_positive(self):
        """Test bayesian_bootstrap_proba all positive."""
        proba = bayesian_bootstrap_proba(n=50, random_state=42)
        assert (proba > 0).all()

    def test_bayesian_bootstrap_proba_py_func_shape(self):
        """Test bayesian_bootstrap_proba returns correct shape."""
        proba = bayesian_bootstrap_proba(n=100, random_state=42)
        assert proba.shape == (100,)

    def test_bayesian_bootstrap_proba_consistency(self):
        """Verify bayesian_bootstrap_proba JIT and py_func produce identical results."""
        jit_result = bayesian_bootstrap_proba(n=100, random_state=42)
        py_result = bayesian_bootstrap_proba(n=100, random_state=42)
        assert np.allclose(jit_result, py_result)


class TestBugFixes:
    """Tests for specific bug fixes."""

    def test_bug2_bayesian_bootstrap_different_seeds_per_class(self):
        """Bug 2: Verify different classes receive different bootstrap probabilities.

        Previously, bayesian_bootstrap_proba was called with the same random_state
        for all classes, causing identical probabilities. Now each class gets
        random_state + class_index.
        """
        # Create 3-class dataset
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        # Get bootstrap samples with bayesian_bootstrap=True
        idx1 = stratified_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=True, random_state=42
        )
        idx2 = stratified_bootstrap_sample(
            y=y, max_samples=len(y), bayesian_bootstrap=True, random_state=42
        )

        # Should be reproducible
        assert np.array_equal(idx1, idx2)

        # The key test: simulate what the old buggy code would produce
        # Old code: all classes got same probabilities
        # New code: each class gets different probabilities (random_state + j)
        proba_class0 = bayesian_bootstrap_proba(n=5, random_state=42)
        proba_class1 = bayesian_bootstrap_proba(n=5, random_state=43)  # 42 + 1
        proba_class2 = bayesian_bootstrap_proba(n=5, random_state=44)  # 42 + 2

        # Classes should have DIFFERENT probabilities
        assert not np.allclose(proba_class0, proba_class1)
        assert not np.allclose(proba_class1, proba_class2)
        assert not np.allclose(proba_class0, proba_class2)

    def test_bug3_stratified_bootstrap_exact_sample_count(self):
        """Bug 3: Verify stratified bootstrap returns exactly max_samples.

        Previously, using round() per class could cause total != max_samples.
        Now uses proper integer allocation (largest remainder method).
        """
        # Test various configurations that previously caused mismatches
        test_cases = [
            # (n_per_class, n_classes, max_samples)
            (5, 3, 4),  # Original bug report case
            (3, 2, 3),  # Previously got 4 instead of 3
            (3, 3, 4),  # Previously got 3 instead of 4
            (3, 4, 2),  # Previously got 0 instead of 2
            (10, 5, 7),  # General case
        ]

        for n_per_class, n_classes, max_samples in test_cases:
            y = np.concatenate([np.full(n_per_class, j) for j in range(n_classes)])
            idx = stratified_bootstrap_sample(
                y=y, max_samples=max_samples, bayesian_bootstrap=False, random_state=42
            )
            assert len(idx) == max_samples, (
                f"Expected {max_samples} samples, got {len(idx)} "
                f"for n_per_class={n_per_class}, n_classes={n_classes}"
            )

    def test_split_data_requires_no_nan(self):
        """Verify that NaN/Inf inputs are rejected at fit time.

        split_data uses fastmath=True for performance, which has undefined behavior
        with NaN. Instead of removing fastmath, we validate inputs upstream in
        _validate_data_fit() and reject NaN/Inf with a clear error message.
        """
        from citrees import ConditionalInferenceTreeClassifier

        X_with_nan = np.array([[1.0], [np.nan], [3.0]])
        y = np.array([0, 1, 0])

        clf = ConditionalInferenceTreeClassifier(random_state=42)

        with pytest.raises(ValueError, match="NaN or Inf"):
            clf.fit(X_with_nan, y)

        X_with_inf = np.array([[1.0], [np.inf], [3.0]])
        with pytest.raises(ValueError, match="NaN or Inf"):
            clf.fit(X_with_inf, y)
