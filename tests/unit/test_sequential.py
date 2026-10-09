"""Tests for citrees._sequential.py."""

import numba
import numpy as np
import pytest
from scipy.stats import beta as scipy_beta

from citrees import _sequential
from citrees._sequential import (
    _beta_cdf,
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


@pytest.mark.skipif(numba.config.DISABLE_JIT, reason="JIT disabled: no compiled kernel to compare")
class TestJitParity:
    """Compiled Numba kernels must match their pure-Python source."""

    KERNELS = {
        "_beta_cdf": (_sequential._beta_cdf, (0.3, 2.0, 5.0)),
        "_adaptive_chunk": (_sequential._adaptive_chunk, (100, 5000, 20)),
        "_scan_adaptive_checkpoints": (
            _sequential._scan_adaptive_checkpoints,
            (np.zeros(64, dtype=np.int64), 0, 0, 64, 20, 0.05, 0.95),
        ),
    }

    @pytest.mark.parametrize("name", sorted(KERNELS))
    def test_jit_matches_py_func(self, name, assert_jit_parity):
        """The compiled kernel and its Python source must return identical values."""
        fn, args = self.KERNELS[name]
        assert_jit_parity(fn, *args)

    def test_every_kernel_has_a_parity_case(self, assert_all_kernels_covered):
        """Fail when a Numba kernel is added to _sequential without a parity case."""
        assert_all_kernels_covered(_sequential, {fn for fn, _ in self.KERNELS.values()})
