"""The adaptive stopping rule's exact null size, as reported in the manuscripts."""

from __future__ import annotations

import pytest
from scipy.stats import beta

from citrees._sequential import _ADAPTIVE_BATCH_SIZE, _beta_cdf
from paper.theory.adaptive_stopping_size import checkpoints, null_size

pytestmark = pytest.mark.paper


def test_reported_sizes() -> None:
    assert null_size(0.05, 0.95, 32, 999) == pytest.approx(0.04878, abs=5e-5)
    assert null_size(0.05, 0.95, 32, 999, strict=False) == pytest.approx(0.04948, abs=5e-5)
    assert null_size(0.05, 0.95, 32, 1537) == pytest.approx(0.04907, abs=5e-5)
    assert null_size(0.05, 0.80, 32, 999) == pytest.approx(0.05220, abs=5e-5)
    assert _ADAPTIVE_BATCH_SIZE == 32


def test_size_below_alpha_at_default_confidence_across_budgets() -> None:
    assert all(null_size(0.05, 0.95, 32, B) <= 0.05 for B in range(20, 3001, 97))


def test_minimum_budget_reduces_to_the_fixed_b_test() -> None:
    for m in (1, 7, 60, 257):
        alpha = 0.05 / m
        B = 20 * m
        assert checkpoints(alpha, 32, B) == [B]
        assert null_size(alpha, 0.95, 32, B) == pytest.approx(int(alpha * (B + 1)) / (B + 1))


def test_code_beta_cdf_matches_reference_on_every_stop_decision() -> None:
    for m in checkpoints(0.05, 32, 999):
        for k in range(m + 1):
            code = _beta_cdf(0.05, 1.0 + k, 1.0 + m - k)
            ref = float(beta.cdf(0.05, 1 + k, 1 + m - k))
            assert ((code >= 0.95) or (1 - code >= 0.95)) == ((ref >= 0.95) or (1 - ref >= 0.95))


@pytest.mark.paper
def test_rejection_probability_is_nonincreasing_in_theta():
    """Monotonicity step of the size proposition: g(theta) nonincreasing and every
    significance stop reports a value strictly below alpha at the shipped settings."""
    import numpy as np

    from paper.theory.adaptive_stopping_size import rejection_given_theta

    for alpha, B in ((0.05, 999), (0.05, 79), (0.05 / 20, 400)):
        thetas = np.linspace(0.001, 0.3, 120)
        values = []
        for theta in thetas:
            g, sig_below = rejection_given_theta(theta, alpha, 0.95, 32, B)
            assert sig_below
            values.append(g)
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:], strict=False))
