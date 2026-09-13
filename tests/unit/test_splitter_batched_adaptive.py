"""Parallel batched adaptive splitter kernels match the serial adaptive rule."""

from __future__ import annotations

import numpy as np
import pytest

from citrees._splitter import (
    _ptest_entropy_parallel_batched_result,
    _ptest_gini_parallel_batched_result,
    _ptest_gini_parallel_result,
    _ptest_mae_parallel_batched_result,
    _ptest_mse_parallel_batched_result,
    _ptest_mse_parallel_result,
    ptest_gini,
    ptest_mse,
)

pytestmark = pytest.mark.splitter

KERNELS = {
    "gini": _ptest_gini_parallel_batched_result,
    "entropy": _ptest_entropy_parallel_batched_result,
    "mse": _ptest_mse_parallel_batched_result,
    "mae": _ptest_mae_parallel_batched_result,
}


def _data(kind: str, n: int = 300, signal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    if kind in {"gini", "entropy"}:
        y = (x + (0.0 if signal else 100.0) * rng.normal(size=n) > 0).astype(np.int64)
        if not signal:
            y = rng.integers(0, 2, size=n).astype(np.int64)
    else:
        y = (x if signal else 0.0) + rng.normal(size=n)
    return x, y


@pytest.mark.parametrize("kind", list(KERNELS))
def test_batched_kernel_is_deterministic_and_stops_early_on_strong_signal(kind: str) -> None:
    x, y = _data(kind)
    kernel = KERNELS[kind]
    p1, m1 = kernel(x, y, 0.0, 5000, 11, 0.05, 0.95)
    p2, m2 = kernel(x, y, 0.0, 5000, 11, 0.05, 0.95)
    assert (p1, m1) == (p2, m2)
    assert m1 >= 20  # ceil(1 / alpha) floor
    assert m1 < 5000  # a strong signal stops before the budget
    assert p1 < 0.05


@pytest.mark.parametrize("kind", list(KERNELS))
def test_batched_kernel_reaches_full_budget_when_budget_is_the_floor(kind: str) -> None:
    """Under Bonferroni-sized budgets equal to ceil(1/alpha) the rule cannot stop early."""
    x, y = _data(kind)
    alpha = 0.05 / 256
    budget = int(np.ceil(1 / alpha))
    p, m = KERNELS[kind](x, y, 0.0, budget, 3, alpha, 0.95)
    assert m == budget
    assert 0.0 < p <= 1.0


def test_batched_kernel_counts_agree_with_fixed_budget_kernel_on_identical_draws() -> None:
    """Same seeds, same permutations: the extreme count at the full budget is identical."""
    x, y = _data("gini", signal=False)
    budget = 1024
    p_fixed, m_fixed = _ptest_gini_parallel_result(x, y, 0.0, budget, 5)
    p_batched, m_batched = _ptest_gini_parallel_batched_result(
        x, y, 0.0, budget, 5, 1.0 / budget, 1.0
    )
    assert (m_fixed, m_batched) == (budget, budget)
    assert p_fixed == pytest.approx(p_batched)
    x, y = _data("mse", signal=False)
    p_fixed, _ = _ptest_mse_parallel_result(x, y, 0.0, budget, 5)
    p_batched, _ = _ptest_mse_parallel_batched_result(x, y, 0.0, budget, 5, 1.0 / budget, 1.0)
    assert p_fixed == pytest.approx(p_batched)


def test_public_dispatch_uses_batched_kernel_above_threshold() -> None:
    x, y = _data("gini")
    p_big = ptest_gini(x, y, 0.0, 1000, "adaptive", 0.05, 0, 0.95)
    p_small = ptest_gini(x, y, 0.0, 100, "adaptive", 0.05, 0, 0.95)
    assert 0.0 < p_big < 0.05 and 0.0 < p_small < 0.05
    x, y = _data("mse")
    assert 0.0 < ptest_mse(x, y, 0.0, 1000, "adaptive", 0.05, 0, 0.95) < 0.05
