"""Tests for citrees._selector.py."""

import numpy as np
import pytest
from scipy.stats import kstest

from citrees._selector import ptest_mc


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
            early_stopping=False,
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
    assert 0.02 < fp_rate < 0.10, f"False positive rate {fp_rate:.3f} outside expected range [0.02, 0.10]"


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
        early_stopping=False,
        alpha=0.05,
        random_state=42,
    )

    # P-value should be 1/(n_resamples+1) = 1/101 ≈ 0.0099, never 0
    expected_min = 1 / (n_resamples + 1)
    assert pval > 0, "P-value should never be exactly zero"
    assert pval == pytest.approx(expected_min, rel=0.01), (
        f"With strong signal, p-value should be minimum possible: {expected_min:.4f}, got {pval:.4f}"
    )
