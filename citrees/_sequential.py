"""Sequential permutation testing with early stopping.

Two methods:
1. Simple: Futility + significance stopping (can inflate Type I error)
2. Adaptive: Bayesian Beta CDF posterior-confidence stopping (speed-oriented; the returned value is a Monte Carlo
   estimate evaluated at a stopping time, not a fixed-B permutation p-value)
"""

from math import ceil, exp, lgamma, log
from typing import Any

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _beta_cdf(x: float, a: float, b: float) -> float:
    """Beta CDF using continued fraction expansion (Lentz's algorithm)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    if x > (a + 1) / (a + b + 2):
        return 1.0 - _beta_cdf(1 - x, b, a)

    max_iter = 200
    eps = 1e-10

    log_prefix = a * log(x) + b * log(1 - x) - log(a)
    log_prefix += lgamma(a + b) - lgamma(a) - lgamma(b)

    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    result = d

    for m in range(1, max_iter):
        m2 = 2 * m

        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        result *= d * c

        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = d * c
        result *= delta

        if abs(delta - 1.0) < eps:
            break

    return exp(log_prefix) * result


# Note: Uses np.random.seed() because this function is compiled with @njit below,
# and Numba doesn't support default_rng() inside @njit.
def _ptest_sequential_simple(
    *,
    func: Any,
    func_arg: Any,
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    alpha: float,
    random_state: int,
) -> float:
    """Sequential permutation test with simple futility + significance stopping.

    Stops early when:
    1. Significance: current p-value < alpha (after min_resamples)
    2. Futility: best possible p-value >= alpha (cannot reject)

    Note: This method can inflate Type I error. Use adaptive for a more conservative posterior-confidence stopping
    rule (or disable early stopping for fixed-B p-values).
    """
    np.random.seed(random_state)
    min_resamples = ceil(1.0 / alpha)
    n_resamples = max(n_resamples, min_resamples)

    theta = np.abs(func(x, y, func_arg, random_state=random_state))
    y_ = y.copy()
    extreme_count = 0

    for i in range(n_resamples):
        np.random.shuffle(y_)
        theta_p = np.abs(func(x, y_, func_arg, random_state=random_state))
        if theta_p >= theta:
            extreme_count += 1

        n = i + 1
        current_pval = (extreme_count + 1) / (n + 1)

        if n >= min_resamples:
            if current_pval < alpha:
                return current_pval

            best_possible = (extreme_count + 1) / (n_resamples + 1)
            if best_possible >= alpha and extreme_count >= 3:
                return current_pval

    return (extreme_count + 1) / (n_resamples + 1)


_ptest_sequential_simple_compiled = njit(cache=True, fastmath=True, nogil=True)(
    _ptest_sequential_simple
)


# Note: Uses np.random.seed() because this function is compiled with @njit below,
# and Numba doesn't support default_rng() inside @njit.
def _ptest_sequential_adaptive(
    *,
    func: Any,
    func_arg: Any,
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    alpha: float,
    confidence: float,
    random_state: int,
) -> float:
    """Sequential permutation test with Bayesian adaptive stopping.

    Uses Beta posterior to estimate probability that true p-value < alpha.
    Stops when confident about significance or non-significance.

    Note: This is a posterior-confidence stopping heuristic. The returned value
    is the +1 Monte Carlo estimate evaluated at a stopping time (not a fixed-B
    permutation p-value). For fixed-B p-value guarantees, disable early
    stopping.
    """
    np.random.seed(random_state)
    min_resamples = ceil(1.0 / alpha)
    n_resamples = max(n_resamples, min_resamples)

    theta = np.abs(func(x, y, func_arg, random_state=random_state))
    y_ = y.copy()
    extreme_count = 0

    for i in range(n_resamples):
        np.random.shuffle(y_)
        theta_p = np.abs(func(x, y_, func_arg, random_state=random_state))
        if theta_p >= theta:
            extreme_count += 1

        n = i + 1
        if n >= min_resamples:
            a = 1.0 + extreme_count
            b = 1.0 + n - extreme_count
            prob_sig = _beta_cdf(alpha, a, b)

            if prob_sig >= confidence:
                return (extreme_count + 1) / (n + 1)
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (n + 1)

    return (extreme_count + 1) / (n_resamples + 1)


_ptest_sequential_adaptive_compiled = njit(cache=True, fastmath=True, nogil=True)(
    _ptest_sequential_adaptive
)
