"""Sequential permutation testing with early stopping.

Three methods:
1. Simple: Futility + significance stopping (can inflate Type I error)
2. Adaptive: Bayesian Beta CDF posterior-confidence stopping (speed-oriented; the returned value is a Monte Carlo
   estimate evaluated at a stopping time, not a fixed-B permutation p-value)
3. Adaptive Batched: Same as adaptive, but checks the stopping criterion every `batch_size` permutations instead of
   every single permutation. Eliminates ~97% of Beta CDF evaluations in calibration runs; these
   adaptive outputs are not theorem-level fixed-B p-values.
"""

from math import exp, lgamma, log

import numpy as np
from numba import njit

_ADAPTIVE_BATCH_SIZE = 32


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


# Note: Uses np.random.seed() for determinism and to mirror the Numba-safe RNG
# pattern used elsewhere in the codebase (Numba doesn't support default_rng()
# inside @njit).
# Note: Uses np.random.seed() for determinism and to mirror the Numba-safe RNG
# pattern used elsewhere in the codebase (Numba doesn't support default_rng()
# inside @njit).
# Note: Uses np.random.seed() for determinism and to mirror the Numba-safe RNG
# pattern used elsewhere in the codebase (Numba doesn't support default_rng()
# inside @njit).
# Parallel adaptive kernels evaluate permutations in chunks and then replay the
# serial stopping rule over the chunk: the posterior is checked at every multiple
# of _ADAPTIVE_BATCH_SIZE permutations (and at the budget) once the floor
# ceil(1 / alpha) is reached, exactly as the serial path does. Extra permutations
# in the chunk past a stop are discarded, so the stopping time, the returned
# (k + 1) / (m + 1) estimate, and the exact null size are unchanged; only the
# parallel efficiency differs. Chunks grow geometrically from one batch so a rule
# that stops early wastes at most as much work as it has already done.
_ADAPTIVE_CHUNK_MAX = 1024


@njit(cache=True, nogil=True)
def _adaptive_chunk(m: int, n_resamples: int, min_resamples: int) -> int:
    """Number of permutations to draw in the next parallel chunk.

    No checkpoint can fire before ``min_resamples`` permutations, so the first
    chunk runs straight to the floor; when the budget equals the floor (the
    Bonferroni ``minimum`` setting) the whole test is one parallel chunk.
    """
    grown = m if m > _ADAPTIVE_BATCH_SIZE else _ADAPTIVE_BATCH_SIZE
    if m < min_resamples and min_resamples - m > grown:
        grown = min_resamples - m
    if grown > _ADAPTIVE_CHUNK_MAX:
        grown = _ADAPTIVE_CHUNK_MAX
    remaining = n_resamples - m
    return grown if grown < remaining else remaining


@njit(cache=True, nogil=True)
def _scan_adaptive_checkpoints(
    flags: np.ndarray,
    m: int,
    extreme_count: int,
    n_resamples: int,
    min_resamples: int,
    alpha: float,
    confidence: float,
) -> tuple[bool, int, int]:
    """Replay the serial adaptive rule over one chunk of extreme-indicator flags.

    Returns ``(stopped, m, extreme_count)``. When ``stopped`` is true, ``m`` and
    ``extreme_count`` are the values at the stopping checkpoint; otherwise they
    cover the whole chunk.
    """
    for j in range(flags.shape[0]):
        extreme_count += flags[j]
        m += 1
        if m >= min_resamples and (m % _ADAPTIVE_BATCH_SIZE == 0 or m == n_resamples):
            prob_sig = _beta_cdf(alpha, 1.0 + extreme_count, 1.0 + m - extreme_count)
            if prob_sig >= confidence or (1.0 - prob_sig) >= confidence:
                return True, m, extreme_count
    return False, m, extreme_count
