from functools import partial
from math import ceil
from typing import Any

import numba
import numpy as np
from numba import njit
from numba import prange as _numba_prange
from sklearn.feature_selection import mutual_info_classif

from citrees._permutation import PermutationTestResult, record_permutation_test
from citrees._registry import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    RegressorSelectors,
    RegressorSelectorTests,
)
from citrees._sequential import _beta_cdf
from citrees._types import EarlyStopping, EarlyStoppingOption

prange: Any = _numba_prange

# Threshold for using parallel permutation tests
_PARALLEL_THRESHOLD = 200
_ADAPTIVE_BATCH_SIZE = 32

# P-value correction: Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero."
# SAGMB 9(1):39. https://pubmed.ncbi.nlm.nih.gov/21044043/
# Use p = (b+1)/(m+1) instead of p = b/m to avoid p=0.
# Note: min_resamples = ceil(1/alpha) remains valid since 1/(m+1) < alpha.


def _ptest_result(
    *,
    func: Any,
    func_arg: Any,
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> PermutationTestResult:
    """Calculate the achieved significance level using a permutation test.

    Parameters
    ----------
    func : Any
        Feature selection function to use in permutation testing.

    func_arg : Any
        Single function argument.

    x : np.ndarray
        Input data, usually the feature in the (x, y) pair.

    y : np.ndarray
        Input data, usually the target in the (x, y) pair.

    n_resamples : int
        Number of resamples.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method:
        - "adaptive": Bayesian Beta CDF posterior-confidence stopping (speed-oriented; returns a +1 Monte Carlo
          estimate at a stopping time)
        - "simple": Futility + significance stopping (inflates Type I error)
        - None: No early stopping (fixed-B test)

    alpha : float
        Alpha level for significance testing.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping. Only used when early_stopping="adaptive".

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.

    """
    # Use default_rng for isolated RNG stream (avoids global state contamination)
    rng = np.random.default_rng(random_state)

    theta = np.abs(func(x, y, func_arg, random_state=random_state))
    y_ = y.copy()

    if early_stopping is None:
        theta_p = np.empty(n_resamples)
        for i in range(n_resamples):
            rng.shuffle(y_)
            theta_p[i] = func(x, y_, func_arg, random_state=random_state)
        p_value = (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)
        return float(p_value), n_resamples

    min_resamples = ceil(1 / alpha)
    n_resamples = max(n_resamples, min_resamples)
    extreme_count = 0

    if early_stopping == EarlyStopping.ADAPTIVE:
        m = 0
        while m < n_resamples:
            batch_end = min(m + _ADAPTIVE_BATCH_SIZE, n_resamples)
            for _ in range(batch_end - m):
                rng.shuffle(y_)
                theta_p = np.abs(func(x, y_, func_arg, random_state=random_state))
                if theta_p >= theta:
                    extreme_count += 1
            m = batch_end

            if m >= min_resamples:
                a = 1.0 + extreme_count
                b = 1.0 + m - extreme_count
                prob_sig = _beta_cdf(alpha, a, b)

                if prob_sig >= confidence:
                    return (extreme_count + 1) / (m + 1), m
                if (1.0 - prob_sig) >= confidence:
                    return (extreme_count + 1) / (m + 1), m

        return (extreme_count + 1) / (n_resamples + 1), n_resamples

    else:  # simple
        for i in range(n_resamples):
            rng.shuffle(y_)
            theta_p = np.abs(func(x, y_, func_arg, random_state=random_state))
            if theta_p >= theta:
                extreme_count += 1

            n = i + 1
            current_pval = (extreme_count + 1) / (n + 1)

            if n >= min_resamples:
                if current_pval < alpha:
                    return current_pval, n

                best_possible = (extreme_count + 1) / (n_resamples + 1)
                if best_possible >= alpha and extreme_count >= 3:
                    return current_pval, n

        return (extreme_count + 1) / (n_resamples + 1), n_resamples


def _ptest_multi_result(
    *,
    funcs: list,
    func_args: list,
    take_abs: list[bool] | None = None,
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> PermutationTestResult:
    """Max-T permutation test for multiple selectors.

    Computes max(selector_scores) INSIDE each permutation to provide fixed-node
    Type I error control in fixed-B mode when using multiple selectors.

    This implements the max-T method from Westfall & Young (1993), which
    accounts for the multiplicity of testing multiple selectors by using
    the maximum statistic under each permutation.

    Parameters
    ----------
    funcs : list
        List of selector functions.

    func_args : list
        Corresponding arguments for each selector function.

    x : np.ndarray
        Input data, usually the feature in the (x, y) pair.

    y : np.ndarray
        Input data, usually the target in the (x, y) pair.

    n_resamples : int
        Number of resamples.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Alpha level for significance testing.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    take_abs : list[bool] or None
        Whether to take the absolute value of each selector score before computing the max. If None, absolute values
        are taken for all selectors (backwards-compatible behavior).

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.

    """
    # Use default_rng for isolated RNG stream (avoids global state contamination)
    rng = np.random.default_rng(random_state)

    if take_abs is None:
        take_abs = [True] * len(funcs)
    if len(take_abs) != len(funcs):
        raise ValueError(
            "take_abs must have the same length as funcs (one flag per selector function)."
        )

    def compute_max_stat(x: np.ndarray, y: np.ndarray) -> float:
        """Compute max statistic across all selectors."""
        max_score = -np.inf
        for func, arg, abs_flag in zip(funcs, func_args, take_abs, strict=False):
            score = func(x, y, arg, random_state=random_state)
            if abs_flag:
                score = abs(score)
            if score > max_score:
                max_score = score
        return max_score

    theta = compute_max_stat(x, y)
    y_ = y.copy()

    if early_stopping is None:
        theta_p = np.empty(n_resamples)
        for i in range(n_resamples):
            rng.shuffle(y_)
            theta_p[i] = compute_max_stat(x, y_)
        p_value = (1 + np.sum(theta_p >= theta)) / (1 + n_resamples)
        return float(p_value), n_resamples

    min_resamples = ceil(1 / alpha)
    n_resamples = max(n_resamples, min_resamples)
    extreme_count = 0

    if early_stopping == EarlyStopping.ADAPTIVE:
        m = 0
        while m < n_resamples:
            batch_end = min(m + _ADAPTIVE_BATCH_SIZE, n_resamples)
            for _ in range(batch_end - m):
                rng.shuffle(y_)
                theta_p = compute_max_stat(x, y_)  # type: ignore[assignment]
                if theta_p >= theta:
                    extreme_count += 1
            m = batch_end

            if m >= min_resamples:
                a = 1.0 + extreme_count
                b = 1.0 + m - extreme_count
                prob_sig = _beta_cdf(alpha, a, b)

                if prob_sig >= confidence:
                    return (extreme_count + 1) / (m + 1), m
                if (1.0 - prob_sig) >= confidence:
                    return (extreme_count + 1) / (m + 1), m

        return (extreme_count + 1) / (n_resamples + 1), n_resamples

    else:  # simple
        for i in range(n_resamples):
            rng.shuffle(y_)
            theta_p = compute_max_stat(x, y_)  # type: ignore[assignment]
            if theta_p >= theta:
                extreme_count += 1

            n = i + 1
            current_pval = (extreme_count + 1) / (n + 1)

            if n >= min_resamples:
                if current_pval < alpha:
                    return current_pval, n

                best_possible = (extreme_count + 1) / (n_resamples + 1)
                if best_possible >= alpha and extreme_count >= 3:
                    return current_pval, n

        return (extreme_count + 1) / (n_resamples + 1), n_resamples


def _ptest_multi(
    *,
    funcs: list,
    func_args: list,
    take_abs: list[bool] | None = None,
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Return a max-T p-value and record its realized permutation count."""
    result = _ptest_multi_result(
        funcs=funcs,
        func_args=func_args,
        take_abs=take_abs,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
        confidence=confidence,
    )
    return record_permutation_test("selector", result)


# Parallel permutation test for multiple correlation (classifier)
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mc_parallel_result(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    random_state: int,
) -> PermutationTestResult:
    """Parallel permutation test for multiple correlation.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    # Compute observed statistic
    mu = x.mean()
    sst = np.sum((x - mu) ** 2)
    if sst == 0:
        return 1.0, 0
    ssb = 0.0
    for j in range(n_classes):
        x_j = x[y == j]
        n_j = len(x_j)
        if n_j > 0:
            mu_j = x_j.mean()
            ssb += n_j * (mu_j - mu) ** 2
    theta = np.sqrt(ssb / sst)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        ssb_perm = 0.0
        for j in range(n_classes):
            x_j = x[y_perm == j]
            n_j = len(x_j)
            if n_j > 0:
                mu_j = x_j.mean()
                ssb_perm += n_j * (mu_j - mu) ** 2
        theta_p[i] = np.sqrt(ssb_perm / sst)

    # +1 correction (Phipson & Smyth 2010)
    p_value = (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)
    return p_value, n_resamples


# Parallel permutation test for pearson correlation (regressor)
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_pc_parallel_result(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    random_state: int,
) -> PermutationTestResult:
    """Parallel permutation test for pearson correlation.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    n = len(x)
    sx = x.sum()
    sx2 = np.sum(x * x)
    sy = y.sum()
    sy2 = np.sum(y * y)
    sxy = np.sum(x * y)

    cov = n * sxy - sx * sy
    ssx = n * sx2 - sx * sx
    ssy = n * sy2 - sy * sy
    denom = np.sqrt(ssx * ssy)
    if denom == 0:
        return 1.0, 0
    theta = np.abs(cov / denom)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        sy_perm = y_perm.sum()
        sy2_perm = np.sum(y_perm * y_perm)
        sxy_perm = np.sum(x * y_perm)

        cov_perm = n * sxy_perm - sx * sy_perm
        ssy_perm = n * sy2_perm - sy_perm * sy_perm
        denom_perm = np.sqrt(ssx * ssy_perm)
        if denom_perm == 0:
            theta_p[i] = 0.0
        else:
            theta_p[i] = np.abs(cov_perm / denom_perm)

    # +1 correction (Phipson & Smyth 2010)
    p_value = (1 + np.sum(theta_p >= theta)) / (1 + n_resamples)
    return p_value, n_resamples


# Batched parallel permutation test for multiple correlation (classifier) with adaptive stopping.
# Runs fixed-size batches in parallel via prange, then checks Beta CDF stopping criterion.
# Calibration suggests similar null rejection for K=32; adaptive outputs are
# not theorem-level fixed-B p-values.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mc_parallel_batched_result(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
) -> PermutationTestResult:
    """Parallel batched permutation test for multiple correlation with adaptive stopping.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    # Compute observed statistic
    mu = x.mean()
    sst = np.sum((x - mu) ** 2)
    if sst == 0:
        return 1.0, 0
    ssb = 0.0
    for j in range(n_classes):
        x_j = x[y == j]
        n_j = len(x_j)
        if n_j > 0:
            mu_j = x_j.mean()
            ssb += n_j * (mu_j - mu) ** 2
    theta = np.sqrt(ssb / sst)

    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0

    while m < n_resamples:
        batch_size = min(_ADAPTIVE_BATCH_SIZE, n_resamples - m)
        # Run batch in parallel
        batch_extreme = np.zeros(batch_size, dtype=np.int64)
        for i in prange(batch_size):
            np.random.seed(random_state + m + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)

            ssb_perm = 0.0
            for j in range(n_classes):
                x_j = x[y_perm == j]
                n_j = len(x_j)
                if n_j > 0:
                    mu_j = x_j.mean()
                    ssb_perm += n_j * (mu_j - mu) ** 2
            theta_p = np.sqrt(ssb_perm / sst)
            if np.abs(theta_p) >= theta:
                batch_extreme[i] = 1

        extreme_count += int(np.sum(batch_extreme))
        m += batch_size

        # Check stopping criterion at batch boundary
        if m >= min_resamples:
            a = 1.0 + extreme_count
            b = 1.0 + m - extreme_count
            prob_sig = _beta_cdf(alpha, a, b)

            if prob_sig >= confidence:
                return (extreme_count + 1) / (m + 1), m
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1), m

    # +1 correction (Phipson & Smyth 2010)
    return (extreme_count + 1) / (n_resamples + 1), n_resamples


# Batched parallel permutation test for pearson correlation (regressor) with adaptive stopping.
# Same pattern as _ptest_mc_parallel_batched_result.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_pc_parallel_batched_result(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
) -> PermutationTestResult:
    """Parallel batched permutation test for pearson correlation with adaptive stopping.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    n = len(x)
    sx = x.sum()
    sx2 = np.sum(x * x)
    sy = y.sum()
    sy2 = np.sum(y * y)
    sxy = np.sum(x * y)

    cov = n * sxy - sx * sy
    ssx = n * sx2 - sx * sx
    ssy = n * sy2 - sy * sy
    denom = np.sqrt(ssx * ssy)
    if denom == 0:
        return 1.0, 0
    theta = np.abs(cov / denom)

    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0

    while m < n_resamples:
        batch_size = min(_ADAPTIVE_BATCH_SIZE, n_resamples - m)
        # Run batch in parallel
        batch_extreme = np.zeros(batch_size, dtype=np.int64)
        for i in prange(batch_size):
            np.random.seed(random_state + m + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)

            sy_perm = y_perm.sum()
            sy2_perm = np.sum(y_perm * y_perm)
            sxy_perm = np.sum(x * y_perm)

            cov_perm = n * sxy_perm - sx * sy_perm
            ssy_perm = n * sy2_perm - sy_perm * sy_perm
            denom_perm = np.sqrt(ssx * ssy_perm)
            theta_p = 0.0 if denom_perm == 0 else np.abs(cov_perm / denom_perm)
            if theta_p >= theta:
                batch_extreme[i] = 1

        extreme_count += int(np.sum(batch_extreme))
        m += batch_size

        # Check stopping criterion at batch boundary
        if m >= min_resamples:
            a = 1.0 + extreme_count
            b = 1.0 + m - extreme_count
            prob_sig = _beta_cdf(alpha, a, b)

            if prob_sig >= confidence:
                return (extreme_count + 1) / (m + 1), m
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1), m

    # +1 correction (Phipson & Smyth 2010)
    return (extreme_count + 1) / (n_resamples + 1), n_resamples


# Parallel permutation tests for RDC.
#
# All four kernels share one layout. The projected X features are computed and
# standardized once, because x never changes across permutations. The prange
# runs over the ``n_threads`` thread slots; slot ``t`` owns one permuted-label
# buffer and one projected-Y buffer, allocated once per test, and processes
# permutations ``t, t + n_threads, ...``. Permutation ``i`` is always seeded with
# ``random_state + i``, so the result is independent of the thread count. The
# earlier versions allocated fresh matrices (and copied X) inside every
# permutation, which on 26-class data meant more than a hundred multi-megabyte
# allocations per permutation across all threads. Multi-class targets are
# handled as the maximum RDC over one-vs-all indicators, whose ECDF needs no
# sort. The thread count is an argument (``numba.get_num_threads()`` at the call
# site) because reading it inside the kernel would disable Numba's on-disk cache.
#
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_rdc_regressor_parallel_result(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    s: float,
    rdc_seed: int,
    n_resamples: int,
    random_state: int,
    n_threads: int,
) -> PermutationTestResult:
    """Full parallel permutation test for RDC (regression), no early stopping.

    ``n_threads`` is the size of the Numba thread pool; each thread owns one set
    of scratch buffers and processes every ``n_threads``-th permutation, so the
    result does not depend on the thread count.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    n = len(x)
    X_feat = _rdc_features(x, k, s, rdc_seed)
    _rdc_standardize_columns(X_feat)
    wy0, wy1 = _rdc_projection_weights(k, s, rdc_seed + 1000)

    Y_buf = np.empty((n_threads, n, 2 * k), dtype=np.float64)
    y_buf = np.empty((n_threads, n), dtype=np.float64)

    # Observed statistic
    _rdc_fill_ecdf_features(_rdc_ecdf(y), k, wy0, wy1, Y_buf[0])
    _rdc_standardize_columns(Y_buf[0])
    theta = _rdc_max_abs_corr(X_feat, Y_buf[0])

    # Full parallel permutation
    theta_p = np.empty(n_resamples, dtype=np.float64)
    for t in prange(n_threads):
        y_perm = y_buf[t]
        Y_perm = Y_buf[t]
        for i in range(t, n_resamples, n_threads):
            np.random.seed(random_state + i)
            y_perm[:] = y
            np.random.shuffle(y_perm)

            _rdc_fill_ecdf_features(_rdc_ecdf(y_perm), k, wy0, wy1, Y_perm)
            _rdc_standardize_columns(Y_perm)
            theta_p[i] = _rdc_max_abs_corr(X_feat, Y_perm)

    # +1 correction (Phipson & Smyth 2010)
    p_value = (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)
    return p_value, n_resamples


# Batched parallel permutation test for RDC (regressor) with adaptive stopping.
# Same pattern as _ptest_pc_parallel_batched_result.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_rdc_regressor_parallel_batched_result(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    s: float,
    rdc_seed: int,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
    n_threads: int,
) -> PermutationTestResult:
    """Parallel batched permutation test for RDC (regression) with adaptive stopping.

    ``n_threads`` is the size of the Numba thread pool; each thread owns one set
    of scratch buffers and processes every ``n_threads``-th permutation, so the
    result does not depend on the thread count.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    n = len(x)
    X_feat = _rdc_features(x, k, s, rdc_seed)
    _rdc_standardize_columns(X_feat)
    wy0, wy1 = _rdc_projection_weights(k, s, rdc_seed + 1000)

    Y_buf = np.empty((n_threads, n, 2 * k), dtype=np.float64)
    y_buf = np.empty((n_threads, n), dtype=np.float64)

    # Observed statistic
    _rdc_fill_ecdf_features(_rdc_ecdf(y), k, wy0, wy1, Y_buf[0])
    _rdc_standardize_columns(Y_buf[0])
    theta = _rdc_max_abs_corr(X_feat, Y_buf[0])
    if theta <= 0.0:
        return 1.0, 0

    # Parallel batched permutation
    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0

    while m < n_resamples:
        batch_size = min(_ADAPTIVE_BATCH_SIZE, n_resamples - m)
        batch_extreme = np.zeros(batch_size, dtype=np.int64)

        for t in prange(n_threads):
            y_perm = y_buf[t]
            Y_perm = Y_buf[t]
            for i in range(t, batch_size, n_threads):
                np.random.seed(random_state + m + i)
                y_perm[:] = y
                np.random.shuffle(y_perm)

                _rdc_fill_ecdf_features(_rdc_ecdf(y_perm), k, wy0, wy1, Y_perm)
                _rdc_standardize_columns(Y_perm)
                if _rdc_max_abs_corr(X_feat, Y_perm) >= theta:
                    batch_extreme[i] = 1

        extreme_count += int(np.sum(batch_extreme))
        m += batch_size

        # Check adaptive stopping criterion
        if m >= min_resamples:
            a = 1.0 + extreme_count
            b = 1.0 + m - extreme_count
            prob_sig = _beta_cdf(alpha, a, b)
            if prob_sig >= confidence:
                return (extreme_count + 1) / (m + 1), m
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1), m

    # +1 correction (Phipson & Smyth 2010)
    return (extreme_count + 1) / (n_resamples + 1), n_resamples


# Parallel permutation test for RDC (classifier), no early stopping.
# Handles multi-class via max RDC over one-vs-all binary encodings; the binary
# case tests the class-1 indicator with a single weight set.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_rdc_classifier_parallel_result(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    k: int,
    s: float,
    rdc_seed: int,
    n_resamples: int,
    random_state: int,
    n_threads: int,
) -> PermutationTestResult:
    """Full parallel permutation test for RDC (classification), no early stopping.

    ``n_threads`` is the size of the Numba thread pool; each thread owns one set
    of scratch buffers and processes every ``n_threads``-th permutation, so the
    result does not depend on the thread count.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    n = len(x)
    X_feat = _rdc_features(x, k, s, rdc_seed)
    _rdc_standardize_columns(X_feat)

    n_indicators = 1 if n_classes == 2 else n_classes
    first_class = 1 if n_classes == 2 else 0
    wy0_all = np.empty((n_indicators, k), dtype=np.float64)
    wy1_all = np.empty((n_indicators, k), dtype=np.float64)
    for c in range(n_indicators):
        w0, w1 = _rdc_projection_weights(k, s, rdc_seed + 1000 + c)
        wy0_all[c] = w0
        wy1_all[c] = w1

    Y_buf = np.empty((n_threads, n, 2 * k), dtype=np.float64)
    y_buf = np.empty((n_threads, n), dtype=y.dtype)

    # Observed statistic
    theta = 0.0
    for c in range(n_indicators):
        _rdc_fill_indicator_features(y, first_class + c, k, wy0_all[c], wy1_all[c], Y_buf[0])
        _rdc_standardize_columns(Y_buf[0])
        rdc_c = _rdc_max_abs_corr(X_feat, Y_buf[0])
        if rdc_c > theta:
            theta = rdc_c

    # Full parallel permutation
    theta_p = np.empty(n_resamples, dtype=np.float64)
    for t in prange(n_threads):
        y_perm = y_buf[t]
        Y_perm = Y_buf[t]
        for i in range(t, n_resamples, n_threads):
            np.random.seed(random_state + i)
            y_perm[:] = y
            np.random.shuffle(y_perm)

            rdc_perm = 0.0
            for c in range(n_indicators):
                _rdc_fill_indicator_features(
                    y_perm, first_class + c, k, wy0_all[c], wy1_all[c], Y_perm
                )
                _rdc_standardize_columns(Y_perm)
                rdc_c = _rdc_max_abs_corr(X_feat, Y_perm)
                if rdc_c > rdc_perm:
                    rdc_perm = rdc_c
            theta_p[i] = rdc_perm

    # +1 correction (Phipson & Smyth 2010)
    p_value = (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)
    return p_value, n_resamples


# Batched parallel permutation test for RDC (classifier) with adaptive stopping.
# Handles multi-class via max RDC over one-vs-all binary encodings.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_rdc_classifier_parallel_batched_result(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    k: int,
    s: float,
    rdc_seed: int,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
    n_threads: int,
) -> PermutationTestResult:
    """Parallel batched permutation test for RDC (classification) with adaptive stopping.

    ``n_threads`` is the size of the Numba thread pool; each thread owns one set
    of scratch buffers and processes every ``n_threads``-th permutation, so the
    result does not depend on the thread count.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    n = len(x)
    X_feat = _rdc_features(x, k, s, rdc_seed)
    _rdc_standardize_columns(X_feat)

    n_indicators = 1 if n_classes == 2 else n_classes
    first_class = 1 if n_classes == 2 else 0
    wy0_all = np.empty((n_indicators, k), dtype=np.float64)
    wy1_all = np.empty((n_indicators, k), dtype=np.float64)
    for c in range(n_indicators):
        w0, w1 = _rdc_projection_weights(k, s, rdc_seed + 1000 + c)
        wy0_all[c] = w0
        wy1_all[c] = w1

    Y_buf = np.empty((n_threads, n, 2 * k), dtype=np.float64)
    y_buf = np.empty((n_threads, n), dtype=y.dtype)

    # Observed statistic
    theta = 0.0
    for c in range(n_indicators):
        _rdc_fill_indicator_features(y, first_class + c, k, wy0_all[c], wy1_all[c], Y_buf[0])
        _rdc_standardize_columns(Y_buf[0])
        rdc_c = _rdc_max_abs_corr(X_feat, Y_buf[0])
        if rdc_c > theta:
            theta = rdc_c
    if theta <= 0.0:
        return 1.0, 0

    # Parallel batched permutation
    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0

    while m < n_resamples:
        batch_size = min(_ADAPTIVE_BATCH_SIZE, n_resamples - m)
        batch_extreme = np.zeros(batch_size, dtype=np.int64)

        for t in prange(n_threads):
            y_perm = y_buf[t]
            Y_perm = Y_buf[t]
            for i in range(t, batch_size, n_threads):
                np.random.seed(random_state + m + i)
                y_perm[:] = y
                np.random.shuffle(y_perm)

                rdc_perm = 0.0
                for c in range(n_indicators):
                    _rdc_fill_indicator_features(
                        y_perm, first_class + c, k, wy0_all[c], wy1_all[c], Y_perm
                    )
                    _rdc_standardize_columns(Y_perm)
                    rdc_c = _rdc_max_abs_corr(X_feat, Y_perm)
                    if rdc_c > rdc_perm:
                        rdc_perm = rdc_c
                if rdc_perm >= theta:
                    batch_extreme[i] = 1

        extreme_count += int(np.sum(batch_extreme))
        m += batch_size

        # Check adaptive stopping criterion
        if m >= min_resamples:
            a = 1.0 + extreme_count
            b = 1.0 + m - extreme_count
            prob_sig = _beta_cdf(alpha, a, b)
            if prob_sig >= confidence:
                return (extreme_count + 1) / (m + 1), m
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1), m

    # +1 correction (Phipson & Smyth 2010)
    return (extreme_count + 1) / (n_resamples + 1), n_resamples


@ClassifierSelectors.register("mc")
@njit(cache=True, nogil=True, fastmath=True)
def mc(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int | None = None) -> float:
    """Calculate the multiple correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature.

    y : np.ndarray
        Target.

    n_classes : int
        Number of classes.

    random_state : int, default=None
        Random seed. Kept for API compatibility.

    Returns
    -------
    float
        Estimated multiple correlation.

    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    # Sum of squares total (SST)
    mu = 0.0
    n = len(x)

    for value in x:
        mu += value
    mu /= n

    sst = 0.0
    dev = x - mu
    dev *= dev
    for value in dev:
        sst += value

    # Sum of squares between (SSB)
    ssb = 0.0
    for j in range(n_classes):
        x_j = x[y == j]
        n_j = len(x_j)

        if not n_j:
            continue

        mu_j = 0.0
        for value in x_j:
            mu_j += value
        mu_j /= n_j

        dev_j = mu_j - mu
        dev_j *= dev_j
        ssb += n_j * dev_j

    if sst <= 0.0 or not np.isfinite(sst):
        return 0.0
    ratio = ssb / sst
    if ratio < 0.0 or not np.isfinite(ratio):
        return 0.0
    return np.sqrt(ratio)


@ClassifierSelectors.register("mi")
def mi(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int) -> float:
    """Calculate the mutual information.

    Parameters
    ----------
    x : np.ndarray
        Feature.

    y : np.ndarray
        Target.

    n_classes : int
        Number of classes. Kept for API compatibility.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated mutual information.

    """
    if x.ndim == 1:
        x = x[:, None]

    return mutual_info_classif(x, y, random_state=random_state)[0]


@RegressorSelectors.register("pc")
@njit(cache=True, nogil=True, fastmath=True)
def pc(x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int | None = None) -> float:
    """Calculate the Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.

    random_state : int, default=None
        Random seed. Kept for API compatibility.

    Returns
    -------
    float
        Estimated Pearson correlation.

    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    return _correlation(x, y) if standardize else _covariance(x, y)


@njit(cache=True, nogil=True, fastmath=True)
def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the covariance.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    Returns
    -------
    float
        Estimated covariance.

    """
    n = len(x)
    sx = 0.0
    sy = 0.0
    sxy = 0.0

    for i in range(n):
        xi = x[i]
        yi = y[i]
        sx += xi
        sy += yi
        sxy += xi * yi

    return (sxy - (sx * sy / n)) / n


@njit(cache=True, nogil=True, fastmath=True)
def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    Returns
    -------
    float
        Estimated Pearson correlation.

    """
    n = len(x)
    sx = 0.0
    sy = 0.0
    sx2 = 0.0
    sy2 = 0.0
    sxy = 0.0

    for i in range(n):
        xi = x[i]
        yi = y[i]
        sx += xi
        sx2 += xi * xi
        sy += yi
        sy2 += yi * yi
        sxy += xi * yi

    cov = n * sxy - sx * sy
    ssx = n * sx2 - sx * sx
    ssy = n * sy2 - sy * sy

    denom = ssx * ssy
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0
    return cov / np.sqrt(denom)


@RegressorSelectors.register("dc")
def dc(x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int | None = None) -> float:
    """Calculate the distance correlation.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.

    random_state : int, default=None
        Random seed. Kept for API compatibility.

    Returns
    -------
    float
        Estimated distance correlation.

    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    # Imported lazily: dcor costs ~3 s at import and is only needed for the
    # distance-correlation selector, which would otherwise be paid by every
    # forest worker process at startup.
    from dcor import distance_correlation as _d_correlation
    from dcor import distance_covariance as _d_covariance

    return float(_d_correlation(x, y)) if standardize else float(_d_covariance(x, y))


# =============================================================================
# Randomized Dependence Coefficient (RDC)
# Lopez-Paz et al. (2013) - https://arxiv.org/abs/1304.7717
# O(n log n) non-linear dependence measure (vs O(n²) for distance correlation)
#
# R reference implementation from paper:
#   rdc <- function(x,y,k,s) {
#     x <- cbind(apply(as.matrix(x),2,function(u) ecdf(u)(u)),1)
#     y <- cbind(apply(as.matrix(y),2,function(u) ecdf(u)(u)),1)
#     wx <- matrix(rnorm(ncol(x)*k,0,s),ncol(x),k)
#     wy <- matrix(rnorm(ncol(y)*k,0,s),ncol(y),k)
#     cancor(cbind(cos(x%*%wx),sin(x%*%wx)), cbind(cos(y%*%wy),sin(y%*%wy)))$cor[1]
#   }
# =============================================================================

_RDC_K = 10
_RDC_S = 1.0 / 6.0  # Bandwidth parameter


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_ecdf(x: np.ndarray) -> np.ndarray:
    """Empirical CDF transform: ecdf(x)(x) = #(X <= x) / n."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        value = j / n
        for k in range(i, j):
            ranks[order[k]] = value
        i = j
    return ranks


# Note: Uses np.random.seed() because Numba doesn't support default_rng() inside @njit.
@njit(cache=True, nogil=True, fastmath=True)
def _rdc_features(x: np.ndarray, k: int, s: float, seed: int) -> np.ndarray:
    """Create RDC features: [cos(X @ w), sin(X @ w)] where X = [ecdf(x), 1]."""
    n = len(x)
    np.random.seed(seed)

    # X = [ecdf(x), 1] has shape (n, 2)
    ecdf_x = _rdc_ecdf(x)

    # w has shape (2, k): random weights for [ecdf, bias]
    w0 = np.empty(k, dtype=np.float64)  # weights for ecdf
    w1 = np.empty(k, dtype=np.float64)  # weights for bias
    for j in range(k):
        w0[j] = np.random.randn() * s
        w1[j] = np.random.randn() * s

    # Compute X @ w = ecdf * w0 + 1 * w1, then [cos, sin]
    features = np.empty((n, 2 * k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            proj = ecdf_x[i] * w0[j] + w1[j]
            features[i, j] = np.cos(proj)
            features[i, k + j] = np.sin(proj)

    return features


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_standardize_columns(A: np.ndarray) -> None:
    """Center every column of ``A`` in place and scale it to unit sum of squares.

    Columns with numerically zero variance are centered but left unscaled, so
    they contribute nothing to any correlation.
    """
    n, p = A.shape
    for j in range(p):
        mu = 0.0
        for i in range(n):
            mu += A[i, j]
        mu /= n
        ss = 0.0
        for i in range(n):
            A[i, j] -= mu
            ss += A[i, j] * A[i, j]
        if ss > 1e-10:
            inv_std = 1.0 / np.sqrt(ss)
            for i in range(n):
                A[i, j] *= inv_std


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_max_abs_corr(X: np.ndarray, Y: np.ndarray) -> float:
    """Maximum absolute column-pair correlation of two standardized matrices."""
    n, p = X.shape
    q = Y.shape[1]
    max_corr = 0.0
    for j in range(p):
        for k in range(q):
            corr = 0.0
            for i in range(n):
                corr += X[i, j] * Y[i, k]
            if corr < 0:
                corr = -corr
            if corr > max_corr:
                max_corr = corr
    return min(max_corr, 1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_cancor(X: np.ndarray, Y: np.ndarray) -> float:
    """RDC-style max absolute pairwise correlation between feature matrices.

    This approximates the original RDC canonical-correlation step with the
    maximum absolute correlation between standardized projected columns. Both
    inputs are standardized in place.
    """
    _rdc_standardize_columns(X)
    _rdc_standardize_columns(Y)
    return _rdc_max_abs_corr(X, Y)


# Note: Uses np.random.seed() because Numba doesn't support default_rng() inside @njit.
@njit(cache=True, nogil=True, fastmath=True)
def _rdc_projection_weights(k: int, s: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random projection weights for the ECDF and bias inputs, drawn as in ``_rdc_features``."""
    np.random.seed(seed)
    w0 = np.empty(k, dtype=np.float64)
    w1 = np.empty(k, dtype=np.float64)
    for j in range(k):
        w0[j] = np.random.randn() * s
        w1[j] = np.random.randn() * s
    return w0, w1


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_fill_ecdf_features(
    ecdf: np.ndarray, k: int, w0: np.ndarray, w1: np.ndarray, out: np.ndarray
) -> None:
    """Write ``[cos(ecdf * w0 + w1), sin(ecdf * w0 + w1)]`` into ``out`` of shape (n, 2k)."""
    n = len(ecdf)
    for i in range(n):
        for j in range(k):
            proj = ecdf[i] * w0[j] + w1[j]
            out[i, j] = np.cos(proj)
            out[i, k + j] = np.sin(proj)


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_fill_indicator_features(
    y: np.ndarray, c: int, k: int, w0: np.ndarray, w1: np.ndarray, out: np.ndarray
) -> None:
    """Write the RDC features of the one-vs-all indicator of class ``c`` into ``out``.

    The empirical CDF of a binary indicator takes two values: members sit at 1
    and non-members at the share of non-members, so no sort is needed. The
    result equals ``_rdc_features`` applied to the indicator with these weights
    up to one floating-point ulp in the non-member ECDF value, whose fast-math
    division the compiler may emit as a reciprocal multiply.
    """
    n = len(y)
    n_members = 0
    for i in range(n):
        if y[i] == c:
            n_members += 1
    e_other = (n - n_members) / n
    for i in range(n):
        e = 1.0 if y[i] == c else e_other
        for j in range(k):
            proj = e * w0[j] + w1[j]
            out[i, j] = np.cos(proj)
            out[i, k + j] = np.sin(proj)


@njit(cache=True, nogil=True, fastmath=True)
def _rdc(x: np.ndarray, y: np.ndarray, k: int, s: float, seed: int) -> float:
    """Randomized Dependence Coefficient."""
    n = len(x)
    if n < 3:
        return 0.0

    # Check constant
    x_min, x_max = x[0], x[0]
    y_min, y_max = y[0], y[0]
    for i in range(1, n):
        if x[i] < x_min:
            x_min = x[i]
        if x[i] > x_max:
            x_max = x[i]
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]
    if x_max == x_min or y_max == y_min:
        return 0.0

    # Create features
    X_feat = _rdc_features(x, k, s, seed)
    Y_feat = _rdc_features(y, k, s, seed + 1000)

    return _rdc_cancor(X_feat, Y_feat)


@ClassifierSelectors.register("rdc")
def rdc_classifier(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    random_state: int | None = None,
    n_projections: int = _RDC_K,
) -> float:
    """RDC for classification.

    O(n log n) non-linear dependence.
    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    seed = 42 if random_state is None else random_state

    if n_classes == 2:
        return _rdc(x, y.astype(np.float64), n_projections, _RDC_S, seed)

    # Multi-class: max RDC over one-vs-all
    max_rdc = 0.0
    for c in range(n_classes):
        rdc_c = _rdc(x, (y == c).astype(np.float64), n_projections, _RDC_S, seed + c)
        if rdc_c > max_rdc:
            max_rdc = rdc_c
    return max_rdc


@RegressorSelectors.register("rdc")
def rdc_regressor(
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    random_state: int | None = None,
    n_projections: int = _RDC_K,
) -> float:
    """RDC for regression.

    O(n log n) non-linear dependence.
    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    seed = 42 if random_state is None else random_state
    return _rdc(x, y, n_projections, _RDC_S, seed)


@ClassifierSelectorTests.register("mc")
def ptest_mc(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Perform a permutation test using the multiple correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    n_classes : int
        Number of classes.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method. "adaptive" (default) uses Bayesian stopping,
        "simple" uses futility stopping, None disables early stopping.

    alpha : float
        Significance threshold.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Estimated achieved significance level.

    """
    if n_resamples >= _PARALLEL_THRESHOLD:
        if early_stopping is None:
            result = _ptest_mc_parallel_result(
                x=x,
                y=y,
                n_classes=n_classes,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        elif early_stopping == EarlyStopping.ADAPTIVE:
            result = _ptest_mc_parallel_batched_result(
                x=x,
                y=y,
                n_classes=n_classes,
                n_resamples=n_resamples,
                random_state=random_state,
                alpha=alpha,
                confidence=confidence,
            )
        else:
            result = _ptest_result(
                func=mc,
                func_arg=n_classes,
                x=x,
                y=y,
                n_resamples=n_resamples,
                early_stopping=early_stopping,
                alpha=alpha,
                random_state=random_state,
                confidence=confidence,
            )
    else:
        result = _ptest_result(
            func=mc,
            func_arg=n_classes,
            x=x,
            y=y,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
            confidence=confidence,
        )
    return record_permutation_test("selector", result)


@ClassifierSelectorTests.register("mi")
def ptest_mi(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Perform a permutation test using the mutual information.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    n_classes : int
        Number of classes.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance threshold.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Estimated achieved significance level.

    """
    result = _ptest_result(
        func=mi,
        func_arg=n_classes,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
        confidence=confidence,
    )
    return record_permutation_test("selector", result)


@RegressorSelectorTests.register("pc")
def ptest_pc(
    *,
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Perform a permutation test using the Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance threshold.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Estimated achieved significance level.

    """
    if n_resamples >= _PARALLEL_THRESHOLD:
        if early_stopping is None:
            result = _ptest_pc_parallel_result(
                x=x,
                y=y,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        elif early_stopping == EarlyStopping.ADAPTIVE:
            result = _ptest_pc_parallel_batched_result(
                x=x,
                y=y,
                n_resamples=n_resamples,
                random_state=random_state,
                alpha=alpha,
                confidence=confidence,
            )
        else:
            result = _ptest_result(
                func=pc,
                func_arg=standardize,
                x=x,
                y=y,
                n_resamples=n_resamples,
                early_stopping=early_stopping,
                alpha=alpha,
                random_state=random_state,
                confidence=confidence,
            )
    else:
        result = _ptest_result(
            func=pc,
            func_arg=standardize,
            x=x,
            y=y,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
            confidence=confidence,
        )
    return record_permutation_test("selector", result)


@RegressorSelectorTests.register("dc")
def ptest_dc(
    *,
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Perform a permutation test using the distance correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance threshold.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Estimated achieved significance level.

    """
    result = _ptest_result(
        func=dc,
        func_arg=standardize,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
        confidence=confidence,
    )
    return record_permutation_test("selector", result)


# =============================================================================
# RDC Permutation Tests
# =============================================================================


@ClassifierSelectorTests.register("rdc")
def ptest_rdc_classifier(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
    n_projections: int = _RDC_K,
) -> float:
    """Perform a permutation test using the Randomized Dependence Coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values (class labels).

    n_classes : int
        Number of classes.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance threshold.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Estimated achieved significance level.

    """
    if n_resamples >= _PARALLEL_THRESHOLD:
        if early_stopping is None:
            result = _ptest_rdc_classifier_parallel_result(
                x=x,
                y=y,
                n_classes=n_classes,
                k=n_projections,
                s=_RDC_S,
                rdc_seed=random_state,
                n_resamples=n_resamples,
                random_state=random_state,
                n_threads=numba.get_num_threads(),
            )
        elif early_stopping == EarlyStopping.ADAPTIVE:
            result = _ptest_rdc_classifier_parallel_batched_result(
                x=x,
                y=y,
                n_classes=n_classes,
                k=n_projections,
                s=_RDC_S,
                rdc_seed=random_state,
                n_resamples=n_resamples,
                random_state=random_state,
                alpha=alpha,
                confidence=confidence,
                n_threads=numba.get_num_threads(),
            )
        else:
            result = _ptest_result(
                func=partial(rdc_classifier, n_projections=n_projections),
                func_arg=n_classes,
                x=x,
                y=y,
                n_resamples=n_resamples,
                early_stopping=early_stopping,
                alpha=alpha,
                random_state=random_state,
                confidence=confidence,
            )
    else:
        result = _ptest_result(
            func=partial(rdc_classifier, n_projections=n_projections),
            func_arg=n_classes,
            x=x,
            y=y,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
            confidence=confidence,
        )
    return record_permutation_test("selector", result)


@RegressorSelectorTests.register("rdc")
def ptest_rdc_regressor(
    *,
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
    n_projections: int = _RDC_K,
) -> float:
    """Perform a permutation test using the Randomized Dependence Coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance threshold.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Estimated achieved significance level.

    """
    if n_resamples >= _PARALLEL_THRESHOLD:
        if early_stopping is None:
            result = _ptest_rdc_regressor_parallel_result(
                x=x,
                y=y,
                k=n_projections,
                s=_RDC_S,
                rdc_seed=random_state,
                n_resamples=n_resamples,
                random_state=random_state,
                n_threads=numba.get_num_threads(),
            )
        elif early_stopping == EarlyStopping.ADAPTIVE:
            result = _ptest_rdc_regressor_parallel_batched_result(
                x=x,
                y=y,
                k=n_projections,
                s=_RDC_S,
                rdc_seed=random_state,
                n_resamples=n_resamples,
                random_state=random_state,
                alpha=alpha,
                confidence=confidence,
                n_threads=numba.get_num_threads(),
            )
        else:
            result = _ptest_result(
                func=partial(rdc_regressor, n_projections=n_projections),
                func_arg=standardize,
                x=x,
                y=y,
                n_resamples=n_resamples,
                early_stopping=early_stopping,
                alpha=alpha,
                random_state=random_state,
                confidence=confidence,
            )
    else:
        result = _ptest_result(
            func=partial(rdc_regressor, n_projections=n_projections),
            func_arg=standardize,
            x=x,
            y=y,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
            confidence=confidence,
        )
    return record_permutation_test("selector", result)
