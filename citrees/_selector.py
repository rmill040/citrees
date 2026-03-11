import os
from math import ceil
from typing import Any

import numpy as np
from dcor import distance_correlation as _d_correlation
from dcor import distance_covariance as _d_covariance
from numba import njit, prange
from sklearn.feature_selection import mutual_info_classif

from citrees._registry import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    RegressorSelectors,
    RegressorSelectorTests,
)
from citrees._sequential import _beta_cdf
from citrees._types import EarlyStopping, EarlyStoppingOption

# Threshold for using parallel permutation tests
_PARALLEL_THRESHOLD = 200

# P-value correction: Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero."
# SAGMB 9(1):39. https://pubmed.ncbi.nlm.nih.gov/21044043/
# Use p = (b+1)/(m+1) instead of p = b/m to avoid p=0.
# Note: min_resamples = ceil(1/alpha) remains valid since 1/(m+1) < alpha.


def _ptest(
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
) -> float:
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
    float
        Estimated achieved significance level.
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
        return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)

    min_resamples = ceil(1 / alpha)
    n_resamples = max(n_resamples, min_resamples)
    extreme_count = 0

    if early_stopping == EarlyStopping.ADAPTIVE:
        batch_size = os.cpu_count() or 1
        m = 0
        while m < n_resamples:
            batch_end = min(m + batch_size, n_resamples)
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
                    return (extreme_count + 1) / (m + 1)
                if (1.0 - prob_sig) >= confidence:
                    return (extreme_count + 1) / (m + 1)

        return (extreme_count + 1) / (n_resamples + 1)

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
                    return current_pval

                best_possible = (extreme_count + 1) / (n_resamples + 1)
                if best_possible >= alpha and extreme_count >= 3:
                    return current_pval

        return (extreme_count + 1) / (n_resamples + 1)


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
    """Max-T permutation test for multiple selectors.

    Computes max(selector_scores) INSIDE each permutation to provide
    valid Type I error control in fixed-B mode when using multiple selectors.

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
    float
        Estimated achieved significance level.
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
        return (1 + np.sum(theta_p >= theta)) / (1 + n_resamples)

    min_resamples = ceil(1 / alpha)
    n_resamples = max(n_resamples, min_resamples)
    extreme_count = 0

    if early_stopping == EarlyStopping.ADAPTIVE:
        batch_size = os.cpu_count() or 1
        m = 0
        while m < n_resamples:
            batch_end = min(m + batch_size, n_resamples)
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
                    return (extreme_count + 1) / (m + 1)
                if (1.0 - prob_sig) >= confidence:
                    return (extreme_count + 1) / (m + 1)

        return (extreme_count + 1) / (n_resamples + 1)

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
                    return current_pval

                best_possible = (extreme_count + 1) / (n_resamples + 1)
                if best_possible >= alpha and extreme_count >= 3:
                    return current_pval

        return (extreme_count + 1) / (n_resamples + 1)


# Parallel permutation test for multiple correlation (classifier)
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mc_parallel(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    random_state: int,
) -> float:
    """Parallel permutation test for multiple correlation."""
    # Compute observed statistic
    mu = x.mean()
    sst = np.sum((x - mu) ** 2)
    if sst == 0:
        return 1.0
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
    return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)


# Parallel permutation test for pearson correlation (regressor)
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_pc_parallel(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    random_state: int,
) -> float:
    """Parallel permutation test for pearson correlation."""
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
        return 1.0
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
    return (1 + np.sum(theta_p >= theta)) / (1 + n_resamples)


# Batched parallel permutation test for multiple correlation (classifier) with adaptive stopping.
# Runs K=32 permutations in parallel via prange, then checks Beta CDF stopping criterion.
# Validated in paper/scripts/theory/batched_stopping_analysis.py: K=32 preserves Type I error.
_BATCH_SIZE_PARALLEL = 32


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mc_parallel_batched(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
) -> float:
    """Parallel batched permutation test for multiple correlation with adaptive stopping."""
    # Compute observed statistic
    mu = x.mean()
    sst = np.sum((x - mu) ** 2)
    if sst == 0:
        return 1.0
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
        batch_size = min(_BATCH_SIZE_PARALLEL, n_resamples - m)
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
                return (extreme_count + 1) / (m + 1)
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1)

    # +1 correction (Phipson & Smyth 2010)
    return (extreme_count + 1) / (n_resamples + 1)


# Batched parallel permutation test for pearson correlation (regressor) with adaptive stopping.
# Same pattern as _ptest_mc_parallel_batched.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_pc_parallel_batched(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
) -> float:
    """Parallel batched permutation test for pearson correlation with adaptive stopping."""
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
        return 1.0
    theta = np.abs(cov / denom)

    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0

    while m < n_resamples:
        batch_size = min(_BATCH_SIZE_PARALLEL, n_resamples - m)
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
            if denom_perm == 0:
                theta_p = 0.0
            else:
                theta_p = np.abs(cov_perm / denom_perm)
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
                return (extreme_count + 1) / (m + 1)
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1)

    # +1 correction (Phipson & Smyth 2010)
    return (extreme_count + 1) / (n_resamples + 1)


# Parallel permutation test for RDC (regressor), no early stopping.
# Precomputes X features and Y projection weights once, then runs permutations in parallel.
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_rdc_regressor_parallel(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    s: float,
    rdc_seed: int,
    n_resamples: int,
    random_state: int,
) -> float:
    """Full parallel permutation test for RDC (regression), no early stopping."""
    n = len(x)
    X_feat = _rdc_features(x, k, s, rdc_seed)

    # Precompute Y projection weights (deterministic from rdc_seed + 1000)
    np.random.seed(rdc_seed + 1000)
    wy0 = np.empty(k, dtype=np.float64)
    wy1 = np.empty(k, dtype=np.float64)
    for j in range(k):
        wy0[j] = np.random.randn() * s
        wy1[j] = np.random.randn() * s

    # Observed statistic
    Y_feat_obs = _rdc_features(y, k, s, rdc_seed + 1000)
    theta = _rdc_cancor(X_feat.copy(), Y_feat_obs)

    # Full parallel permutation
    theta_p = np.empty(n_resamples, dtype=np.float64)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        ecdf_y = _rdc_ecdf(y_perm)
        Y_feat_perm = np.empty((n, 2 * k), dtype=np.float64)
        for ii in range(n):
            for jj in range(k):
                proj = ecdf_y[ii] * wy0[jj] + wy1[jj]
                Y_feat_perm[ii, jj] = np.cos(proj)
                Y_feat_perm[ii, k + jj] = np.sin(proj)

        rdc_val = _rdc_cancor(X_feat.copy(), Y_feat_perm)
        theta_p[i] = rdc_val

    # +1 correction (Phipson & Smyth 2010)
    return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)


# Batched parallel permutation test for RDC (regressor) with adaptive stopping.
# Same pattern as _ptest_pc_parallel_batched.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_rdc_regressor_parallel_batched(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    s: float,
    rdc_seed: int,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
) -> float:
    """Parallel batched permutation test for RDC (regression) with adaptive stopping."""
    n = len(x)

    # Precompute X features (x never changes across permutations)
    X_feat = _rdc_features(x, k, s, rdc_seed)

    # Precompute Y projection weights (deterministic from rdc_seed + 1000)
    np.random.seed(rdc_seed + 1000)
    wy0 = np.empty(k, dtype=np.float64)
    wy1 = np.empty(k, dtype=np.float64)
    for j in range(k):
        wy0[j] = np.random.randn() * s
        wy1[j] = np.random.randn() * s

    # Observed statistic
    Y_feat_obs = _rdc_features(y, k, s, rdc_seed + 1000)
    theta = _rdc_cancor(X_feat.copy(), Y_feat_obs)
    if theta <= 0.0:
        return 1.0

    # Parallel batched permutation
    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0

    while m < n_resamples:
        batch_size = min(_BATCH_SIZE_PARALLEL, n_resamples - m)
        batch_extreme = np.zeros(batch_size, dtype=np.int64)

        for i in prange(batch_size):
            np.random.seed(random_state + m + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)

            ecdf_y = _rdc_ecdf(y_perm)
            Y_feat_perm = np.empty((n, 2 * k), dtype=np.float64)
            for ii in range(n):
                for jj in range(k):
                    proj = ecdf_y[ii] * wy0[jj] + wy1[jj]
                    Y_feat_perm[ii, jj] = np.cos(proj)
                    Y_feat_perm[ii, k + jj] = np.sin(proj)

            rdc_val = _rdc_cancor(X_feat.copy(), Y_feat_perm)
            if rdc_val >= theta:
                batch_extreme[i] = 1

        extreme_count += int(np.sum(batch_extreme))
        m += batch_size

        # Check adaptive stopping criterion
        if m >= min_resamples:
            a = 1.0 + extreme_count
            b = 1.0 + m - extreme_count
            prob_sig = _beta_cdf(alpha, a, b)
            if prob_sig >= confidence:
                return (extreme_count + 1) / (m + 1)
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1)

    # +1 correction (Phipson & Smyth 2010)
    return (extreme_count + 1) / (n_resamples + 1)


# Parallel permutation test for RDC (classifier), no early stopping.
# Handles multi-class via max RDC over one-vs-all binary encodings.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_rdc_classifier_parallel(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    k: int,
    s: float,
    rdc_seed: int,
    n_resamples: int,
    random_state: int,
) -> float:
    """Full parallel permutation test for RDC (classification), no early stopping."""
    n = len(x)
    X_feat = _rdc_features(x, k, s, rdc_seed)

    # Observed statistic
    if n_classes == 2:
        y_float = y.astype(np.float64)
        Y_feat_obs = _rdc_features(y_float, k, s, rdc_seed + 1000)
        theta = _rdc_cancor(X_feat.copy(), Y_feat_obs)
    else:
        theta = 0.0
        for c in range(n_classes):
            y_bin = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if y[i] == c:
                    y_bin[i] = 1.0
            Y_feat_c = _rdc_features(y_bin, k, s, rdc_seed + 1000 + c)
            rdc_c = _rdc_cancor(X_feat.copy(), Y_feat_c)
            if rdc_c > theta:
                theta = rdc_c

    # Precompute Y projection weights per class
    if n_classes == 2:
        n_weight_sets = 1
    else:
        n_weight_sets = n_classes
    wy0_all = np.empty((n_weight_sets, k), dtype=np.float64)
    wy1_all = np.empty((n_weight_sets, k), dtype=np.float64)
    for c in range(n_weight_sets):
        seed_c = rdc_seed + 1000 + c if n_classes > 2 else rdc_seed + 1000
        np.random.seed(seed_c)
        for j in range(k):
            wy0_all[c, j] = np.random.randn() * s
            wy1_all[c, j] = np.random.randn() * s

    # Full parallel permutation
    theta_p = np.empty(n_resamples, dtype=np.float64)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        if n_classes == 2:
            y_float_perm = y_perm.astype(np.float64)
            ecdf_y = _rdc_ecdf(y_float_perm)
            Y_feat_perm = np.empty((n, 2 * k), dtype=np.float64)
            for ii in range(n):
                for jj in range(k):
                    proj = ecdf_y[ii] * wy0_all[0, jj] + wy1_all[0, jj]
                    Y_feat_perm[ii, jj] = np.cos(proj)
                    Y_feat_perm[ii, k + jj] = np.sin(proj)
            theta_p[i] = _rdc_cancor(X_feat.copy(), Y_feat_perm)
        else:
            rdc_perm = 0.0
            for c in range(n_classes):
                y_bin = np.zeros(n, dtype=np.float64)
                for idx in range(n):
                    if y_perm[idx] == c:
                        y_bin[idx] = 1.0
                ecdf_y = _rdc_ecdf(y_bin)
                Y_feat_c = np.empty((n, 2 * k), dtype=np.float64)
                for ii in range(n):
                    for jj in range(k):
                        proj = ecdf_y[ii] * wy0_all[c, jj] + wy1_all[c, jj]
                        Y_feat_c[ii, jj] = np.cos(proj)
                        Y_feat_c[ii, k + jj] = np.sin(proj)
                rdc_c = _rdc_cancor(X_feat.copy(), Y_feat_c)
                if rdc_c > rdc_perm:
                    rdc_perm = rdc_c
            theta_p[i] = rdc_perm

    # +1 correction (Phipson & Smyth 2010)
    return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)


# Batched parallel permutation test for RDC (classifier) with adaptive stopping.
# Handles multi-class via max RDC over one-vs-all binary encodings.
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_rdc_classifier_parallel_batched(
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
) -> float:
    """Parallel batched permutation test for RDC (classification) with adaptive stopping."""
    n = len(x)
    X_feat = _rdc_features(x, k, s, rdc_seed)

    # Observed statistic
    if n_classes == 2:
        y_float = y.astype(np.float64)
        Y_feat_obs = _rdc_features(y_float, k, s, rdc_seed + 1000)
        theta = _rdc_cancor(X_feat.copy(), Y_feat_obs)
    else:
        theta = 0.0
        for c in range(n_classes):
            y_bin = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if y[i] == c:
                    y_bin[i] = 1.0
            Y_feat_c = _rdc_features(y_bin, k, s, rdc_seed + 1000 + c)
            rdc_c = _rdc_cancor(X_feat.copy(), Y_feat_c)
            if rdc_c > theta:
                theta = rdc_c

    if theta <= 0.0:
        return 1.0

    # Precompute Y projection weights per class
    if n_classes == 2:
        n_weight_sets = 1
    else:
        n_weight_sets = n_classes
    wy0_all = np.empty((n_weight_sets, k), dtype=np.float64)
    wy1_all = np.empty((n_weight_sets, k), dtype=np.float64)
    for c in range(n_weight_sets):
        seed_c = rdc_seed + 1000 + c if n_classes > 2 else rdc_seed + 1000
        np.random.seed(seed_c)
        for j in range(k):
            wy0_all[c, j] = np.random.randn() * s
            wy1_all[c, j] = np.random.randn() * s

    # Parallel batched permutation
    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0

    while m < n_resamples:
        batch_size = min(_BATCH_SIZE_PARALLEL, n_resamples - m)
        batch_extreme = np.zeros(batch_size, dtype=np.int64)

        for i in prange(batch_size):
            np.random.seed(random_state + m + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)

            if n_classes == 2:
                y_float_perm = y_perm.astype(np.float64)
                ecdf_y = _rdc_ecdf(y_float_perm)
                Y_feat_perm = np.empty((n, 2 * k), dtype=np.float64)
                for ii in range(n):
                    for jj in range(k):
                        proj = ecdf_y[ii] * wy0_all[0, jj] + wy1_all[0, jj]
                        Y_feat_perm[ii, jj] = np.cos(proj)
                        Y_feat_perm[ii, k + jj] = np.sin(proj)
                rdc_perm = _rdc_cancor(X_feat.copy(), Y_feat_perm)
            else:
                rdc_perm = 0.0
                for c in range(n_classes):
                    y_bin = np.zeros(n, dtype=np.float64)
                    for idx in range(n):
                        if y_perm[idx] == c:
                            y_bin[idx] = 1.0
                    ecdf_y = _rdc_ecdf(y_bin)
                    Y_feat_c = np.empty((n, 2 * k), dtype=np.float64)
                    for ii in range(n):
                        for jj in range(k):
                            proj = ecdf_y[ii] * wy0_all[c, jj] + wy1_all[c, jj]
                            Y_feat_c[ii, jj] = np.cos(proj)
                            Y_feat_c[ii, k + jj] = np.sin(proj)
                    rdc_c = _rdc_cancor(X_feat.copy(), Y_feat_c)
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
                return (extreme_count + 1) / (m + 1)
            if (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1)

    # +1 correction (Phipson & Smyth 2010)
    return (extreme_count + 1) / (n_resamples + 1)


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

    return _d_correlation(x, y) if standardize else _d_covariance(x, y)


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

_RDC_K = 10  # Number of random projections (paper uses 20, we use 10 for speed)
_RDC_S = 1.0 / 6.0  # Bandwidth parameter


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_ecdf(x: np.ndarray) -> np.ndarray:
    """Empirical CDF transform: ecdf(x)(x) = rank(x) / n."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        ranks[order[i]] = (i + 1) / n
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
def _rdc_cancor(X: np.ndarray, Y: np.ndarray) -> float:
    """Largest canonical correlation between X and Y feature matrices.

    Computes max absolute correlation between standardized columns.
    """
    n, p = X.shape
    q = Y.shape[1]

    # Standardize X columns (in-place)
    for j in range(p):
        mu = 0.0
        for i in range(n):
            mu += X[i, j]
        mu /= n
        ss = 0.0
        for i in range(n):
            X[i, j] -= mu
            ss += X[i, j] * X[i, j]
        if ss > 1e-10:
            inv_std = 1.0 / np.sqrt(ss)
            for i in range(n):
                X[i, j] *= inv_std

    # Standardize Y columns (in-place)
    for j in range(q):
        mu = 0.0
        for i in range(n):
            mu += Y[i, j]
        mu /= n
        ss = 0.0
        for i in range(n):
            Y[i, j] -= mu
            ss += Y[i, j] * Y[i, j]
        if ss > 1e-10:
            inv_std = 1.0 / np.sqrt(ss)
            for i in range(n):
                Y[i, j] *= inv_std

    # Find max |corr(X[:,j], Y[:,k])| - X,Y already standardized so corr = X'Y/n
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
    x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int | None = None
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
        return _rdc(x, y.astype(np.float64), _RDC_K, _RDC_S, seed)

    # Multi-class: max RDC over one-vs-all
    max_rdc = 0.0
    for c in range(n_classes):
        rdc_c = _rdc(x, (y == c).astype(np.float64), _RDC_K, _RDC_S, seed + c)
        if rdc_c > max_rdc:
            max_rdc = rdc_c
    return max_rdc


@RegressorSelectors.register("rdc")
def rdc_regressor(
    x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int | None = None
) -> float:
    """RDC for regression.

    O(n log n) non-linear dependence.
    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    seed = 42 if random_state is None else random_state
    return _rdc(x, y, _RDC_K, _RDC_S, seed)


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
            return _ptest_mc_parallel(
                x=x,
                y=y,
                n_classes=n_classes,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        elif early_stopping == EarlyStopping.ADAPTIVE:
            return _ptest_mc_parallel_batched(
                x=x,
                y=y,
                n_classes=n_classes,
                n_resamples=n_resamples,
                random_state=random_state,
                alpha=alpha,
                confidence=confidence,
            )

    return _ptest(
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
    return _ptest(
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
            return _ptest_pc_parallel(
                x=x,
                y=y,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        elif early_stopping == EarlyStopping.ADAPTIVE:
            return _ptest_pc_parallel_batched(
                x=x,
                y=y,
                n_resamples=n_resamples,
                random_state=random_state,
                alpha=alpha,
                confidence=confidence,
            )

    return _ptest(
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
    return _ptest(
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
            return _ptest_rdc_classifier_parallel(
                x=x,
                y=y,
                n_classes=n_classes,
                k=_RDC_K,
                s=_RDC_S,
                rdc_seed=random_state,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        elif early_stopping == EarlyStopping.ADAPTIVE:
            return _ptest_rdc_classifier_parallel_batched(
                x=x,
                y=y,
                n_classes=n_classes,
                k=_RDC_K,
                s=_RDC_S,
                rdc_seed=random_state,
                n_resamples=n_resamples,
                random_state=random_state,
                alpha=alpha,
                confidence=confidence,
            )

    return _ptest(
        func=rdc_classifier,
        func_arg=n_classes,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
        confidence=confidence,
    )


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
            return _ptest_rdc_regressor_parallel(
                x=x,
                y=y,
                k=_RDC_K,
                s=_RDC_S,
                rdc_seed=random_state,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        elif early_stopping == EarlyStopping.ADAPTIVE:
            return _ptest_rdc_regressor_parallel_batched(
                x=x,
                y=y,
                k=_RDC_K,
                s=_RDC_S,
                rdc_seed=random_state,
                n_resamples=n_resamples,
                random_state=random_state,
                alpha=alpha,
                confidence=confidence,
            )

    return _ptest(
        func=rdc_regressor,
        func_arg=standardize,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
        confidence=confidence,
    )
