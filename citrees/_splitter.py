from math import ceil
from typing import Any

import numpy as np
from numba import njit
from numba import prange as _numba_prange

from citrees._permutation import PermutationTestResult, record_permutation_test
from citrees._registry import (
    ClassifierSplitters,
    ClassifierSplitterTests,
    RegressorSplitters,
    RegressorSplitterTests,
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
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> PermutationTestResult:
    """Perform a permutation test for split selection.

    Parameters
    ----------
    func : Any
        Split selection function to use in permutation testing.

    x : np.ndarray
        Input data, usually the feature in the (x, y) pair.

    y : np.ndarray
        Input data, usually the target in the (x, y) pair.

    threshold : float
        Threshold used to create binary split on x.

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
        Confidence threshold for adaptive stopping.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.

    """
    # Use default_rng for isolated RNG stream (avoids global state contamination)
    rng = np.random.default_rng(random_state)

    idx = x <= threshold
    n = len(y)
    n_left = int(np.sum(idx))
    n_right = n - n_left
    if n_left == 0 or n_right == 0:
        return 1.0, 0

    w_left = n_left / n
    w_right = n_right / n
    theta = (w_left * func(y[idx])) + (w_right * func(y[~idx]))
    y_ = y.copy()

    if early_stopping is None:
        theta_p = np.empty(n_resamples)
        for i in range(n_resamples):
            rng.shuffle(y_)
            theta_p[i] = (w_left * func(y_[idx])) + (w_right * func(y_[~idx]))
        p_value = (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
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
                theta_p = (w_left * func(y_[idx])) + (w_right * func(y_[~idx]))
                if theta_p <= theta:
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
            theta_p = (w_left * func(y_[idx])) + (w_right * func(y_[~idx]))
            if theta_p <= theta:
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


# Parallel permutation test for Gini index (classifier)
# Serial split-statistic helpers for the parallel kernels. See the note above the
# selector helpers in ``_selector.py``: top-level array reductions inside an
# ``@njit(parallel=True)`` function are auto-parallelized and their fastmath
# summation order depends on the thread count, so the observed statistic must be
# computed by the same serial code as the permuted statistics.
@njit(cache=True, fastmath=True, nogil=True)
def _gini_split_stat(
    y_left: np.ndarray,
    y_right: np.ndarray,
    n_left: int,
    n_right: int,
    w_left: float,
    w_right: float,
) -> float:
    p_left = np.bincount(y_left) / n_left
    p_right = np.bincount(y_right) / n_right
    return w_left * (1 - np.sum(p_left * p_left)) + w_right * (1 - np.sum(p_right * p_right))


@njit(cache=True, fastmath=True, nogil=True)
def _mse_split_stat(
    y_left: np.ndarray, y_right: np.ndarray, w_left: float, w_right: float
) -> float:
    dev_left = y_left - y_left.mean()
    dev_right = y_right - y_right.mean()
    return w_left * np.mean(dev_left * dev_left) + w_right * np.mean(dev_right * dev_right)


@njit(cache=True, fastmath=True, nogil=True)
def _entropy_split_stat(
    y_left: np.ndarray,
    y_right: np.ndarray,
    n_left: int,
    n_right: int,
    w_left: float,
    w_right: float,
) -> float:
    p_left = np.bincount(y_left) / n_left
    entropy_left = 0.0
    for p in p_left:
        if p > 0:
            entropy_left -= p * np.log2(p)
    p_right = np.bincount(y_right) / n_right
    entropy_right = 0.0
    for p in p_right:
        if p > 0:
            entropy_right -= p * np.log2(p)
    return w_left * entropy_left + w_right * entropy_right


@njit(cache=True, fastmath=True, nogil=True)
def _mae_split_stat(
    y_left: np.ndarray, y_right: np.ndarray, w_left: float, w_right: float
) -> float:
    dev_left = np.abs(y_left - np.median(y_left))
    dev_right = np.abs(y_right - np.median(y_right))
    return w_left * np.mean(dev_left) + w_right * np.mean(dev_right)


# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_gini_parallel_result(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    random_state: int,
) -> PermutationTestResult:
    """Parallel permutation test for Gini index.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    idx = x <= threshold
    y_left = y[idx]
    y_right = y[~idx]

    # Compute observed statistic
    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return 1.0, 0
    n = len(y)
    w_left = n_left / n
    w_right = n_right / n

    theta = _gini_split_stat(y_left, y_right, n_left, n_right, w_left, w_right)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        y_left_perm = y_perm[idx]
        y_right_perm = y_perm[~idx]

        theta_p[i] = _gini_split_stat(y_left_perm, y_right_perm, n_left, n_right, w_left, w_right)

        # +1 correction (Phipson & Smyth 2010)
    p_value = (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
    return p_value, n_resamples


# Parallel permutation test for MSE (regressor)
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mse_parallel_result(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    random_state: int,
) -> PermutationTestResult:
    """Parallel permutation test for MSE.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    idx = x <= threshold
    y_left = y[idx]
    y_right = y[~idx]

    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return 1.0, 0
    n = len(y)
    w_left = n_left / n
    w_right = n_right / n

    # Compute observed statistic
    theta = _mse_split_stat(y_left, y_right, w_left, w_right)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        y_left_perm = y_perm[idx]
        y_right_perm = y_perm[~idx]

        theta_p[i] = _mse_split_stat(y_left_perm, y_right_perm, w_left, w_right)

        # +1 correction (Phipson & Smyth 2010)
    p_value = (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
    return p_value, n_resamples


# Parallel permutation test for Entropy (classifier)
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_entropy_parallel_result(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    random_state: int,
) -> PermutationTestResult:
    """Parallel permutation test for entropy.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    idx = x <= threshold
    y_left = y[idx]
    y_right = y[~idx]

    # Compute observed statistic
    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return 1.0, 0
    n = len(y)
    w_left = n_left / n
    w_right = n_right / n

    theta = _entropy_split_stat(y_left, y_right, n_left, n_right, w_left, w_right)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        y_left_perm = y_perm[idx]
        y_right_perm = y_perm[~idx]

        theta_p[i] = _entropy_split_stat(
            y_left_perm, y_right_perm, n_left, n_right, w_left, w_right
        )

        # +1 correction (Phipson & Smyth 2010)
    p_value = (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
    return p_value, n_resamples


# Parallel permutation test for MAE (regressor)
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mae_parallel_result(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    random_state: int,
) -> PermutationTestResult:
    """Parallel permutation test for MAE.

    Returns
    -------
    tuple[float, int]
        P-value and realized number of permutations.
    """
    idx = x <= threshold
    y_left = y[idx]
    y_right = y[~idx]

    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return 1.0, 0
    n = len(y)
    w_left = n_left / n
    w_right = n_right / n

    # Compute observed statistic
    theta = _mae_split_stat(y_left, y_right, w_left, w_right)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        y_left_perm = y_perm[idx]
        y_right_perm = y_perm[~idx]

        theta_p[i] = _mae_split_stat(y_left_perm, y_right_perm, w_left, w_right)

    p_value = (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
    return p_value, n_resamples


@ClassifierSplitters.register("gini")
@njit(cache=True, fastmath=True, nogil=True)
def gini(y: np.ndarray) -> float:
    """Gini index of node.

    Parameters
    ----------
    y : np.ndarray
        Training target.

    Returns
    -------
    float
        Gini index impurity.

    """
    if y.ndim > 1:
        y = y.ravel()

    n = len(y)
    p = np.bincount(y) / n
    return 1 - np.sum(p * p)


@ClassifierSplitters.register("entropy")
@njit(cache=True, fastmath=True, nogil=True)
def entropy(y: np.ndarray) -> float:
    """Entropy of node.

    Parameters
    ----------
    y : np.ndarray
        Training target.

    Returns
    -------
    float
        Entropy impurity.

    """
    if y.ndim > 1:
        y = y.ravel()

    n = len(y)
    p = np.bincount(y) / n
    p = p[p != 0]
    return -np.sum(np.log2(p) * p)


@ClassifierSplitterTests.register("gini")
def ptest_gini(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Permutation test for Gini index split selection.

    Parameters
    ----------
    x : np.ndarray
        Feature values used for splitting.

    y : np.ndarray
        Target values (class labels).

    threshold : float
        Threshold value for creating binary split.

    n_resamples : int
        Number of permutation resamples.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance level.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Achieved significance level (p-value).

    """
    if early_stopping is None and n_resamples >= _PARALLEL_THRESHOLD:
        result = _ptest_gini_parallel_result(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    else:
        result = _ptest_result(
            func=gini,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
            confidence=confidence,
        )
    return record_permutation_test("splitter", result)


@ClassifierSplitterTests.register("entropy")
def ptest_entropy(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Permutation test for entropy split selection.

    Parameters
    ----------
    x : np.ndarray
        Feature values used for splitting.

    y : np.ndarray
        Target values (class labels).

    threshold : float
        Threshold value for creating binary split.

    n_resamples : int
        Number of permutation resamples.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance level.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Achieved significance level (p-value).

    """
    if early_stopping is None and n_resamples >= _PARALLEL_THRESHOLD:
        result = _ptest_entropy_parallel_result(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    else:
        result = _ptest_result(
            func=entropy,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
            confidence=confidence,
        )
    return record_permutation_test("splitter", result)


@RegressorSplitters.register("mse")
@njit(cache=True, fastmath=True, nogil=True)
def mse(y: np.ndarray) -> float:
    """Mean squared error impurity of a node.

    Computes the variance of target values in a node, used as the
    impurity criterion for regression tree splits.

    Parameters
    ----------
    y : np.ndarray
        Target values in the node.

    Returns
    -------
    float
        Mean squared error (variance) of the target values.

    """
    if y.ndim > 1:
        y = y.ravel()

    dev = y - y.mean()
    dev *= dev

    return np.mean(dev)


@RegressorSplitters.register("mae")
@njit(cache=True, fastmath=True, nogil=True)
def mae(y: np.ndarray) -> float:
    """Mean absolute error impurity of a node.

    Computes the mean absolute deviation from the median for target values
    in a node. Using the median makes this an L1-optimal impurity criterion.

    Parameters
    ----------
    y : np.ndarray
        Target values in the node.

    Returns
    -------
    float
        Mean absolute error of the target values.

    """
    if y.ndim > 1:
        y = y.ravel()

    dev = np.abs(y - np.median(y))

    return np.mean(dev)


@RegressorSplitterTests.register("mse")
def ptest_mse(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Permutation test for MSE split selection.

    Parameters
    ----------
    x : np.ndarray
        Feature values used for splitting.

    y : np.ndarray
        Target values (continuous).

    threshold : float
        Threshold value for creating binary split.

    n_resamples : int
        Number of permutation resamples.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance level.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Achieved significance level (p-value).

    """
    if early_stopping is None and n_resamples >= _PARALLEL_THRESHOLD:
        result = _ptest_mse_parallel_result(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    else:
        result = _ptest_result(
            func=mse,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
            confidence=confidence,
        )
    return record_permutation_test("splitter", result)


@RegressorSplitterTests.register("mae")
def ptest_mae(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> float:
    """Permutation test for MAE split selection.

    Parameters
    ----------
    x : np.ndarray
        Feature values used for splitting.

    y : np.ndarray
        Target values (continuous).

    threshold : float
        Threshold value for creating binary split.

    n_resamples : int
        Number of permutation resamples.

    early_stopping : {"simple", "adaptive"} or None
        Early stopping method.

    alpha : float
        Significance level.

    random_state : int
        Random seed.

    confidence : float, default=0.95
        Confidence threshold for adaptive stopping.

    Returns
    -------
    float
        Achieved significance level (p-value).

    """
    if early_stopping is None and n_resamples >= _PARALLEL_THRESHOLD:
        result = _ptest_mae_parallel_result(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            random_state=random_state,
        )
    else:
        result = _ptest_result(
            func=mae,
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
            confidence=confidence,
        )
    return record_permutation_test("splitter", result)


# =============================================================================
# Max-type split test over all candidate thresholds
# =============================================================================
# One permutation test on the minimum impurity over the K candidate thresholds
# (Westfall and Young, 1993). Under the permutation null (labels exchangeable
# given the feature) the minimum over the same K candidates is recomputed for
# every permuted response, so the p-value is the inclusive rank of the observed
# minimum among B + 1 exchangeable values and satisfies P(p <= alpha) <= alpha for
# every alpha. This is weak familywise control over the K thresholds: under the
# global null the chance of declaring any split is at most alpha, regardless of
# how the thresholds are correlated. Ties count as extreme, which makes the test
# conservative rather than exact. The budget is about 1 / alpha permutations
# instead of K tests of K / alpha permutations each.
#
# Classifier and regressor kernels are separate because the impurity helpers
# need integer labels for bincount and floating targets for means and medians.
# The metric code is 0 = gini, 1 = entropy for classifiers and 0 = mse, 1 = mae
# for regressors. Observed and permuted statistics use the same serial helpers.
#
# Note: Uses np.random.seed() because Numba's Generator support is not thread-safe.
# Per-iteration seeding with (random_state + i) in prange is the recommended pattern
# for reproducible parallel RNG in Numba. See: https://github.com/numba/numba/issues/7686

_CLF_METRIC_CODES = {"gini": 0, "entropy": 1}
_REG_METRIC_CODES = {"mse": 0, "mae": 1}


@njit(cache=True, fastmath=True, nogil=True)
def _split_masks(x: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean left-side masks, one row per threshold, and the left counts."""
    n = x.shape[0]
    k = thresholds.shape[0]
    masks = np.empty((k, n), dtype=np.bool_)
    n_left = np.empty(k, dtype=np.int64)
    for j in range(k):
        count = 0
        for i in range(n):
            left = x[i] <= thresholds[j]
            masks[j, i] = left
            if left:
                count += 1
        n_left[j] = count
    return masks, n_left


@njit(cache=True, fastmath=True, nogil=True)
def _clf_min_split_stat(
    y: np.ndarray, masks: np.ndarray, n_left: np.ndarray, metric: int
) -> tuple[float, int]:
    """Minimum classification impurity over thresholds and the index attaining it."""
    n = y.shape[0]
    best = np.inf
    best_j = -1
    for j in range(masks.shape[0]):
        nl = n_left[j]
        nr = n - nl
        if nl == 0 or nr == 0:
            continue
        y_left = y[masks[j]]
        y_right = y[~masks[j]]
        w_left = nl / n
        w_right = nr / n
        if metric == 0:
            value = _gini_split_stat(y_left, y_right, nl, nr, w_left, w_right)
        else:
            value = _entropy_split_stat(y_left, y_right, nl, nr, w_left, w_right)
        if value < best:
            best = value
            best_j = j
    return best, best_j


@njit(cache=True, fastmath=True, nogil=True)
def _reg_min_split_stat(
    y: np.ndarray, masks: np.ndarray, n_left: np.ndarray, metric: int
) -> tuple[float, int]:
    """Minimum regression impurity over thresholds and the index attaining it."""
    n = y.shape[0]
    best = np.inf
    best_j = -1
    for j in range(masks.shape[0]):
        nl = n_left[j]
        nr = n - nl
        if nl == 0 or nr == 0:
            continue
        y_left = y[masks[j]]
        y_right = y[~masks[j]]
        w_left = nl / n
        w_right = nr / n
        if metric == 0:
            value = _mse_split_stat(y_left, y_right, w_left, w_right)
        else:
            value = _mae_split_stat(y_left, y_right, w_left, w_right)
        if value < best:
            best = value
            best_j = j
    return best, best_j


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_maxt_clf_parallel_result(
    y: np.ndarray,
    masks: np.ndarray,
    n_left: np.ndarray,
    metric: int,
    n_resamples: int,
    random_state: int,
) -> tuple[float, int, int]:
    """Full max-type permutation test for a classifier; returns p, permutations, best index."""
    theta, best_j = _clf_min_split_stat(y, masks, n_left, metric)
    if best_j < 0:
        return 1.0, 0, 0
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)
        theta_p[i] = _clf_min_split_stat(y_perm, masks, n_left, metric)[0]
    p_value = (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
    return p_value, n_resamples, best_j


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_maxt_reg_parallel_result(
    y: np.ndarray,
    masks: np.ndarray,
    n_left: np.ndarray,
    metric: int,
    n_resamples: int,
    random_state: int,
) -> tuple[float, int, int]:
    """Full max-type permutation test for a regressor; returns p, permutations, best index."""
    theta, best_j = _reg_min_split_stat(y, masks, n_left, metric)
    if best_j < 0:
        return 1.0, 0, 0
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)
        theta_p[i] = _reg_min_split_stat(y_perm, masks, n_left, metric)[0]
    p_value = (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)
    return p_value, n_resamples, best_j


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_maxt_clf_parallel_batched_result(
    y: np.ndarray,
    masks: np.ndarray,
    n_left: np.ndarray,
    metric: int,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
) -> tuple[float, int, int]:
    """Adaptive max-type permutation test for a classifier (Beta posterior stopping)."""
    theta, best_j = _clf_min_split_stat(y, masks, n_left, metric)
    if best_j < 0:
        return 1.0, 0, 0
    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0
    while m < n_resamples:
        batch_size = min(_ADAPTIVE_BATCH_SIZE, n_resamples - m)
        batch_extreme = np.zeros(batch_size, dtype=np.int64)
        for i in prange(batch_size):
            np.random.seed(random_state + m + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)
            if _clf_min_split_stat(y_perm, masks, n_left, metric)[0] <= theta:
                batch_extreme[i] = 1
        extreme_count += int(np.sum(batch_extreme))
        m += batch_size
        if m >= min_resamples:
            prob_sig = _beta_cdf(alpha, 1.0 + extreme_count, 1.0 + m - extreme_count)
            if prob_sig >= confidence or (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1), m, best_j
    return (extreme_count + 1) / (n_resamples + 1), n_resamples, best_j


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_maxt_reg_parallel_batched_result(
    y: np.ndarray,
    masks: np.ndarray,
    n_left: np.ndarray,
    metric: int,
    n_resamples: int,
    random_state: int,
    alpha: float,
    confidence: float,
) -> tuple[float, int, int]:
    """Adaptive max-type permutation test for a regressor (Beta posterior stopping)."""
    theta, best_j = _reg_min_split_stat(y, masks, n_left, metric)
    if best_j < 0:
        return 1.0, 0, 0
    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    extreme_count = 0
    m = 0
    while m < n_resamples:
        batch_size = min(_ADAPTIVE_BATCH_SIZE, n_resamples - m)
        batch_extreme = np.zeros(batch_size, dtype=np.int64)
        for i in prange(batch_size):
            np.random.seed(random_state + m + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)
            if _reg_min_split_stat(y_perm, masks, n_left, metric)[0] <= theta:
                batch_extreme[i] = 1
        extreme_count += int(np.sum(batch_extreme))
        m += batch_size
        if m >= min_resamples:
            prob_sig = _beta_cdf(alpha, 1.0 + extreme_count, 1.0 + m - extreme_count)
            if prob_sig >= confidence or (1.0 - prob_sig) >= confidence:
                return (extreme_count + 1) / (m + 1), m, best_j
    return (extreme_count + 1) / (n_resamples + 1), n_resamples, best_j


def ptest_maxt(
    x: np.ndarray,
    y: np.ndarray,
    thresholds: np.ndarray,
    splitter: str,
    n_resamples: int,
    early_stopping: EarlyStoppingOption,
    alpha: float,
    random_state: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Max-type permutation test over all candidate thresholds.

    Returns the p-value of the minimum impurity over ``thresholds`` and the
    threshold attaining it. With ``early_stopping`` set to None the full budget
    is used; otherwise the adaptive Beta-posterior rule stops the test early
    (the simple rule is mapped to the adaptive rule for this test).
    """
    thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
    if thresholds.size == 0:
        raise ValueError("ptest_maxt requires at least one candidate threshold")
    masks, n_left = _split_masks(np.ascontiguousarray(x, dtype=np.float64), thresholds)
    if splitter in _CLF_METRIC_CODES:
        metric = _CLF_METRIC_CODES[splitter]
        labels = np.ascontiguousarray(y, dtype=np.int64)
        if early_stopping is None:
            p_value, realized, best_j = _ptest_maxt_clf_parallel_result(
                labels, masks, n_left, metric, n_resamples, random_state
            )
        else:
            p_value, realized, best_j = _ptest_maxt_clf_parallel_batched_result(
                labels, masks, n_left, metric, n_resamples, random_state, alpha, confidence
            )
    elif splitter in _REG_METRIC_CODES:
        metric = _REG_METRIC_CODES[splitter]
        target = np.ascontiguousarray(y, dtype=np.float64)
        if early_stopping is None:
            p_value, realized, best_j = _ptest_maxt_reg_parallel_result(
                target, masks, n_left, metric, n_resamples, random_state
            )
        else:
            p_value, realized, best_j = _ptest_maxt_reg_parallel_batched_result(
                target, masks, n_left, metric, n_resamples, random_state, alpha, confidence
            )
    else:
        raise ValueError(f"splitter '{splitter}' has no max-type split test")
    record_permutation_test("splitter", (float(p_value), int(realized)))
    if realized == 0:
        # Every candidate left one side empty: nothing was tested, so no
        # threshold attained the statistic. The p-value of 1.0 already blocks
        # the split; report the threshold as undefined rather than inventing one.
        return float(p_value), float("nan")
    return float(p_value), float(thresholds[best_j])
