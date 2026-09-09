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
# Standardized max-type split test over all candidate thresholds
# =============================================================================
# One permutation test on all K candidate thresholds at once. For every
# permutation b (and for the observed labels, row 0) the weighted child impurity
# is computed at every candidate, giving a (B + 1) x K matrix. Each column is
# standardized by the mean and standard deviation of its B + 1 entries, and the
# test statistic of a row is the minimum standardized impurity across candidates
# (the max-T construction of Westfall and Young, 1993; the same standardization
# that conditional inference trees apply to their split statistics, Hothorn,
# Hornik and Zeileis, 2006, Section 3). Standardizing puts noisy candidates,
# such as thresholds that isolate a few extreme observations, on the same scale
# as the others, so the minimum is not dominated by whichever candidate has the
# widest null distribution.
#
# Validity: under label exchangeability given the feature, the B + 1 rows are
# exchangeable. Column standardization uses symmetric functions of the rows and
# the row minimum is a fixed function of each row, so the B + 1 row statistics
# stay exchangeable, and the inclusive-rank p-value
# (1 + #{permuted <= observed}) / (B + 1) satisfies P(p <= alpha) <= alpha for
# every alpha. This is weak familywise control over the K thresholds; ties count
# as extreme, so the test is conservative rather than exact. The chosen threshold
# is the candidate with the smallest observed standardized impurity. Candidates
# that leave one side empty, or whose impurity is constant across permutations,
# carry no information and are excluded from the minimum. See docs/maxt.md.
#
# Classifier and regressor kernels are separate because the impurity helpers
# need integer labels for bincount and floating targets for means and medians.
# Metric codes: 0 = gini, 1 = entropy (classifier); 0 = mse, 1 = mae (regressor).
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
def _clf_split_stats(
    y: np.ndarray, masks: np.ndarray, n_left: np.ndarray, metric: int, out: np.ndarray
) -> None:
    """Weighted classification impurity at every threshold; inf where a side is empty."""
    n = y.shape[0]
    for j in range(masks.shape[0]):
        nl = n_left[j]
        nr = n - nl
        if nl == 0 or nr == 0:
            out[j] = np.inf
            continue
        y_left = y[masks[j]]
        y_right = y[~masks[j]]
        if metric == 0:
            out[j] = _gini_split_stat(y_left, y_right, nl, nr, nl / n, nr / n)
        else:
            out[j] = _entropy_split_stat(y_left, y_right, nl, nr, nl / n, nr / n)


@njit(cache=True, fastmath=True, nogil=True)
def _reg_split_stats(
    y: np.ndarray, masks: np.ndarray, n_left: np.ndarray, metric: int, out: np.ndarray
) -> None:
    """Weighted regression impurity at every threshold; inf where a side is empty."""
    n = y.shape[0]
    for j in range(masks.shape[0]):
        nl = n_left[j]
        nr = n - nl
        if nl == 0 or nr == 0:
            out[j] = np.inf
            continue
        y_left = y[masks[j]]
        y_right = y[~masks[j]]
        if metric == 0:
            out[j] = _mse_split_stat(y_left, y_right, nl / n, nr / n)
        else:
            out[j] = _mae_split_stat(y_left, y_right, nl / n, nr / n)


@njit(cache=True, fastmath=True, nogil=True)
def _standardized_min_test(
    observed: np.ndarray, permuted: np.ndarray, m: int
) -> tuple[float, int, int]:
    """Standardized max-type test from the observed row and the first ``m`` permuted rows.

    Returns the p-value, the number of permuted rows at least as extreme as the
    observed row, and the index of the candidate with the smallest observed
    standardized impurity (-1 when no candidate is informative).
    """
    k = observed.shape[0]
    count_rows = m + 1
    mean = np.empty(k)
    scale = np.empty(k)
    usable = np.zeros(k, dtype=np.bool_)
    for j in range(k):
        if not np.isfinite(observed[j]):
            continue
        total = observed[j]
        for b in range(m):
            total += permuted[b, j]
        mu = total / count_rows
        ss = (observed[j] - mu) ** 2
        for b in range(m):
            d = permuted[b, j] - mu
            ss += d * d
        sd = np.sqrt(ss / count_rows)
        if sd > 0.0:
            mean[j] = mu
            scale[j] = sd
            usable[j] = True
    theta = np.inf
    best_j = -1
    for j in range(k):
        if usable[j]:
            z = (observed[j] - mean[j]) / scale[j]
            if z < theta:
                theta = z
                best_j = j
    if best_j < 0:
        return 1.0, 0, -1
    extreme = 0
    for b in range(m):
        row_min = np.inf
        for j in range(k):
            if usable[j]:
                z = (permuted[b, j] - mean[j]) / scale[j]
                if z < row_min:
                    row_min = z
        if row_min <= theta:
            extreme += 1
    return (1.0 + extreme) / (1.0 + m), extreme, best_j


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_maxt_clf_parallel_result(
    y: np.ndarray,
    masks: np.ndarray,
    n_left: np.ndarray,
    metric: int,
    n_resamples: int,
    random_state: int,
) -> tuple[float, int, int]:
    """Full-budget standardized max-type test for a classifier; returns p, permutations, best index."""
    k = masks.shape[0]
    observed = np.empty(k)
    _clf_split_stats(y, masks, n_left, metric, observed)
    permuted = np.empty((n_resamples, k))
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)
        _clf_split_stats(y_perm, masks, n_left, metric, permuted[i])
    p_value, _, best_j = _standardized_min_test(observed, permuted, n_resamples)
    if best_j < 0:
        return 1.0, 0, -1
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
    """Full-budget standardized max-type test for a regressor; returns p, permutations, best index."""
    k = masks.shape[0]
    observed = np.empty(k)
    _reg_split_stats(y, masks, n_left, metric, observed)
    permuted = np.empty((n_resamples, k))
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)
        _reg_split_stats(y_perm, masks, n_left, metric, permuted[i])
    p_value, _, best_j = _standardized_min_test(observed, permuted, n_resamples)
    if best_j < 0:
        return 1.0, 0, -1
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
    """Adaptive standardized max-type test for a classifier (Beta posterior stopping).

    The column moments are recomputed from all permutations drawn so far after
    each batch, so the standardized observed value is re-evaluated as the
    budget grows; the reported value is a stopping-time estimate, as for the
    other adaptive kernels.
    """
    k = masks.shape[0]
    observed = np.empty(k)
    _clf_split_stats(y, masks, n_left, metric, observed)
    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    permuted = np.empty((n_resamples, k))
    m = 0
    while m < n_resamples:
        batch_size = min(_ADAPTIVE_BATCH_SIZE, n_resamples - m)
        for i in prange(batch_size):
            np.random.seed(random_state + m + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)
            _clf_split_stats(y_perm, masks, n_left, metric, permuted[m + i])
        m += batch_size
        if m >= min_resamples:
            p_value, extreme, best_j = _standardized_min_test(observed, permuted, m)
            if best_j < 0:
                return 1.0, 0, -1
            prob_sig = _beta_cdf(alpha, 1.0 + extreme, 1.0 + m - extreme)
            if prob_sig >= confidence or (1.0 - prob_sig) >= confidence:
                return p_value, m, best_j
    p_value, _, best_j = _standardized_min_test(observed, permuted, n_resamples)
    if best_j < 0:
        return 1.0, 0, -1
    return p_value, n_resamples, best_j


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
    """Adaptive standardized max-type test for a regressor (Beta posterior stopping)."""
    k = masks.shape[0]
    observed = np.empty(k)
    _reg_split_stats(y, masks, n_left, metric, observed)
    min_resamples = int(np.ceil(1.0 / alpha))
    if n_resamples < min_resamples:
        n_resamples = min_resamples
    permuted = np.empty((n_resamples, k))
    m = 0
    while m < n_resamples:
        batch_size = min(_ADAPTIVE_BATCH_SIZE, n_resamples - m)
        for i in prange(batch_size):
            np.random.seed(random_state + m + i)
            y_perm = y.copy()
            np.random.shuffle(y_perm)
            _reg_split_stats(y_perm, masks, n_left, metric, permuted[m + i])
        m += batch_size
        if m >= min_resamples:
            p_value, extreme, best_j = _standardized_min_test(observed, permuted, m)
            if best_j < 0:
                return 1.0, 0, -1
            prob_sig = _beta_cdf(alpha, 1.0 + extreme, 1.0 + m - extreme)
            if prob_sig >= confidence or (1.0 - prob_sig) >= confidence:
                return p_value, m, best_j
    p_value, _, best_j = _standardized_min_test(observed, permuted, n_resamples)
    if best_j < 0:
        return 1.0, 0, -1
    return p_value, n_resamples, best_j


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
    """Standardized max-type permutation test over all candidate thresholds.

    Returns the p-value of the minimum standardized impurity over ``thresholds``
    and the threshold attaining it (NaN when no candidate is informative). With
    ``early_stopping`` set to None the full budget is used; otherwise the
    adaptive Beta-posterior rule stops the test early (the simple rule is mapped
    to the adaptive rule for this test).
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
    if best_j < 0:
        # No candidate was informative (every side empty, or impurity constant
        # across permutations). The p-value of 1.0 blocks the split; report the
        # threshold as undefined rather than inventing one.
        return float(p_value), float("nan")
    return float(p_value), float(thresholds[best_j])
