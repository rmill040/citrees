from math import ceil
from typing import Any, Literal

import numpy as np
from numba import njit, prange

from citrees._registry import (
    ClassifierSplitters,
    ClassifierSplitterTests,
    RegressorSplitters,
    RegressorSplitterTests,
)
from citrees._sequential import _beta_cdf

EarlyStoppingOption = Literal["simple", "adaptive"] | None

# Threshold for using parallel permutation tests
_PARALLEL_THRESHOLD = 200

# P-value correction: Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero."
# SAGMB 9(1):39. https://pubmed.ncbi.nlm.nih.gov/21044043/
# Use p = (b+1)/(m+1) instead of p = b/m to avoid p=0.
# Note: min_resamples = ceil(1/alpha) remains valid since 1/(m+1) < alpha.


def _ptest(
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
) -> float:
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
        - "adaptive": Bayesian Beta CDF stopping (valid Type I error, default)
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
    float
        Estimated achieved significance level.
    """
    np.random.seed(random_state)

    idx = x <= threshold
    theta = func(y[idx]) + func(y[~idx])
    y_ = y.copy()

    if early_stopping is None:
        theta_p = np.empty(n_resamples)
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p[i] = func(y_[idx]) + func(y_[~idx])
        return (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)

    min_resamples = ceil(1 / alpha)
    n_resamples = max(n_resamples, min_resamples)
    extreme_count = 0

    if early_stopping == "adaptive":
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p = func(y_[idx]) + func(y_[~idx])
            if theta_p <= theta:
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

    else:  # simple
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p = func(y_[idx]) + func(y_[~idx])
            if theta_p <= theta:
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


# Parallel permutation test for Gini index (classifier)
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_gini_parallel(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    random_state: int,
) -> float:
    """Parallel permutation test for Gini index."""
    idx = x <= threshold
    y_left = y[idx]
    y_right = y[~idx]

    # Compute observed statistic
    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return 1.0

    p_left = np.bincount(y_left) / n_left
    p_right = np.bincount(y_right) / n_right
    theta = (1 - np.sum(p_left * p_left)) + (1 - np.sum(p_right * p_right))

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        y_left_perm = y_perm[idx]
        y_right_perm = y_perm[~idx]

        p_left_perm = np.bincount(y_left_perm) / n_left
        p_right_perm = np.bincount(y_right_perm) / n_right
        theta_p[i] = (1 - np.sum(p_left_perm * p_left_perm)) + (
            1 - np.sum(p_right_perm * p_right_perm)
        )

        # +1 correction (Phipson & Smyth 2010)
    return (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)


# Parallel permutation test for MSE (regressor)
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mse_parallel(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    random_state: int,
) -> float:
    """Parallel permutation test for MSE."""
    idx = x <= threshold
    y_left = y[idx]
    y_right = y[~idx]

    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return 1.0

    # Compute observed statistic
    dev_left = y_left - y_left.mean()
    dev_right = y_right - y_right.mean()
    theta = np.mean(dev_left * dev_left) + np.mean(dev_right * dev_right)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        y_left_perm = y_perm[idx]
        y_right_perm = y_perm[~idx]

        dev_left_perm = y_left_perm - y_left_perm.mean()
        dev_right_perm = y_right_perm - y_right_perm.mean()
        theta_p[i] = np.mean(dev_left_perm * dev_left_perm) + np.mean(
            dev_right_perm * dev_right_perm
        )

        # +1 correction (Phipson & Smyth 2010)
    return (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)


# Parallel permutation test for Entropy (classifier)
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_entropy_parallel(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    random_state: int,
) -> float:
    """Parallel permutation test for entropy."""
    idx = x <= threshold
    y_left = y[idx]
    y_right = y[~idx]

    # Compute observed statistic
    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return 1.0

    # Entropy left
    p_left = np.bincount(y_left) / n_left
    entropy_left = 0.0
    for p in p_left:
        if p > 0:
            entropy_left -= p * np.log2(p)

    # Entropy right
    p_right = np.bincount(y_right) / n_right
    entropy_right = 0.0
    for p in p_right:
        if p > 0:
            entropy_right -= p * np.log2(p)

    theta = entropy_left + entropy_right

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        y_left_perm = y_perm[idx]
        y_right_perm = y_perm[~idx]

        # Entropy left perm
        p_left_perm = np.bincount(y_left_perm) / n_left
        entropy_left_perm = 0.0
        for p in p_left_perm:
            if p > 0:
                entropy_left_perm -= p * np.log2(p)

        # Entropy right perm
        p_right_perm = np.bincount(y_right_perm) / n_right
        entropy_right_perm = 0.0
        for p in p_right_perm:
            if p > 0:
                entropy_right_perm -= p * np.log2(p)

        theta_p[i] = entropy_left_perm + entropy_right_perm

        # +1 correction (Phipson & Smyth 2010)
    return (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)


# Parallel permutation test for MAE (regressor)
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mae_parallel(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    random_state: int,
) -> float:
    """Parallel permutation test for MAE."""
    idx = x <= threshold
    y_left = y[idx]
    y_right = y[~idx]

    n_left = len(y_left)
    n_right = len(y_right)
    if n_left == 0 or n_right == 0:
        return 1.0

    # Compute observed statistic
    dev_left = np.abs(y_left - y_left.mean())
    dev_right = np.abs(y_right - y_right.mean())
    theta = np.mean(dev_left) + np.mean(dev_right)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        y_left_perm = y_perm[idx]
        y_right_perm = y_perm[~idx]

        dev_left_perm = np.abs(y_left_perm - y_left_perm.mean())
        dev_right_perm = np.abs(y_right_perm - y_right_perm.mean())
        theta_p[i] = np.mean(dev_left_perm) + np.mean(dev_right_perm)

    return (1 + np.sum(theta_p <= theta)) / (1 + n_resamples)


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
        return _ptest_gini_parallel(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return _ptest(
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
        return _ptest_entropy_parallel(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return _ptest(
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

    Computes the mean absolute deviation from the mean for target values
    in a node, used as a robust impurity criterion for regression tree splits.

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

    dev = np.abs(y - y.mean())

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
        return _ptest_mse_parallel(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return _ptest(
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
        return _ptest_mae_parallel(
            x=x,
            y=y,
            threshold=threshold,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return _ptest(
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
