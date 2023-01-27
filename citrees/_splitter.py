from math import ceil
from typing import Any

import numpy as np
from numba import njit

from ._registry import ClassifierSplitters, ClassifierSplitterTests, RegressorSplitters, RegressorSplitterTests


def _permutation_test(
    *,
    func: Any,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
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

    early_stopping : bool
        Whether to early stop permutation testing if null hypothesis can be rejected.

    alpha : float
        Alpha level for significance testing.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated achieved significance level.
    """
    np.random.seed(random_state)

    idx = x <= threshold
    theta = func(y[idx]) + func(y[~idx])
    y_ = y.copy()
    theta_p = np.empty(n_resamples)

    if early_stopping:
        # Handle cases where n_resamples is less than min_resamples and early stopping is not possible
        min_resamples = ceil(1 / alpha)
        if n_resamples < min_resamples:
            n_resamples = min_resamples
            theta_p = np.empty(n_resamples)
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p[i] = func(y_[idx]) + func(y_[~idx])
            if i >= min_resamples - 1:
                asl = np.mean(theta_p[: i + 1] <= theta)
                if asl < alpha:
                    break

    else:
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p[i] = func(y_[idx]) + func(y_[~idx])
        asl = np.mean(theta_p <= theta)

    return asl


# Compiled version of permutation test
_permutation_test_compiled = njit(fastmath=True, nogil=True)(_permutation_test)


@ClassifierSplitters.register("gini")
@njit(cache=True, fastmath=True, nogil=True)
def gini_index(y: np.ndarray) -> float:
    """Calculate gini index.

    Parameters
    ----------
    y : np.ndarray
        Training target.

    Returns
    -------
    """
    if y.ndim > 1:
        y = y.ravel()

    n = len(y)
    p = np.bincount(y) / n
    return 1 - np.sum(p * p)


@ClassifierSplitterTests.register("gini")
def permutation_test_gini_index(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    return _permutation_test_compiled(
        func=gini_index,
        x=x,
        y=y,
        threshold=threshold,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@RegressorSplitters.register("mse")
@njit(fastmath=True, nogil=True)
def mean_squared_error(y: np.ndarray) -> float:
    """Mean squared error.

    Parameters
    ----------

    Returns
    -------
    """
    if y.ndim > 1:
        y = y.ravel()

    dev = y - y.mean()
    dev *= dev

    return np.mean(dev)


@RegressorSplitters.register("mae")
def mean_absolute_error(y: np.ndarray) -> float:
    """Mean absolute error.

    Parameters
    ----------

    Returns
    -------
    """
    if y.ndim > 1:
        y = y.ravel()

    return np.mean(np.abs(y - y.mean()), 2)


@RegressorSplitterTests.register("mse")
def permutation_test_mse(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    return _permutation_test_compiled(
        func=mean_squared_error,
        x=x,
        y=y,
        threshold=threshold,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@RegressorSplitterTests.register("mae")
def permutation_test_mae(
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    return _permutation_test_compiled(
        func=mean_absolute_error,
        x=x,
        y=y,
        threshold=threshold,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )
