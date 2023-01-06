from math import ceil
from typing import Any, Optional

import numpy as np
from numba import njit

from ._registry import (
    ClassifierSplitters,
    ClassifierSplitterTests,
    RegressorSplitters,
    RegressorSplitterTests,
    ThresholdMethods,
)


def _permutation_test(
    func: Any,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Perform a permutation test."""
    np.random.seed(random_state)

    idx = x <= threshold
    theta = func(y[idx]) + func(y[~idx])
    y_ = y.copy()
    theta_p = np.empty(n_resamples)

    if early_stopping:
        asl = None
        # Handle cases where n_resamples is less than min_resamples and early stopping is not possible
        min_resamples = ceil(1 / alpha)
        if n_resamples < min_resamples:
            n_resamples = min_resamples
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p[i] = func(y_[idx]) + func(y_[~idx])
            if i >= min_resamples:
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
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if y.ndim > 1:
        y = y.ravel()

    n = len(y)
    p = np.bincount(y) / n
    return 1 - np.sum(p * p)


@ClassifierSplitterTests.register("gini")
def permutation_test_gini(
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


@ThresholdMethods.register("exact")
@njit(fastmath=True, nogil=True)
def exact(x: np.ndarray, max_thresholds: Optional[int] = None) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.unique(x)


@ThresholdMethods.register("random")
@njit(fastmath=True, nogil=True)
def random(x: np.ndarray, max_thresholds: int) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.random.choice(x, size=max_thresholds, replace=False)


@ThresholdMethods.register("percentile")
@njit(fastmath=True, nogil=True)
def percentile(x: np.ndarray, max_thresholds: int) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.percentile(x, q=q)


@ThresholdMethods.register("histogram")
@njit(fastmath=True, nogil=True)
def histogram(x: np.ndarray, max_thresholds: int) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.histogram_bin_edges(x, bins=max_thresholds)
