from typing import Optional

import numpy as np
from numba import njit

from ._registry import ThresholdMethods


@ThresholdMethods.register("exact")
@njit(fastmath=True, nogil=True)
def exact(x: np.ndarray, max_thresholds: Optional[int] = None) -> np.ndarray:
    """Random permutation of all unique values of x.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int, default=None
        Maximum number of thresholds to generate. Kept here for API compatibility with other threshold methods.

    Returns
    -------
    np.ndarray
        Random permutation of unique values of x.
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.random.permutation(np.unique(x))


@ThresholdMethods.register("random")
@njit(fastmath=True, nogil=True)
def random(x: np.ndarray, max_thresholds: int) -> np.ndarray:
    """Random permutation of a random sample of unique values.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int
        Maximum number of thresholds to generate.

    Returns
    -------
    np.ndarray
        Random permutation of unique values of x.
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.random.choice(np.unique(x), size=max_thresholds, replace=False)


@ThresholdMethods.register("percentile")
@njit(fastmath=True, nogil=True)
def percentile(x: np.ndarray, max_thresholds: int) -> np.ndarray:
    """Random permutation of percentiles

    Parameters
    ----------

    Returns
    -------
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.random.permutation(np.percentile(x, q=max_thresholds))


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

    return np.random.permutation(np.histogram_bin_edges(x, bins=max_thresholds))
