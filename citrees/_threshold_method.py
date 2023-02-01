from typing import Optional

import numpy as np
from numba import njit

from ._registry import ThresholdMethods


@ThresholdMethods.register("exact")
@njit(fastmath=True, nogil=True)
def exact(x: np.ndarray, max_thresholds: Optional[int] = None) -> np.ndarray:
    """Random permutation of unique midpoints in array.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int, default=None
        Maximum number of thresholds to generate. Kept here for API compatibility with other threshold methods.

    Returns
    -------
    np.ndarray
        Thresholds in array.
    """
    if x.ndim > 1:
        x = x.ravel()

    values = np.unique(x)
    midpoints = (values[:-1] + values[1:]) / 2

    return np.random.permutation(midpoints)


@ThresholdMethods.register("random")
@njit(fastmath=True, nogil=True)
def random(x: np.ndarray, max_thresholds: int) -> np.ndarray:
    """Random permutation of a random sample of unique midpoints in array.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int
        Maximum number of thresholds to generate.

    Returns
    -------
    np.ndarray
        Thresholds in array.
    """
    if x.ndim > 1:
        x = x.ravel()

    values = np.unique(x)
    midpoints = (values[:-1] + values[1:]) / 2

    return np.random.choice(midpoints, size=max_thresholds, replace=False)


@ThresholdMethods.register("percentile")
@njit(fastmath=True, nogil=True)
def percentile(x: np.ndarray, max_thresholds: int) -> np.ndarray:
    """Random permutation of percentiles of array.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int
        Maximum number of thresholds to generate.

    Returns
    -------
    np.ndarray
        Thresholds in array.
    """
    if x.ndim > 1:
        x = x.ravel()

    values = np.unique(np.percentile(x, q=max_thresholds))

    return np.random.permutation(values)


@ThresholdMethods.register("histogram")
@njit(fastmath=True, nogil=True)
def histogram(x: np.ndarray, max_thresholds: int) -> np.ndarray:
    """Random permutation of histogram bin edges of array.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int
        Maximum number of thresholds to generate.

    Returns
    -------
    np.ndarray
        Thresholds in array.
    """
    if x.ndim > 1:
        x = x.ravel()

    values = np.unique(np.histogram_bin_edges(x, bins=max_thresholds))

    return np.random.permutation(values)
