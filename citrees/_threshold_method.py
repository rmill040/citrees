from typing import Optional

import numpy as np
from numba import njit

from ._registry import ThresholdMethods


@ThresholdMethods.register("exact")
@njit(fastmath=True, nogil=True)
def exact(x: np.ndarray, max_thresholds: Optional[int] = None, random_state: Optional[int] = None) -> np.ndarray:
    """Unique midpoints in array.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int, default=None
        Maximum number of thresholds to generate. Kept here for API compatibility with other threshold methods.

    random_state : int, default=None
        Random seed. Kept here for API compatibility with other threshold methods.

    Returns
    -------
    np.ndarray
        Thresholds in array.
    """
    if x.ndim > 1:
        x = x.ravel()

    values = np.unique(x)
    midpoints = (values[:-1] + values[1:]) / 2

    return midpoints


@ThresholdMethods.register("random")
@njit(fastmath=True, nogil=True)
def random(x: np.ndarray, max_thresholds: int, random_state: int) -> np.ndarray:
    """Random sample of unique midpoints in array.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int
        Maximum number of thresholds to generate.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Thresholds in array.
    """
    prng = np.random.RandomState(random_state)

    if x.ndim > 1:
        x = x.ravel()

    values = np.unique(x)
    midpoints = (values[:-1] + values[1:]) / 2
    max_thresholds = min(len(midpoints), max_thresholds)

    return prng.choice(midpoints, size=max_thresholds, replace=False)


@ThresholdMethods.register("percentile")
@njit(fastmath=True, nogil=True)
def percentile(x: np.ndarray, max_thresholds: int, random_state: Optional[int] = None) -> np.ndarray:
    """Percentiles of array.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int
        Maximum number of thresholds to generate.

    random_state : int, default=None
        Random seed. Kept here for API compatibility with other threshold methods.

    Returns
    -------
    np.ndarray
        Thresholds in array.
    """
    if x.ndim > 1:
        x = x.ravel()

    q = np.linspace(0, 100, max_thresholds)
    return np.unique(np.percentile(x, q=q))


@ThresholdMethods.register("histogram")
@njit(fastmath=True, nogil=True)
def histogram(x: np.ndarray, max_thresholds: int, random_state: Optional[int] = None) -> np.ndarray:
    """Histogram bin edges of array.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    max_thresholds : int
        Maximum number of thresholds to generate.

    random_state : int, default=None
        Random seed. Kept here for API compatibility with other threshold methods.

    Returns
    -------
    np.ndarray
        Thresholds in array.
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.unique(np.histogram(x, bins=max_thresholds)[1])
