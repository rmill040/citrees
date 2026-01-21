
import numpy as np
from numba import njit

from citrees._registry import ThresholdMethods


@ThresholdMethods.register("exact")
@njit(cache=True, fastmath=True, nogil=True)
def exact(
    x: np.ndarray, max_thresholds: int | None = None, random_state: int | None = None
) -> np.ndarray:
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


# Note: Uses np.random.seed() because Numba doesn't support default_rng() inside @njit.
@ThresholdMethods.register("random")
@njit(cache=True, fastmath=True, nogil=True)
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
    np.random.seed(random_state)

    if x.ndim > 1:
        x = x.ravel()

    values = np.unique(x)
    midpoints = (values[:-1] + values[1:]) / 2
    max_thresholds = min(len(midpoints), max_thresholds)

    return np.random.choice(midpoints, size=max_thresholds, replace=False)


@ThresholdMethods.register("percentile")
@njit(cache=True, fastmath=True, nogil=True)
def percentile(
    x: np.ndarray, max_thresholds: int, random_state: int | None = None
) -> np.ndarray:
    """Percentiles of midpoints in array.

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

    values = np.unique(x)
    if len(values) < 2:
        return np.empty(0, dtype=np.float64)

    midpoints = (values[:-1] + values[1:]) / 2
    max_thresholds = min(len(midpoints), max_thresholds)
    q = np.linspace(0, 100, max_thresholds)

    return np.unique(np.percentile(midpoints, q=q))


@ThresholdMethods.register("histogram")
@njit(cache=True, fastmath=True, nogil=True)
def histogram(x: np.ndarray, max_thresholds: int, random_state: int | None = None) -> np.ndarray:
    """Histogram bin edges of midpoints in array.

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

    values = np.unique(x)
    if len(values) < 2:
        return np.empty(0, dtype=np.float64)

    midpoints = (values[:-1] + values[1:]) / 2
    max_thresholds = min(len(midpoints), max_thresholds)

    return np.unique(np.histogram(midpoints, bins=max_thresholds)[1])
