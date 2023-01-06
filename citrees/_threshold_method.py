from typing import Optional

import numpy as np
from numba import njit

from ._registry import ThresholdMethods


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

    return np.percentile(x, q=max_thresholds)


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
