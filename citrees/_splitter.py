from typing import Union

import numpy as np
from numba import njit

from ._registry import splitters, thresholds


@thresholds.register("random")
@njit(fastmath=True, nogil=True)
def random(x: np.ndarray, size: int) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.random.choice(x, size=size, replace=False)


@thresholds.register("percentile")
@njit(fastmath=True, nogil=True)
def percentile(x: np.ndarray, q: Union[int, np.ndarray]) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.percentile(x, q=q)


@thresholds.register("histogram")
@njit(fastmath=True, nogil=True)
def histogram(x: np.ndarray, bins: Union[int, str, np.ndarray]) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if x.ndim > 1:
        x = x.ravel()

    return np.histogram_bin_edges(x, bins=bins)[1]


@splitters.register("gini")
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
