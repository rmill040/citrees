import numpy as np
from numba import njit

from ._registry import splitters


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
