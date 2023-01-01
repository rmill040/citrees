from numba import njit
import numpy as np


@njit(fastmath=True, nogil=True)
def random_sample(x: np.ndarray, size: int, replace: bool = False):
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    return np.random.choice(x, size=size, replace=replace)
