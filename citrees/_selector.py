from numba import njit
import numpy as np
from sklearn.feature_selection import mutual_info_classif

from ._registry import clf_selectors


@clf_selectors.register("mc")
@njit(nogil=True, fastmath=True)
def mc(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Calculate multiple correlation.

    Parameters
    ----------
    x : np.ndarray
        Array of features.

    y : np.ndarray
        Array of labels.

    Returns
    -------
    float
        Estimated multiple correlation.
    """
    # Sum of squares total (SST)
    mu = 0.0
    n = len(x)

    for value in x:
        mu += value
    mu /= n

    dev = x - mu
    sst = np.sum(dev * dev)

    # Sum of squares between (SSB)
    ssb = 0.0
    for j in classes:
        x_j = x[y == j]
        n_j = len(x_j)

        if not n_j:
            continue

        mu_j = 0.0
        for value in x_j:
            mu_j += value
        mu_j /= n_j

        dev_j = mu_j - mu
        ssb += n_j * dev_j * dev_j

    return np.sqrt(ssb / sst)


@clf_selectors.register("mi")
def mi(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate mutual information.

    Parameters
    ----------
    x : np.ndarray
        Array of features.

    y : np.ndarray
        Array of labels.

    Returns
    -------
    float
        Estimated mutual information.
    """
    if x.ndim == 1:
        x = x[:, None]

    return mutual_info_classif(x, y)[0]


# class BaseSelector:
#     pass


# class AutoSelector:
#     pass


# class PermutationSelector:
#     pass


# class ExactSelector:
#     pass
