import numpy as np
from numba import njit
from sklearn.feature_selection import mutual_info_classif

from ._registry import selectors


@selectors.register("mc")
@njit(nogil=True, fastmath=True)
def multiple_correlation(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    """Calculate multiple correlation.

    Parameters
    ----------
    x : np.ndarray
        Array of features.

    y : np.ndarray
        Array of labels.

    classes : np.ndarray
        Array of unique class labels.

    Returns
    -------
    float
        Estimated multiple correlation.
    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    # Sum of squares total (SST)
    mu = 0.0
    n = len(x)

    for value in x:
        mu += value
    mu /= n

    sst = 0.0
    dev = x - mu
    dev *= dev
    for value in dev:
        sst += value

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
        dev_j *= dev_j
        ssb += n_j * dev_j

    return np.sqrt(ssb / sst)


@selectors.register("mi")
def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
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


@selectors.register("pc")
@njit(nogil=True, fastmath=True)
def pearson_correlation(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    """Calculate Pearson correlation.

    Parameters
    ----------
    x : np.ndarray
        Array of features.

    y : np.ndarray
        Array of labels.

    standardize : bool, optional (default=True)
        Whether to standardize the return value, if True, Pearson correlation returned, if False covariance returned.

    Returns
    -------
    float
        Estimated Pearson correlation.
    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    return _correlation(x, y) if standardize else _covariance(x, y)


@njit(nogil=True, fastmath=True)
def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate covariance.

    Parameters
    ----------
    x : np.ndarray
        Array of features.

    y : np.ndarray
        Array of labels.

    Returns
    -------
    float
        Estimated covariance.
    """
    n = len(x)
    sx = 0.0
    sy = 0.0
    sxy = 0.0

    for i in range(n):
        xi = x[i]
        yi = y[i]
        sx += xi
        sy += yi
        sxy += xi * yi

    return (sxy - (sx * sy / n)) / n


@njit(nogil=True, fastmath=True)
def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Pearson correlation.

    Parameters
    ----------
    x : np.ndarray
        Array of features.

    y : np.ndarray
        Array of labels.

    Returns
    -------
    float
        Estimated Pearson correlation.
    """
    n = len(x)
    sx = 0.0
    sy = 0.0
    sx2 = 0.0
    sy2 = 0.0
    sxy = 0.0

    for i in range(n):
        xi = x[i]
        yi = y[i]
        sx += xi
        sx2 += xi * xi
        sy += yi
        sy2 += yi * yi
        sxy += xi * yi

    cov = n * sxy - sx * sy
    ssx = n * sx2 - sx * sx
    ssy = n * sy2 - sy * sy

    return 0.0 if ssx == 0.0 or ssy == 0.0 else cov / np.sqrt(ssx * ssy)
