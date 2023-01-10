from math import ceil
from typing import Tuple, Union

import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True)
def bayesian_boostrap_proba(n: int, random_state: int):
    """ADD HERE.
    
    Parameters
    ----------
    
    Returns
    -------
    """
    np.random.seed(random_state)

    p = np.random.exponential(scale=1.0, size=n)
    return p / p.sum()


@njit(fastmath=True, nogil=True)
def random_sample(*, x: np.ndarray, size: int, replace: bool = False) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    return np.random.choice(x, size=size, replace=replace)


@njit(cache=True, fastmath=True, nogil=True)
def estimate_proba(*, y: np.ndarray, n_classes: np.ndarray) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    return np.array([np.mean(y == j) for j in range(n_classes)])


@njit(cache=True, fastmath=True, nogil=True)
def estimate_mean(y: np.ndarray) -> np.ndarray:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    return np.mean(y)


@njit(cache=True, fastmath=True, nogil=True)
def calculate_max_value(*, n_values: int, desired_max: Union[str, float, int]) -> int:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if desired_max == "sqrt":
        total = ceil(np.sqrt(n_values))
    elif desired_max == "log":
        total = ceil(np.log(n_values))
    elif type(desired_max) is float:
        total = ceil(n_values * desired_max)
    elif type(desired_max) is int:
        total = min(desired_max, n_values)
    return total


@njit(fastmath=True, nogil=True)
def split_data(
    *,
    X: np.ndarray, y: np.ndarray, feature: int, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data based on feature and threshold.

    Parameters
    ----------
    X : np.ndarray
        Features.

    y : np.ndarray:
        Labels.

    feature : int
        Index of feature to use for splitting.

    threshold : float
        Threshold value to use for creating binary split.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Features and labels in left and right splits with order (X_left, y_left, X_right, y_right).
    """
    idx = X[:, feature] <= threshold
    return X[idx], y[idx], X[~idx], y[~idx]
