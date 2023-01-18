from math import ceil
from typing import List, Optional, Tuple, Union

import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True)
def random_sample(*, x: np.ndarray, size: int, replace: bool = False) -> np.ndarray:
    """Generate a random sample.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    size : int
        Size of random sample.

    replace : bool, default=False
        Whether to sample with replacement.

    Returns
    -------
    np.ndarray
        Random sample.
    """
    return np.random.choice(x, size=size, replace=replace)


@njit(cache=True, fastmath=True, nogil=True)
def estimate_proba(*, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Estimate class probabilities.

    Note: This function assumes that for K classes, the labels are 0, 1, ..., K-1.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    n_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Estimated probabilities for each class.
    """
    return np.array([np.mean(y == j) for j in range(n_classes)])


@njit(cache=True, fastmath=True, nogil=True)
def estimate_mean(y: np.ndarray) -> float:
    """Estimate the mean.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    Returns
    -------
    float
        Estimate mean.
    """
    return np.mean(y)


def calculate_max_value(*, n_values: int, desired_max: Optional[Union[str, float, int]] = None) -> int:
    """ADD HERE.

    Parameters
    ----------

    Returns
    -------
    """
    if type(desired_max) is int:
        total = min(desired_max, n_values)
    elif desired_max == "sqrt":
        total = ceil(np.sqrt(n_values))
    elif desired_max == "log2":
        total = ceil(np.log2(n_values))
    elif type(desired_max) is float:
        total = ceil(n_values * desired_max)
    else:
        total = n_values
    return total


@njit(fastmath=True, nogil=True)
def split_data(
    *, X: np.ndarray, y: np.ndarray, feature: int, threshold: float
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


@njit(fastmath=True, nogil=True)
def bayesian_bootstrap_proba(n: int) -> np.ndarray:
    """Generate Bayesian bootstrap probabilities for a sample of size n.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Bootstrap probabilities associated with each sample.
    """
    p = np.random.exponential(scale=1.0, size=n)
    return p / p.sum()


def stratify_bootstrap_sample(
    *, idx_classes: List[np.ndarray], bayesian_bootstrap: bool, random_state: int
) -> List[np.ndarray]:
    """Indices for stratified bootstrap sampling in classification.

    Parameters
    ----------

    Returns
    -------
    """
    np.random.seed(random_state)

    idx = []
    for idx_class in idx_classes:
        n_class = len(idx_class)
        p = bayesian_bootstrap_proba(n_class) if bayesian_bootstrap else None
        idx.append(np.random.choice(idx_class, size=n_class, p=p, replace=True))

    return idx


def stratify_bootstrap_unsampled_idx(
    *, idx_classes: List[np.ndarray], bayesian_bootstrap: bool, random_state: int
) -> List[np.ndarray]:
    """Unsampled indices for stratified bootstrap sampling in classification.

    Parameters
    ----------

    Returns
    -------
    """
    np.random.seed(random_state)
    idx_sampled = stratify_bootstrap_sample(
        idx_classes=idx_classes, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
    )
    idx_unsampled = [np.setdiff1d(idx_class, idx_sampled[j]) for j, idx_class in enumerate(idx_classes)]

    return idx_unsampled


def balanced_bootstrap_sample(
    *, idx_classes: List[np.ndarray], n: int, bayesian_bootstrap: bool, random_state: int
) -> List[np.ndarray]:
    """Indices for balanced bootstrap sampling in classification.

    Parameters
    ----------

    Returns
    -------
    """
    np.random.seed(random_state)

    idx = []
    for idx_class in idx_classes:
        n_class = len(idx_class)
        p = bayesian_bootstrap_proba(n_class) if bayesian_bootstrap else None
        idx.append(np.random.choice(idx_class, size=n, p=p, replace=True))

    return idx


def balanced_bootstrap_unsampled_idx(
    *, idx_classes: List[np.ndarray], n: int, bayesian_bootstrap: bool, random_state: int
) -> List[np.ndarray]:
    """Unsampled indices for balanced bootstrap sampling in classification.

    Parameters
    ----------

    Returns
    -------
    """
    np.random.seed(random_state)
    idx_sampled = balanced_bootstrap_sample(
        idx_classes=idx_classes, n=n, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
    )
    idx_unsampled = [np.setdiff1d(idx_class, idx_sampled[j]) for j, idx_class in enumerate(idx_classes)]

    return idx_unsampled


def classic_bootstrap_sample(*, idx: np.ndarray, n: int, bayesian_bootstrap: bool, random_state: int) -> np.ndarray:
    """Indices for classic bootstrapping.

    Parameters
    ----------

    Returns
    -------
    """
    np.random.seed(random_state)

    p = bayesian_bootstrap_proba(n) if bayesian_bootstrap else None

    return np.random.choice(idx, size=n, p=p, replace=True)


def classic_bootstrap_unsampled_idx(
    *, idx: np.ndarray, n: int, bayesian_bootstrap: bool, random_state: int
) -> np.ndarray:
    """Unsampled indices for classic bootstrapping.

    Parameters
    ----------

    Returns
    -------
    """
    np.random.seed(random_state)

    idx_sampled = classic_bootstrap_sample(
        idx=idx, n=n, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
    )
    idx_unsampled = np.setdiff1d(idx, idx_sampled)

    return idx_unsampled
