from math import ceil
from typing import Optional, Tuple, Union

import numpy as np
from numba import njit


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
    """Calculate the maximum desired value based on a fixed input size.

    Parameters
    ----------
    n_values : int
        Total number of values.

    desired_max : Union[str, float, int], default=None
        Desired number of values.

    Returns
    -------
    int
        Maximum value.
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

    return min(n_values, total)


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
    X_left : np.ndarray
        Features in left child node.

    y_left : np.ndarray
        Target in left child node.

    X_right : np.ndarray
        Features in right child node.

    y_right : np.ndarray
        Target in right child node.
    """
    idx = X[:, feature] <= threshold
    return X[idx], y[idx], X[~idx], y[~idx]


@njit(fastmath=True, nogil=True)
def bayesian_bootstrap_proba(*, n: int, random_state: int) -> np.ndarray:
    """Generate Bayesian bootstrap probabilities for a sample of size n.

    Parameters
    ----------
    n : int
        Number of samples.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Bootstrap probabilities associated with each sample.
    """
    np.random.seed(random_state)

    p = np.random.exponential(scale=1.0, size=n)
    return p / p.sum()


def stratified_bootstrap_sample(
    *, y: np.ndarray, max_samples: int, bayesian_bootstrap: bool, random_state: int
) -> np.ndarray:
    """Indices for stratified bootstrap sampling in classification.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    bayesian_bootstrap : bool
        Whether to use Bayesian bootstrap.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    prng = np.random.RandomState(random_state)

    n = len(y)
    n_classes = len(np.unique(y))
    idx_classes = [np.where(y == j)[0] for j in range(n_classes)]
    idx = []
    for idx_class in idx_classes:
        n_class = len(idx_class)
        p = bayesian_bootstrap_proba(n=n_class, random_state=random_state) if bayesian_bootstrap else None
        idx.append(prng.choice(idx_class, size=n_class, p=p, replace=True))

    # Subsample if needed
    if max_samples < n:
        for j in range(n_classes):
            n_class = len(idx[j])
            ratio = n_class / n
            idx[j] = prng.choice(idx[j], size=round(ratio * max_samples), replace=False)

    return np.concatenate(idx)


def stratified_bootstrap_unsampled_idx(
    *, y: np.ndarray, max_samples: int, bayesian_bootstrap: bool, random_state: int
) -> np.ndarray:
    """Unsampled indices for stratified bootstrap sampling in classification.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    bayesian_bootstrap : bool
        Whether to use Bayesian bootstrap.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    idx_sampled = stratified_bootstrap_sample(
        y=y, max_samples=max_samples, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
    )
    idx_all = np.arange(len(y), dtype=int)
    idx_unsampled = np.setdiff1d(idx_all, idx_sampled)

    return idx_unsampled


def balanced_bootstrap_sample(
    *, y: np.ndarray, max_samples: int, bayesian_bootstrap: bool, random_state: int
) -> np.ndarray:
    """Indices for balanced bootstrap sampling in classification.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    bayesian_bootstrap : bool
        Whether to use Bayesian bootstrap.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    prng = np.random.RandomState(random_state)

    n = len(y)
    n_classes = len(np.unique(y))
    idx_classes = [np.where(y == j)[0] for j in range(n_classes)]
    idx = []
    n_per_class = np.bincount(y).min()
    for idx_class in idx_classes:
        p = bayesian_bootstrap_proba(n=len(idx_class), random_state=random_state) if bayesian_bootstrap else None
        idx.append(prng.choice(idx_class, size=n_per_class, p=p, replace=True))

    # Subsample if needed
    if max_samples < n:
        ratio = n_per_class / n
        for j in range(n_classes):
            idx[j] = prng.choice(idx[j], size=round(ratio * max_samples), replace=False)

    return np.concatenate(idx)


def balanced_bootstrap_unsampled_idx(
    *, y: np.ndarray, max_samples: int, bayesian_bootstrap: bool, random_state: int
) -> np.ndarray:
    """Unsampled indices for balanced bootstrap sampling in classification.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    bayesian_bootstrap : bool
        Whether to use Bayesian bootstrap.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    idx_sampled = balanced_bootstrap_sample(
        y=y, max_samples=max_samples, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
    )
    idx_all = np.arange(len(y), dtype=int)
    idx_unsampled = np.setdiff1d(idx_all, idx_sampled)

    return idx_unsampled


def classic_bootstrap_sample(
    *, y: np.ndarray, max_samples: int, bayesian_bootstrap: bool, random_state: int
) -> np.ndarray:
    """Indices for classic bootstrapping.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    bayesian_bootstrap : bool
        Whether to use Bayesian bootstrap.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    prng = np.random.RandomState(random_state)

    n = len(y)
    p = bayesian_bootstrap_proba(n=n, random_state=random_state) if bayesian_bootstrap else None
    idx = prng.choice(range(n), size=n, p=p, replace=True)

    if max_samples < n:
        idx = prng.choice(idx, size=max_samples, replace=False)
    return idx


def classic_bootstrap_unsampled_idx(
    *, y: np.ndarray, max_samples: int, bayesian_bootstrap: bool, random_state: int
) -> np.ndarray:
    """Unsampled indices for classic bootstrapping.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    bayesian_bootstrap : bool
        Whether to use Bayesian bootstrap.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    idx_sampled = classic_bootstrap_sample(
        y=y, max_samples=max_samples, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
    )
    idx_all = np.arange(len(y), dtype=int)
    idx_unsampled = np.setdiff1d(idx_all, idx_sampled)

    return idx_unsampled
