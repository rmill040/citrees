from math import ceil
from typing import List, Optional, Tuple, Union

import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True)
def random_sample(*, x: np.ndarray, size: int, random_state: int, replace: bool = False) -> np.ndarray:
    """Generate a random sample.

    Parameters
    ----------
    x : np.ndarray
        Input data.

    size : int
        Size of random sample.

    random_state : int
        Random seed.

    replace : bool, default=False
        Whether to sample with replacement.

    Returns
    -------
    np.ndarray
        Random sample.
    """
    np.random.seed(random_state)

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
        Fetaures in left child node.

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


def stratified_bootstrap_sample(*, y: np.ndarray, max_samples: int, bayesian_bootstrap: bool, random_state: int) -> np.ndarray:
    """Indices for stratified bootstrap sampling in classification.

    Parameters
    ----------

    Returns
    -------
    """
    rng = np.random.RandomState(random_state)

    # n = len(y)
    # idx_classes = [np.where(y == j)[0] for j in np.unique(y)]
    # idx = []
    # for idx_class in idx_classes:
    #     n_class = len(idx_class)
    #     p = bayesian_bootstrap_proba(n=n_class, random_state=random_state) if bayesian_bootstrap else None
    #     idx.append(rng.choice(idx_class, size=n_class, p=p, replace=True))

    # return np.concatenate(idx)


# def stratified_bootstrap_unsampled_idx(
#     *, idx_classes: List[np.ndarray], bayesian_bootstrap: bool, random_state: int
# ) -> List[np.ndarray]:
#     """Unsampled indices for stratified bootstrap sampling in classification.

#     Parameters
#     ----------

#     Returns
#     -------
#     """
#     idx_sampled = stratified_bootstrap_sample(
#         idx_classes=idx_classes, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
#     )
#     idx_unsampled = [np.setdiff1d(idx_class, idx_sampled[j]) for j, idx_class in enumerate(idx_classes)]

#     return idx_unsampled


# def balanced_bootstrap_sample(
#     *, idx_classes: List[np.ndarray], bayesian_bootstrap: bool, random_state: int
# ) -> List[np.ndarray]:
#     """Indices for balanced bootstrap sampling in classification.

#     Parameters
#     ----------

#     Returns
#     -------
#     """
#     rng = np.random.RandomState(random_state)

#     idx = []
#     for idx_class in idx_classes:
#         n_class = len(idx_class)
#         p = bayesian_bootstrap_proba(n=n_class, random_state=random_state) if bayesian_bootstrap else None
#         idx.append(rng.choice(idx_class, size=n_class, p=p, replace=True))

#     return idx


# def balanced_bootstrap_unsampled_idx(
#     *, idx_classes: List[np.ndarray], bayesian_bootstrap: bool, random_state: int
# ) -> List[np.ndarray]:
#     """Unsampled indices for balanced bootstrap sampling in classification.

#     Parameters
#     ----------

#     Returns
#     -------
#     """
#     idx_sampled = balanced_bootstrap_sample(
#         idx_classes=idx_classes, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state
#     )
#     idx_unsampled = [np.setdiff1d(idx_class, idx_sampled[j]) for j, idx_class in enumerate(idx_classes)]

#     return idx_unsampled


# def classic_bootstrap_sample(*, idx: np.ndarray, bayesian_bootstrap: bool, random_state: int) -> np.ndarray:
#     """Indices for classic bootstrapping.

#     Parameters
#     ----------

#     Returns
#     -------
#     """
#     rng = np.random.RandomState(random_state)

#     n = len(idx)
#     p = bayesian_bootstrap_proba(n=n, random_state=random_state) if bayesian_bootstrap else None

#     return rng.choice(idx, size=n, p=p, replace=True)


# def classic_bootstrap_unsampled_idx(*, idx: np.ndarray, bayesian_bootstrap: bool, random_state: int) -> np.ndarray:
#     """Unsampled indices for classic bootstrapping.

#     Parameters
#     ----------

#     Returns
#     -------
#     """
#     idx_sampled = classic_bootstrap_sample(idx=idx, bayesian_bootstrap=bayesian_bootstrap, random_state=random_state)
#     idx_unsampled = np.setdiff1d(idx, idx_sampled)

#     return idx_unsampled
