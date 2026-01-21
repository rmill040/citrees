from math import ceil

import numpy as np
from numba import njit

from citrees._types import MaxValuesMethod


def _allocate_samples(weights: np.ndarray, total: int) -> np.ndarray:
    """Allocate integer samples proportionally to weights, guaranteeing sum equals total.

    Uses the largest remainder method (Hamilton method) for fair apportionment.

    Parameters
    ----------
    weights : np.ndarray
        Non-negative weights for each class (e.g., class sizes).

    total : int
        Total number of samples to allocate.

    Returns
    -------
    np.ndarray
        Integer allocation for each class, summing to total.
    """
    if total <= 0 or len(weights) == 0:
        return np.zeros(len(weights), dtype=int)

    weights = np.asarray(weights, dtype=float)
    weight_sum = weights.sum()
    if weight_sum == 0:
        # Equal allocation if all weights are zero
        base = total // len(weights)
        allocation = np.full(len(weights), base, dtype=int)
        remainder = total - allocation.sum()
        allocation[:remainder] += 1
        return allocation

    # Calculate ideal (fractional) allocation
    ideal = weights * total / weight_sum

    # Floor allocation
    allocation = np.floor(ideal).astype(int)

    # Distribute remainder to classes with largest fractional parts
    remainder = total - allocation.sum()
    if remainder > 0:
        fractional_parts = ideal - allocation
        # Get indices sorted by fractional part (descending)
        indices = np.argsort(-fractional_parts)
        allocation[indices[:remainder]] += 1

    return allocation


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


def calculate_max_value(*, n_values: int, desired_max: str | float | int | None = None) -> int:
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
    if type(desired_max) is int or np.issubdtype(type(desired_max), np.integer):
        total = min(desired_max, n_values)
    elif desired_max == MaxValuesMethod.SQRT:
        total = ceil(np.sqrt(n_values))
    elif desired_max == MaxValuesMethod.LOG2:
        total = ceil(np.log2(n_values))
    elif type(desired_max) is float:
        total = ceil(n_values * desired_max)
    else:
        total = n_values

    return min(n_values, total)


@njit(cache=True, fastmath=True, nogil=True)
def split_data(
    *, X: np.ndarray, y: np.ndarray, feature: int, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data based on feature and threshold.

    Note: Input X must not contain NaN values. NaN validation is performed upstream
    in _validate_data_fit(). With fastmath=True, NaN comparisons have undefined behavior.

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


# Note: Uses np.random.seed() because Numba doesn't support default_rng() inside @njit.
@njit(cache=True, fastmath=True, nogil=True)
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
    for j, idx_class in enumerate(idx_classes):
        n_class = len(idx_class)
        # Vary seed per class to ensure independent bootstrap probabilities
        p = (
            bayesian_bootstrap_proba(n=n_class, random_state=random_state + j)
            if bayesian_bootstrap
            else None
        )
        idx.append(prng.choice(idx_class, size=n_class, p=p, replace=True))

    # Subsample if needed (use proper integer allocation to guarantee sum = max_samples)
    if max_samples < n:
        class_sizes = np.array([len(idx[j]) for j in range(n_classes)])
        allocation = _allocate_samples(class_sizes, max_samples)
        for j in range(n_classes):
            if allocation[j] < len(idx[j]):
                idx[j] = prng.choice(idx[j], size=allocation[j], replace=False)

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
        y=y,
        max_samples=max_samples,
        bayesian_bootstrap=bayesian_bootstrap,
        random_state=random_state,
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
    for j, idx_class in enumerate(idx_classes):
        # Vary seed per class to ensure independent bootstrap probabilities
        p = (
            bayesian_bootstrap_proba(n=len(idx_class), random_state=random_state + j)
            if bayesian_bootstrap
            else None
        )
        idx.append(prng.choice(idx_class, size=n_per_class, p=p, replace=True))

    # Subsample if needed (use proper integer allocation to guarantee sum = max_samples)
    if max_samples < n:
        # For balanced bootstrap, all classes have equal weight
        class_sizes = np.array([len(idx[j]) for j in range(n_classes)])
        allocation = _allocate_samples(class_sizes, max_samples)
        for j in range(n_classes):
            if allocation[j] < len(idx[j]):
                idx[j] = prng.choice(idx[j], size=allocation[j], replace=False)

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
        y=y,
        max_samples=max_samples,
        bayesian_bootstrap=bayesian_bootstrap,
        random_state=random_state,
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
        y=y,
        max_samples=max_samples,
        bayesian_bootstrap=bayesian_bootstrap,
        random_state=random_state,
    )
    idx_all = np.arange(len(y), dtype=int)
    idx_unsampled = np.setdiff1d(idx_all, idx_sampled)

    return idx_unsampled
