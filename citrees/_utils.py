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
        total = min(int(desired_max), n_values)  # type: ignore[arg-type]
    elif desired_max == MaxValuesMethod.SQRT:
        total = ceil(np.sqrt(n_values))
    elif desired_max == MaxValuesMethod.LOG2:
        total = ceil(np.log2(n_values))
    elif isinstance(desired_max, float | np.floating):
        total = ceil(n_values * desired_max)
    else:
        total = n_values

    return min(n_values, total)  # type: ignore[return-value]


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


def stratified_bootstrap_sample(
    *, y: np.ndarray, max_samples: int, random_state: int
) -> np.ndarray:
    """Indices for stratified bootstrap sampling in classification.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

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
        idx.append(prng.choice(idx_class, size=n_class, replace=True))

    # Subsample if needed (use proper integer allocation to guarantee sum = max_samples)
    if max_samples < n:
        class_sizes = np.array([len(idx[j]) for j in range(n_classes)])
        allocation = _allocate_samples(class_sizes, max_samples)
        for j in range(n_classes):
            if allocation[j] < len(idx[j]):
                idx[j] = prng.choice(idx[j], size=allocation[j], replace=False)

    return np.concatenate(idx)


def stratified_bootstrap_unsampled_idx(
    *, y: np.ndarray, max_samples: int, random_state: int
) -> np.ndarray:
    """Unsampled indices for stratified bootstrap sampling in classification.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

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
        random_state=random_state,
    )
    idx_all = np.arange(len(y), dtype=int)
    idx_unsampled = np.setdiff1d(idx_all, idx_sampled)

    return idx_unsampled


def undersample_bootstrap_sample(
    *, y: np.ndarray, max_samples: int, random_state: int
) -> np.ndarray:
    """Indices for class-balanced undersampling (bootstrap with per-class cap).

    This method draws ``n_min`` samples *per class* with replacement, where
    ``n_min`` is the minority class count in ``y``. The resulting bootstrap size
    is ``K * n_min`` where ``K`` is the number of classes (and is therefore
    <= ``len(y)``). If ``max_samples < K * n_min``, the sample is truncated to
    exactly ``max_samples`` with class counts as equal as possible.

    Parameters
    ----------
    y : np.ndarray
        Input labels (assumed encoded as 0..K-1).

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    prng = np.random.RandomState(random_state)

    n_classes = len(np.unique(y))
    idx_classes = [np.where(y == j)[0] for j in range(n_classes)]
    class_counts = np.array([len(idx_class) for idx_class in idx_classes], dtype=int)
    n_min = int(class_counts.min())

    idx = []
    for j, idx_class in enumerate(idx_classes):
        idx.append(prng.choice(idx_class, size=n_min, replace=True))

    total = n_classes * n_min
    if max_samples < total:
        base = max_samples // n_classes
        remainder = max_samples - base * n_classes
        allocation = np.full(n_classes, base, dtype=int)
        if remainder:
            allocation[prng.choice(np.arange(n_classes), size=remainder, replace=False)] += 1
        for j in range(n_classes):
            if allocation[j] < len(idx[j]):
                idx[j] = prng.choice(idx[j], size=allocation[j], replace=False)

    return np.concatenate(idx)


def undersample_bootstrap_unsampled_idx(
    *, y: np.ndarray, max_samples: int, random_state: int
) -> np.ndarray:
    """Unsampled indices for class-balanced undersampling."""
    idx_sampled = undersample_bootstrap_sample(
        y=y,
        max_samples=max_samples,
        random_state=random_state,
    )
    idx_all = np.arange(len(y), dtype=int)
    idx_unsampled = np.setdiff1d(idx_all, idx_sampled)

    return idx_unsampled


def oversample_bootstrap_sample(
    *, y: np.ndarray, max_samples: int, random_state: int
) -> np.ndarray:
    """Indices for class-balanced oversampling (fixed-size bootstrap).

    This method produces a bootstrap sample of size ``max_samples`` whose class
    counts are as equal as possible (difference at most 1), sampling with
    replacement within each class. This can oversample minority classes relative
    to their original frequency while keeping the total sample size fixed.

    Parameters
    ----------
    y : np.ndarray
        Input labels (assumed encoded as 0..K-1).

    max_samples : int
        Total number of samples in the bootstrap sample.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    prng = np.random.RandomState(random_state)

    n_classes = len(np.unique(y))
    idx_classes = [np.where(y == j)[0] for j in range(n_classes)]

    base = max_samples // n_classes
    remainder = max_samples - base * n_classes
    allocation = np.full(n_classes, base, dtype=int)
    if remainder:
        allocation[prng.choice(np.arange(n_classes), size=remainder, replace=False)] += 1

    idx = []
    for j, idx_class in enumerate(idx_classes):
        if allocation[j] == 0:
            continue
        idx.append(prng.choice(idx_class, size=int(allocation[j]), replace=True))

    return np.concatenate(idx) if idx else np.empty(0, dtype=int)


def oversample_bootstrap_unsampled_idx(
    *, y: np.ndarray, max_samples: int, random_state: int
) -> np.ndarray:
    """Unsampled indices for class-balanced oversampling."""
    idx_sampled = oversample_bootstrap_sample(
        y=y,
        max_samples=max_samples,
        random_state=random_state,
    )
    idx_all = np.arange(len(y), dtype=int)
    idx_unsampled = np.setdiff1d(idx_all, idx_sampled)

    return idx_unsampled


def classic_bootstrap_sample(
    *, y: np.ndarray, max_samples: int, random_state: int
) -> np.ndarray:
    """Indices for classic bootstrapping.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

    random_state : int
        Random seed.

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample.
    """
    prng = np.random.RandomState(random_state)

    n = len(y)
    idx = prng.choice(range(n), size=n, replace=True)

    if max_samples < n:
        idx = prng.choice(idx, size=max_samples, replace=False)
    return idx


def classic_bootstrap_unsampled_idx(
    *, y: np.ndarray, max_samples: int, random_state: int
) -> np.ndarray:
    """Unsampled indices for classic bootstrapping.

    Parameters
    ----------
    y : np.ndarray
        Input data.

    max_samples : int
        Maximum number of samples in a bootstrap sample.

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
        random_state=random_state,
    )
    idx_all = np.arange(len(y), dtype=int)
    idx_unsampled = np.setdiff1d(idx_all, idx_sampled)

    return idx_unsampled
