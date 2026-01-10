from math import ceil
from typing import Any, Optional

import numpy as np
from dcor import distance_correlation as _d_correlation
from dcor import distance_covariance as _d_covariance
from numba import njit, prange
from sklearn.feature_selection import mutual_info_classif

from citrees._registry import ClassifierSelectors, ClassifierSelectorTests, RegressorSelectors, RegressorSelectorTests

# Threshold for using parallel permutation tests
_PARALLEL_THRESHOLD = 200


def _permutation_test(
    *,
    func: Any,
    func_arg: Any,
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Calculate the achieved significance level using a permutation test.

    Parameters
    ----------
    func : Any
        Feature selection function to use in permutation testing.

    func_arg : Any
        Single function argument.

    x : np.ndarray
        Input data, usually the feature in the (x, y) pair.

    y : np.ndarray
        Input data, usually the target in the (x, y) pair.

    n_resamples : int
        Number of resamples.

    early_stopping : bool
        Whether to early stop permutation testing if null hypothesis can be rejected.

    alpha : float
        Alpha level for significance testing.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated achieved significance level.
    """
    np.random.seed(random_state)

    theta = np.abs(func(x, y, func_arg, random_state=random_state))
    y_ = y.copy()
    theta_p = np.empty(n_resamples)

    if early_stopping:
        # Handle cases where n_resamples is less than min_resamples and early stopping is not possible
        min_resamples = ceil(1 / alpha)
        if n_resamples < min_resamples:
            n_resamples = min_resamples
            theta_p = np.empty(n_resamples)
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p[i] = func(x, y_, func_arg, random_state=random_state)
            if i >= min_resamples - 1:
                asl = np.mean(np.abs(theta_p[: i + 1]) >= theta)
                if asl < alpha:
                    break

    else:
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p[i] = func(x, y_, func_arg, random_state=random_state)
        asl = np.mean(np.abs(theta_p) >= theta)

    return asl


# Compiled version of permutation test
_permutation_test_compiled = njit(cache=True, fastmath=True, nogil=True)(_permutation_test)


# Parallel permutation test for multiple correlation (classifier)
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _permutation_test_mc_parallel(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    random_state: int,
) -> float:
    """Parallel permutation test for multiple correlation."""
    # Compute observed statistic
    mu = x.mean()
    sst = np.sum((x - mu) ** 2)
    if sst == 0:
        return 1.0
    ssb = 0.0
    for j in range(n_classes):
        x_j = x[y == j]
        n_j = len(x_j)
        if n_j > 0:
            mu_j = x_j.mean()
            ssb += n_j * (mu_j - mu) ** 2
    theta = np.sqrt(ssb / sst)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        ssb_perm = 0.0
        for j in range(n_classes):
            x_j = x[y_perm == j]
            n_j = len(x_j)
            if n_j > 0:
                mu_j = x_j.mean()
                ssb_perm += n_j * (mu_j - mu) ** 2
        theta_p[i] = np.sqrt(ssb_perm / sst)

    return np.mean(np.abs(theta_p) >= theta)


# Parallel permutation test for pearson correlation (regressor)
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _permutation_test_pc_parallel(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    random_state: int,
) -> float:
    """Parallel permutation test for pearson correlation."""
    n = len(x)
    sx = x.sum()
    sx2 = np.sum(x * x)
    sy = y.sum()
    sy2 = np.sum(y * y)
    sxy = np.sum(x * y)

    cov = n * sxy - sx * sy
    ssx = n * sx2 - sx * sx
    ssy = n * sy2 - sy * sy
    denom = np.sqrt(ssx * ssy)
    if denom == 0:
        return 1.0
    theta = np.abs(cov / denom)

    # Parallel permutation
    theta_p = np.empty(n_resamples)
    for i in prange(n_resamples):
        np.random.seed(random_state + i)
        y_perm = y.copy()
        np.random.shuffle(y_perm)

        sy_perm = y_perm.sum()
        sy2_perm = np.sum(y_perm * y_perm)
        sxy_perm = np.sum(x * y_perm)

        cov_perm = n * sxy_perm - sx * sy_perm
        ssy_perm = n * sy2_perm - sy_perm * sy_perm
        denom_perm = np.sqrt(ssx * ssy_perm)
        if denom_perm == 0:
            theta_p[i] = 0.0
        else:
            theta_p[i] = np.abs(cov_perm / denom_perm)

    return np.mean(theta_p >= theta)


@ClassifierSelectors.register("mc")
@njit(cache=True, nogil=True, fastmath=True)
def multiple_correlation(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: Optional[int] = None) -> float:
    """Calculate the multiple correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature.

    y : np.ndarray
        Target.

    n_classes : int
        Number of classes.

    random_state : int, default=None
        Random seed. Kept for API compatibility.

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
    for j in range(n_classes):
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

    try:
        return np.sqrt(ssb / sst)
    except Exception:
        return 0.0


@ClassifierSelectors.register("mi")
def mutual_information(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int) -> float:
    """Calculate the mutual information.

    Parameters
    ----------
    x : np.ndarray
        Feature.

    y : np.ndarray
        Target.

    n_classes : int
        Number of classes. Kept for API compatibility.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated mutual information.
    """
    if x.ndim == 1:
        x = x[:, None]

    return mutual_info_classif(x, y, random_state=random_state)[0]


@ClassifierSelectors.register("hybrid")
def hybrid_classifier(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int) -> float:
    """Calculate the multiple correlation and mutual information and return the higher metric.

    Parameters
    ----------
    x : np.ndarray
        Feature.

    y : np.ndarray
        Target.

    n_classes : int
        Number of classes.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated metric.
    """
    mc = multiple_correlation(x=x, y=y, n_classes=n_classes, random_state=random_state)
    mi = mutual_information(x=x, y=y, n_classes=n_classes, random_state=random_state)

    return max(mc, mi)


@RegressorSelectors.register("pc")
@njit(cache=True, nogil=True, fastmath=True)
def pearson_correlation(x: np.ndarray, y: np.ndarray, standardize: bool, random_state: Optional[int] = None) -> float:
    """Calculate the Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.

    random_state : int, default=None
        Random seed. Kept for API compatibility.

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


@njit(cache=True, nogil=True, fastmath=True)
def _covariance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the covariance.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

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


@njit(cache=True, nogil=True, fastmath=True)
def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

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

    try:
        return cov / np.sqrt(ssx * ssy)
    except Exception:
        return 0.0


@RegressorSelectors.register("dc")
def distance_correlation(x: np.ndarray, y: np.ndarray, standardize: bool, random_state: Optional[int] = None) -> float:
    """Calculate the distance correlation.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.

    random_state : int, default=None
        Random seed. Kept for API compatibility.

    Returns
    -------
    float
        Estimated distance correlation.
    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    return _d_correlation(x, y) if standardize else _d_covariance(x, y)


@RegressorSelectors.register("hybrid")
def hybrid_regressor(x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int) -> float:
    """Calculate the Pearson correlation and distance correlation and return the higher metric.

    Parameters
    ----------
    x : np.ndarray
        Feature.

    y : np.ndarray
        Target.

    standardize : bool
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated metric.
    """
    pc = np.abs(pearson_correlation(x=x, y=y, standardize=standardize, random_state=random_state))
    dc = distance_correlation(x=x, y=y, standardize=standardize, random_state=random_state)

    return max(pc, dc)


@ClassifierSelectorTests.register("mc")
def permutation_test_multiple_correlation(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Perform a permutation test using the multiple correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    n_classes : int
        Number of classes.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : bool
        Whether to implement early stopping during permutation testing.

    alpha : float
        Threshold used to compare the estimated achieved significance level to and early stop permutation testing.
        This parameter is only used when early_stopping is True.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated achieved significance level.
    """
    # Use parallel version when not using early stopping and enough resamples
    if not early_stopping and n_resamples >= _PARALLEL_THRESHOLD:
        return _permutation_test_mc_parallel(
            x=x,
            y=y,
            n_classes=n_classes,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return _permutation_test_compiled(
        func=multiple_correlation,
        func_arg=n_classes,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@ClassifierSelectorTests.register("mi")
def permutation_test_mutual_information(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Perform a permutation test using the mutual information.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    n_classes : int
        Number of classes. Kept for API compatibility.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : bool
        Whether to implement early stopping during permutation testing.

    alpha : float
        Threshold used to compare the estimated achieved significance level to and early stop permutation testing.
        This parameter is only used when early_stopping is True.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated achieved significance level.
    """
    return _permutation_test(
        func=mutual_information,
        func_arg=n_classes,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@ClassifierSelectorTests.register("hybrid")
def permutation_test_hybrid_classifier(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Perform a permutation test using the larger of the multiple correlation or mutual information.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    n_classes : int
        Number of classes.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : bool
        Whether to implement early stopping during permutation testing.

    alpha : float
        Threshold used to compare the estimated achieved significance level to and early stop permutation testing.
        This parameter is only used when early_stopping is True.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated achieved significance level.
    """
    mc = multiple_correlation(x, y, n_classes, random_state)
    mi = mutual_information(x, y, n_classes, random_state)

    if mc >= mi:
        asl = permutation_test_multiple_correlation(
            x=x,
            y=y,
            n_classes=n_classes,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
        )
    else:
        asl = permutation_test_mutual_information(
            x=x,
            y=y,
            n_classes=n_classes,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
        )
    return asl


@RegressorSelectorTests.register("pc")
def permutation_test_pearson_correlation(
    *,
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state,
) -> float:
    """Perform a permutation test using the Pearson correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : bool
        Whether to implement early stopping during permutation testing.

    alpha : float
        Threshold used to compare the estimated achieved significance level to and early stop permutation testing.
        This parameter is only used when early_stopping is True.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated achieved significance level.
    """
    # Use parallel version when not using early stopping and enough resamples
    if not early_stopping and n_resamples >= _PARALLEL_THRESHOLD:
        return _permutation_test_pc_parallel(
            x=x,
            y=y,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return _permutation_test_compiled(
        func=pearson_correlation,
        func_arg=standardize,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@RegressorSelectorTests.register("dc")
def permutation_test_distance_correlation(
    *,
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Perform a permutation test using the distsance correlation coefficient.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : bool
        Whether to implement early stopping during permutation testing.

    alpha : float
        Threshold used to compare the estimated achieved significance level to and early stop permutation testing.
        This parameter is only used when early_stopping is True.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated achieved significance level.
    """
    return _permutation_test(
        func=distance_correlation,
        func_arg=standardize,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@RegressorSelectorTests.register("hybrid")
def permutation_test_hybrid_regressor(
    *,
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Perform a permutation test using the larger of the Pearson correlation and distance correlation.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.

    n_resamples : int
        Number of permutations to perform.

    early_stopping : bool
        Whether to implement early stopping during permutation testing.

    alpha : float
        Threshold used to compare the estimated achieved significance level to and early stop permutation testing.
        This parameter is only used when early_stopping is True.

    random_state : int
        Random seed.

    Returns
    -------
    float
        Estimated achieved significance level.
    """
    pc = np.abs(pearson_correlation(x, y, standardize=True, random_state=random_state))
    dc = distance_correlation(x, y, standardize=True, random_state=random_state)

    if pc >= dc:
        asl = permutation_test_pearson_correlation(
            x=x,
            y=y,
            standardize=standardize,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
        )
    else:
        asl = permutation_test_distance_correlation(
            x=x,
            y=y,
            standardize=standardize,
            n_resamples=n_resamples,
            early_stopping=early_stopping,
            alpha=alpha,
            random_state=random_state,
        )

    return asl
