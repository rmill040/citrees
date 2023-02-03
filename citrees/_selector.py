from math import ceil
from typing import Any

import numpy as np
from dcor import distance_correlation as _d_correlation
from dcor import distance_covariance as _d_covariance
from numba import njit
from sklearn.feature_selection import mutual_info_classif

from ._registry import ClassifierSelectors, ClassifierSelectorTests, RegressorSelectors, RegressorSelectorTests


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

    theta = func(x, y, func_arg, random_state=random_state)
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
                asl = np.mean(theta_p[: i + 1] >= theta)
                if asl < alpha:
                    break

    else:
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p[i] = func(x, y_, func_arg, random_state=random_state)
        asl = np.mean(theta_p >= theta)

    return asl


# Compiled version of permutation test
_permutation_test_compiled = njit(fastmath=True, nogil=True)(_permutation_test)


@ClassifierSelectors.register("mc")
@njit(nogil=True, fastmath=True)
def multiple_correlation(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int) -> float:
    """Calculate the multiple correlation coefficient between x and y.

    Parameters
    ----------
    x : np.ndarray
        Feature.

    y : np.ndarray
        Target.

    n_classes : int
        Number of classes.
        
    random_state : int
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

    # Early stop here if denominator will be a divide by 0 error
    if sst == 0.0:
        return 0.0

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

    return np.sqrt(ssb / sst)


@ClassifierSelectors.register("mi")
def mutual_information(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int) -> float:
    """Calculate the mutual information between x and y.

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
    """ADD HERE.
    
    Parameters
    ----------
    
    Returns
    -------
    """
    mc = multiple_correlation(x=x, y=y, n_classes=n_classes, random_state=random_state)
    mi = mutual_information(x=x, y=y, n_classes=n_classes, random_state=random_state)

    return max(mc, mi)


@RegressorSelectors.register("pc")
@njit(nogil=True, fastmath=True)
def pearson_correlation(x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int) -> float:
    """Calculate the Pearson correlation coefficient between x and y.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : np.ndarray
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.
        
    random_state : int
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


@njit(nogil=True, fastmath=True)
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


@njit(nogil=True, fastmath=True)
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

    return 0.0 if ssx == 0.0 or ssy == 0.0 else cov / np.sqrt(ssx * ssy)


@RegressorSelectors.register("dc")
def distance_correlation(x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int) -> float:
    """Calculate the distance correlation.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : np.ndarray
        Whether to standardize the result. If True, return the correlation, if False, return the covariance.
        
    random_state : int
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
    """Perform a permutation test using either the multiple correlation coefficient or mutual information whichever is
    larger on the original data.

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

    standardize : np.ndarray
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

    standardize : np.ndarray
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
    """Perform a permutation test using either the Pearson correlation or distance correlation whichever is larger on
    the original data.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : np.ndarray
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
    pc = abs(pearson_correlation(x, y, standardize=True))
    dc = distance_correlation(x, y, standardize=True)

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
