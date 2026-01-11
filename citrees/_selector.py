from math import ceil
from typing import Any

import numpy as np
from dcor import distance_correlation as _d_correlation
from dcor import distance_covariance as _d_covariance
from numba import njit, prange
from sklearn.feature_selection import mutual_info_classif

from citrees._registry import (
    ClassifierSelectors,
    ClassifierSelectorTests,
    RegressorSelectors,
    RegressorSelectorTests,
)

# Threshold for using parallel permutation tests
_PARALLEL_THRESHOLD = 200

# P-value correction: Phipson & Smyth (2010). "Permutation P-values Should Never Be Zero."
# SAGMB 9(1):39. https://pubmed.ncbi.nlm.nih.gov/21044043/
# Use p = (b+1)/(m+1) instead of p = b/m to avoid p=0.
# Note: min_resamples = ceil(1/alpha) remains valid since 1/(m+1) < alpha.


def _ptest(
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
                # +1 correction (Phipson & Smyth 2010)
                asl = (1 + np.sum(np.abs(theta_p[: i + 1]) >= theta)) / (2 + i)
                if asl < alpha:
                    break

    else:
        for i in range(n_resamples):
            np.random.shuffle(y_)
            theta_p[i] = func(x, y_, func_arg, random_state=random_state)
        # +1 correction (Phipson & Smyth 2010)
        asl = (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)

    return asl


# Compiled version of permutation test
_ptest_compiled = njit(cache=True, fastmath=True, nogil=True)(_ptest)


# Parallel permutation test for multiple correlation (classifier)
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_mc_parallel(
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

    # +1 correction (Phipson & Smyth 2010)
    return (1 + np.sum(np.abs(theta_p) >= theta)) / (1 + n_resamples)


# Parallel permutation test for pearson correlation (regressor)
@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _ptest_pc_parallel(
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

    # +1 correction (Phipson & Smyth 2010)
    return (1 + np.sum(theta_p >= theta)) / (1 + n_resamples)


@ClassifierSelectors.register("mc")
@njit(cache=True, nogil=True, fastmath=True)
def mc(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int | None = None) -> float:
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
def mi(x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int) -> float:
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


@RegressorSelectors.register("pc")
@njit(cache=True, nogil=True, fastmath=True)
def pc(
    x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int | None = None
) -> float:
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
def dc(
    x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int | None = None
) -> float:
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


# =============================================================================
# Randomized Dependence Coefficient (RDC)
# Lopez-Paz et al. (2013) - https://arxiv.org/abs/1304.7717
# O(n log n) non-linear dependence measure (vs O(n²) for distance correlation)
#
# R reference implementation from paper:
#   rdc <- function(x,y,k,s) {
#     x <- cbind(apply(as.matrix(x),2,function(u) ecdf(u)(u)),1)
#     y <- cbind(apply(as.matrix(y),2,function(u) ecdf(u)(u)),1)
#     wx <- matrix(rnorm(ncol(x)*k,0,s),ncol(x),k)
#     wy <- matrix(rnorm(ncol(y)*k,0,s),ncol(y),k)
#     cancor(cbind(cos(x%*%wx),sin(x%*%wx)), cbind(cos(y%*%wy),sin(y%*%wy)))$cor[1]
#   }
# =============================================================================

_RDC_K = 10  # Number of random projections (paper uses 20, we use 10 for speed)
_RDC_S = 1.0 / 6.0  # Bandwidth parameter


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_ecdf(x: np.ndarray) -> np.ndarray:
    """Empirical CDF transform: ecdf(x)(x) = rank(x) / n."""
    n = len(x)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    for i in range(n):
        ranks[order[i]] = (i + 1) / n
    return ranks


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_features(x: np.ndarray, k: int, s: float, seed: int) -> np.ndarray:
    """Create RDC features: [cos(X @ w), sin(X @ w)] where X = [ecdf(x), 1]."""
    n = len(x)
    np.random.seed(seed)

    # X = [ecdf(x), 1] has shape (n, 2)
    ecdf_x = _rdc_ecdf(x)

    # w has shape (2, k): random weights for [ecdf, bias]
    w0 = np.empty(k, dtype=np.float64)  # weights for ecdf
    w1 = np.empty(k, dtype=np.float64)  # weights for bias
    for j in range(k):
        w0[j] = np.random.randn() * s
        w1[j] = np.random.randn() * s

    # Compute X @ w = ecdf * w0 + 1 * w1, then [cos, sin]
    features = np.empty((n, 2 * k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            proj = ecdf_x[i] * w0[j] + w1[j]
            features[i, j] = np.cos(proj)
            features[i, k + j] = np.sin(proj)

    return features


@njit(cache=True, nogil=True, fastmath=True)
def _rdc_cancor(X: np.ndarray, Y: np.ndarray) -> float:
    """Largest canonical correlation between X and Y feature matrices.

    Computes max absolute correlation between standardized columns.
    """
    n, p = X.shape
    q = Y.shape[1]

    # Standardize X columns (in-place)
    for j in range(p):
        mu = 0.0
        for i in range(n):
            mu += X[i, j]
        mu /= n
        ss = 0.0
        for i in range(n):
            X[i, j] -= mu
            ss += X[i, j] * X[i, j]
        if ss > 1e-10:
            inv_std = 1.0 / np.sqrt(ss)
            for i in range(n):
                X[i, j] *= inv_std

    # Standardize Y columns (in-place)
    for j in range(q):
        mu = 0.0
        for i in range(n):
            mu += Y[i, j]
        mu /= n
        ss = 0.0
        for i in range(n):
            Y[i, j] -= mu
            ss += Y[i, j] * Y[i, j]
        if ss > 1e-10:
            inv_std = 1.0 / np.sqrt(ss)
            for i in range(n):
                Y[i, j] *= inv_std

    # Find max |corr(X[:,j], Y[:,k])| - X,Y already standardized so corr = X'Y/n
    max_corr = 0.0
    for j in range(p):
        for k in range(q):
            corr = 0.0
            for i in range(n):
                corr += X[i, j] * Y[i, k]
            if corr < 0:
                corr = -corr
            if corr > max_corr:
                max_corr = corr

    return min(max_corr, 1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _rdc(x: np.ndarray, y: np.ndarray, k: int, s: float, seed: int) -> float:
    """Randomized Dependence Coefficient."""
    n = len(x)
    if n < 3:
        return 0.0

    # Check constant
    x_min, x_max = x[0], x[0]
    y_min, y_max = y[0], y[0]
    for i in range(1, n):
        if x[i] < x_min:
            x_min = x[i]
        if x[i] > x_max:
            x_max = x[i]
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]
    if x_max - x_min < 1e-10 or y_max - y_min < 1e-10:
        return 0.0

    # Create features
    X_feat = _rdc_features(x, k, s, seed)
    Y_feat = _rdc_features(y, k, s, seed + 1000)

    return _rdc_cancor(X_feat, Y_feat)


@ClassifierSelectors.register("rdc")
def rdc_classifier(
    x: np.ndarray, y: np.ndarray, n_classes: int, random_state: int | None = None
) -> float:
    """RDC for classification.

    O(n log n) non-linear dependence.
    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    seed = 42 if random_state is None else random_state

    if n_classes == 2:
        return _rdc(x, y.astype(np.float64), _RDC_K, _RDC_S, seed)

    # Multi-class: max RDC over one-vs-all
    max_rdc = 0.0
    for c in range(n_classes):
        rdc_c = _rdc(x, (y == c).astype(np.float64), _RDC_K, _RDC_S, seed + c)
        if rdc_c > max_rdc:
            max_rdc = rdc_c
    return max_rdc


@RegressorSelectors.register("rdc")
def rdc_regressor(
    x: np.ndarray, y: np.ndarray, standardize: bool, random_state: int | None = None
) -> float:
    """RDC for regression.

    O(n log n) non-linear dependence.
    """
    if x.ndim > 1:
        x = x.ravel()
    if y.ndim > 1:
        y = y.ravel()

    seed = 42 if random_state is None else random_state
    return _rdc(x, y, _RDC_K, _RDC_S, seed)


@ClassifierSelectorTests.register("mc")
def ptest_mc(
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
        return _ptest_mc_parallel(
            x=x,
            y=y,
            n_classes=n_classes,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return _ptest_compiled(
        func=mc,
        func_arg=n_classes,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@ClassifierSelectorTests.register("mi")
def ptest_mi(
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
    return _ptest(
        func=mi,
        func_arg=n_classes,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@RegressorSelectorTests.register("pc")
def ptest_pc(
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
        return _ptest_pc_parallel(
            x=x,
            y=y,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    return _ptest_compiled(
        func=pc,
        func_arg=standardize,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@RegressorSelectorTests.register("dc")
def ptest_dc(
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
    return _ptest(
        func=dc,
        func_arg=standardize,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


# =============================================================================
# RDC Permutation Tests
# =============================================================================


@ClassifierSelectorTests.register("rdc")
def ptest_rdc_classifier(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Perform a permutation test using the Randomized Dependence Coefficient.

    RDC-based permutation test for feature selection in classification.
    O(n log n) per permutation vs O(n²) for distance correlation.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values (class labels).

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
    return _ptest(
        func=rdc_classifier,
        func_arg=n_classes,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )


@RegressorSelectorTests.register("rdc")
def ptest_rdc_regressor(
    *,
    x: np.ndarray,
    y: np.ndarray,
    standardize: bool,
    n_resamples: int,
    early_stopping: bool,
    alpha: float,
    random_state: int,
) -> float:
    """Perform a permutation test using the Randomized Dependence Coefficient.

    RDC-based permutation test for feature selection in regression.
    O(n log n) per permutation vs O(n²) for distance correlation.

    Parameters
    ----------
    x : np.ndarray
        Feature values.

    y : np.ndarray
        Target values.

    standardize : bool
        Whether to standardize the result. RDC is inherently standardized.

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
    return _ptest(
        func=rdc_regressor,
        func_arg=standardize,
        x=x,
        y=y,
        n_resamples=n_resamples,
        early_stopping=early_stopping,
        alpha=alpha,
        random_state=random_state,
    )
