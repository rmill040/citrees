from __future__ import absolute_import, division, print_function

from joblib import delayed, Parallel
from numba import njit
import numpy as np

from scorers import c_dcor, mc_fast, mi, pcor, py_dcor, rdc, rdc_fast


##########################
"""CONTINUOUS SELECTORS"""
##########################

@njit(cache=True, nogil=True)
def permutation_test_pcor(x, y, B=100, random_state=None):
    """Permutation test for Pearson correlation

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    B : int
        Number of permutations

    random_state : int
        Sets seed for random number generator

    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)

    # Estimate correlation from original data
    theta = np.fabs(pcor(x, y))

    # Permutations
    y_      = y.copy()
    theta_p = np.zeros(B)
    for i in range(B):
        np.random.shuffle(y_)
        theta_p[i] = pcor(x, y_) # Call jitted function directly

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


@njit(cache=True, nogil=True)
def permutation_test_dcor(x, y, B=100, random_state=None):
    """Permutation test for distance correlation

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    B : int
        Number of permutations

    random_state : int
        Sets seed for random number generator

    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)

    # Estimate correlation from original data
    theta = np.fabs(py_dcor(x, y))

    # Permutations
    y_      = y.copy()
    theta_p = np.zeros(B)
    for i in range(B):
        np.random.shuffle(y_)
        theta_p[i] = py_dcor(x, y_) # Call jitted function directly

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


@njit(cache=True, nogil=True)
def permutation_test_rdc(x, y, B=100, random_state=None):
    """Permutation test for randomized dependence coefficient

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    B : int
        Number of permutations

    random_state : int
        Sets seed for random number generator

    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)

    # Estimate correlation from original data
    theta = np.fabs(rdc_fast(x, y))

    # Permutations
    y_      = y.copy()
    theta_p = np.zeros(B)
    for i in range(B):
        np.random.shuffle(y_)
        theta_p[i] = rdc_fast(x, y_) # Call jitted function directly

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


def _permutation(x, y, func=None, **kwargs):
    """Helper function to perform arbitrary permutation test

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    func : function handle
        Function to perform on permuted data

    Returns
    -------
    value : float
        Return value of function handle
    """
    np.random.shuffle(y)
    try:
        return func(x, y, **kwargs)
    except:
        return 0.0


def permutation_test_dcor_parallel(x, y, B=100, n_jobs=-1,
                                   random_state=None):
    """Parallel implementation of permutation test for distance correlation

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    B : int
        Number of permutations

    n_jobs : int
        Number of cpus to use for processing

    random_state : int
        Sets seed for random number generator

    Returns
    -------
    p : float
        Achieved significance level
    """
    n = x.shape[0]
    np.random.seed(random_state)

    # Define function handle
    func = py_dcor if n < 500 else c_dcor

    # Estimate correlation from original data
    theta = np.fabs(func(x, y))

    # Permutations
    y_ = y.copy()
    theta_p = [
        Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_permutation)(x, y_, func) for i in range(B)
            )
        ]

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


def permutation_test_rdc_parallel(x, y, B=100, n_jobs=-1, k=10, random_state=None):
    """Parallel implementation of permutation test for randomized dependence
    coefficient

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    B : int
        Number of permutations

    n_jobs : int
        Number of cpus to use for processing

    k : int
        Number of random projections for cca

    random_state : int
        Sets seed for random number generator

    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)
    func   = rdc
    kwargs = {'k': k}

    # Estimate correlation from original data
    theta = np.fabs(rdc(x, y, **kwargs))

    # Permutations
    y_ = y.copy()
    if n_jobs == 1:
        theta_p = np.zeros(B)
        for i in range(B):
            theta_p[i] = _permutation(x, y_, func, **kwargs)
    else:
        theta_p = [
            Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(_permutation)(x, y_, func, **kwargs)
                    for i in range(B)
                )
            ]

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


########################
"""DISCRETE SELECTORS"""
########################

@njit(cache=True, nogil=True, fastmath=True)
def permutation_test_mc(x, y, B=100, n_classes=None, random_state=None):
    """Permutation test for multiple correlation

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    n_classes : int
        Number of classes

    B : int
        Number of permutations

    random_state : int
        Sets seed for random number generator

    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)

    # Estimate correlation from original data
    theta = mc_fast(x, y, n_classes)

    # Permutations
    y_      = y.copy()
    theta_p = np.zeros(B)
    for i in range(B):
        np.random.shuffle(y_)
        theta_p[i] = mc_fast(x, y_, n_classes) # Call jitted function directly

    # Achieved significance level
    return np.mean(theta_p >= theta)


def permutation_test_mi(x, y, B=100, random_state=None, **kwargs):
    """Permutation test for mutual information

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    n_classes : int
        Number of classes

    B : int
        Number of permutations

    random_state : int
        Sets seed for random number generator

    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)

    # Estimate correlation from original data
    theta = mi(x, y)

    # Permutations
    y_      = y.copy()
    theta_p = np.zeros(B)
    for i in range(B):
        np.random.shuffle(y_)
        theta_p[i] = mi(x, y_)

    # Achieved significance level
    return np.mean(theta_p >= theta)
