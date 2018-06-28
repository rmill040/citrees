from __future__ import division, print_function

from joblib import delayed, Parallel
from numba import njit
import numpy as np
from scipy.stats import chi2

from scorers import c_dcor, pcor, py_dcor, rdc, rdc_fast


@njit(cache=True, nogil=True)
def permutation_test_pcor(x, y, agg, B=100, random_state=None):
    """Permutation test for Pearson correlation
    
    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    agg : 1d array-like
        Array of x, y concatenated

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
    theta_p, n_x, n_y = np.zeros(B), len(x), len(y)
    for i in range(B):
        np.random.shuffle(agg)
        theta_p[i] = pcor(agg[:n_x], agg[n_y:]) # Call jitted function directly

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


def _permutation(agg, n_x, n_y, func=None, **kwargs):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    np.random.shuffle(agg)
    try:
        return func(agg[:n_x], agg[n_y:], **kwargs)
    except:
        return 0.0


def test_rdc(x, y, agg, B=100, n_jobs=-1, k=10, random_state=None):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)
    n = x.shape[0]
    if True:
        return permutation_test_rdc_parallel(x, y, agg, B, n_jobs, random_state)
    else:
        rho   = rdc(x, y, k=k)
        chisq = (2.5 - n)*np.log(1-rho*rho)
        return 1-chi2.cdf(chisq, 1)


@njit(cache=True, nogil=True)
def permutation_test_rdc(x, y, agg, B=100, random_state=None):
    """Permutation test for randomized dependence coefficient
    
    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    agg : 1d array-like
        Array of x, y concatenated

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
    theta_p, n_x, n_y = np.zeros(B), len(x), len(y)
    for i in range(B):
        np.random.shuffle(agg)
        theta_p[i] = rdc_fast(agg[:n_x], agg[n_y:]) # Call jitted function directly

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


def permutation_test_rdc_parallel(x, y, agg, B=100, n_jobs=-1, k=10, random_state=None):
    """ADD
    
    Parameters
    ----------
    
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
    n_x, n_y = len(x), len(y)
    if n_jobs == 1:
        theta_p = np.zeros(B)
        for i in range(B):
            theta_p[i] = _permutation(agg, n_x, n_y, func, **kwargs)
    else:
        theta_p = [
            Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(_permutation)(agg, n_x, n_y, func, **kwargs) 
                    for i in range(B)
                )
            ]

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


@njit(cache=True, nogil=True)
def permutation_test_dcor(x, y, agg, B=100, random_state=None):
    """ADD

    NOTE: The jitted Python function is called here
    
    Parameters
    ----------
    
    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)

    # Estimate correlation from original data
    theta = np.fabs(py_dcor(x, y))

    # Permutations
    theta_p, n_x, n_y = np.zeros(B), len(x), len(y)
    for i in range(B):
        np.random.shuffle(agg)
        theta_p[i] = py_dcor(agg[:n_x], agg[n_y:])

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


def permutation_test_dcor_parallel(x, y, agg, B=100, n_jobs=-1,
                                   random_state=None):
    """ADD
    
    Parameters
    ----------
    
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
    n_x, n_y = len(x), len(y)
    theta_p = [
        Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_permutation)(agg, n_x, n_y, func) for i in range(B)
            )
        ]

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)
