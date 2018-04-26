from __future__ import division, print_function

from joblib import delayed, Parallel
from numba import autojit
import numpy as np

from scorers import c_dcor, pcor, py_dcor


@autojit(cache=True, nogil=True, nopython=True)
def permutation_test_pcor(x, y, agg, B=100):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # Estimate correlation from original data
    theta = np.fabs(pcor(x, y))

    # Permutations
    theta_p, n_x, n_y = np.zeros(B), len(x), len(y)
    for i in xrange(B):
        np.random.shuffle(agg)
        theta_p[i] = pcor(agg[:n_x], agg[n_y:])

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


@autojit(cache=True, nogil=True, nopython=True)
def permutation_test_dcor(x, y, agg, n, s, B=100):
    """ADD

    NOTE: The jitted Python function is called here
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # Estimate correlation from original data
    theta = np.fabs(py_dcor(x, y, n, s))

    # Permutations
    theta_p, n_x, n_y = np.zeros(B), len(x), len(y)
    for i in xrange(B):
        np.random.shuffle(agg)
        theta_p[i] = py_dcor(agg[:n_x], agg[n_y:], n, s)

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


def _permutation(agg, n_x, n_y, n, s, func=None):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    np.random.shuffle(agg)
    return func(agg[:n_x], agg[n_y:], n, s)


def permutation_test_dcor_parallel(x, y, agg, n, s, B=100, n_jobs=-1):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # Define function handle
    func = py_dcor if n < 500 else c_dcor

    # Estimate correlation from original data
    theta = np.fabs(func(x, y, n, s))

    # Permutations
    n_x, n_y = len(x), len(y)
    theta_p = [
        Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_permutation)(agg, n_x, n_y, n, s, func) for i in xrange(B)
            )
        ]

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


if __name__ == '__main__':
    n=500
    s=int(n*(n-1)/2.)
    x=np.random.normal(0, 1, n)

    import time
    start=time.time()
    permutation_test_dcor_parallel(x, x, np.concatenate([x, x]), n, s)
    print(time.time()-start)

    start=time.time()
    permutation_test_dcor(x, x, np.concatenate([x, x]), n, s, B=100)
    print(time.time()-start)

    start=time.time()
    permutation_test_dcor_parallel(x, x, np.concatenate([x, x]), n, s)
    print(time.time()-start)

    start=time.time()
    permutation_test_dcor(x, x, np.concatenate([x, x]), n, s, B=100)
    print(time.time()-start)
