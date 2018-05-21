from __future__ import division, print_function

from joblib import delayed, Parallel
from numba import autojit
import numpy as np

from scorers import c_dcor, pcor, py_dcor


@autojit(cache=True, nogil=True, nopython=True)
def permutation_test_pcor(x, y, agg, B=100, random_state=None):
    """Permutation test for Pearson correlation
    
    Parameters
    ----------
    x : 1d array-like
        ADD

    y : 1d array-like
        ADD

    agg : 1d array-like
        ADD

    B : int
        ADD

    random_state : int
        ADD
    
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
    for i in xrange(B):
        np.random.shuffle(agg)
        theta_p[i] = pcor(agg[:n_x], agg[n_y:])

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


@autojit(cache=True, nogil=True, nopython=True)
def permutation_test_dcor(x, y, agg, n, B=100, random_state=None):
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
    theta = np.fabs(py_dcor(x, y, n))

    # Permutations
    theta_p, n_x, n_y = np.zeros(B), len(x), len(y)
    for i in xrange(B):
        np.random.shuffle(agg)
        theta_p[i] = py_dcor(agg[:n_x], agg[n_y:], n)

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


def _permutation(agg, n_x, n_y, n, func=None):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    np.random.shuffle(agg)
    return func(agg[:n_x], agg[n_y:], n)


def permutation_test_dcor_parallel(x, y, agg, n, B=100, n_jobs=-1,
                                   random_state=None):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    p : float
        Achieved significance level
    """
    np.random.seed(random_state)

    # Define function handle
    func = py_dcor if n < 500 else c_dcor

    # Estimate correlation from original data
    theta = np.fabs(func(x, y, n))

    # Permutations
    n_x, n_y = len(x), len(y)
    theta_p = [
        Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_permutation)(agg, n_x, n_y, n, func) for i in xrange(B)
            )
        ]

    # Achieved significance level
    return np.mean(np.fabs(theta_p) >= theta)


if __name__ == '__main__':
    n=75
    x=np.random.normal(0, 1, n)
    y=np.random.normal(0, 1, n)

    p = np.zeros(500)
    for i in xrange(500):
        p[i] = permutation_test_pcor(x, y, np.concatenate([x, y]), n, np.random.randint(1, 1000000))
    import pdb; pdb.set_trace()

    # permutation_test_dcor(x, y, np.concatenate([x, y]), n)
    # permutation_test_dcor(x, y, np.concatenate([x, y]), n)

