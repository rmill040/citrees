from __future__ import print_function

from numba import autojit
import numpy as np

from externals.six.moves import range


@autojit(nopython=True, cache=True, nogil=True)
def auc_score(y_true, y_prob):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    y_true, n   = y_true[np.argsort(y_prob)], len(y_true)
    nfalse, auc = 0, 0.0
    for i in range(n):
        nfalse += 1 - y_true[i]
        auc    += y_true[i] * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def logger(name, message):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    print('[{name}] {message}'.format(name=name.upper(), message=message))
