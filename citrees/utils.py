from __future__ import absolute_import, print_function

from numba import autojit
import numpy as np

from externals.six.moves import range


def bayes_boot_probs(n):
    """Bayesian bootstrap sampling for case weights
    
    Parameters
    ----------
    n : int
        Number of Bayesian bootstrap samples
    
    Returns
    -------
    p : 1d array-like
        Array of sampling probabilities
    """
    p = np.random.exponential(scale=1.0, size=n)
    return p/p.sum()


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
    """Prints messages with style "[NAME] message"
    
    Parameters
    ----------
    name : str
        Short title of message, for example, train or test

    message : str
        Main description to be displayed in terminal
    
    Returns
    -------
    None
    """
    print('[{name}] {message}'.format(name=name.upper(), message=message))


def estimate_margin(y_probs, y_true):
    """Estimates margin function of forest ensemble

    Note : This function is similar to margin in R's randomForest package
    
    Parameters
    ----------
    y_probs : 2d array-like
        Predicted probabilities where each row represents predicted
        class distribution for sample and each column corresponds to 
        estimated class probability

    y_true : 1d array-like
        Array of true class labels
    
    Returns
    -------
    margin : float
        Estimated margin of forest ensemble
    """
    # Calculate probability of correct class
    n, p        = y_probs.shape
    true_probs  = y_probs[np.arange(n, dtype=int), y_true]

    # Calculate maximum probability for incorrect class
    other_probs = np.zeros(n)
    for i in range(n):
        mask            = np.zeros(p, dtype=bool)
        mask[y_true[i]] = True
        other_idx       = np.ma.array(y_probs[i,:], mask=mask).argmax()
        other_probs[i]  = y_probs[i, other_idx]
    
    # Margin is P(y == j) - max(P(y != j))
    return true_probs - other_probs
