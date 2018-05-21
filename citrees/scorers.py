from __future__ import division, print_function

import ctypes
from numba import autojit
import numpy as np
from os.path import dirname, join
import pandas as pd

from externals.six.moves import range

# __all__ = [""]

# Define constants for wrapping C functions
SHARED_OBJECT_DIR = join(dirname(__file__), 'bin')

# Weighted distance correlation
CFUNC_DCORS_PATH               = join(SHARED_OBJECT_DIR, 'dcor.so')
CFUNC_DCORS_DLL                = ctypes.CDLL(CFUNC_DCORS_PATH)
CFUNC_DCORS_DLL.wdcor.argtypes = (
        ctypes.POINTER(ctypes.c_double), # x
        ctypes.POINTER(ctypes.c_double), # y
        ctypes.c_int,                    # n   
        ctypes.POINTER(ctypes.c_double)  # w              
        )
CFUNC_DCORS_DLL.wdcor.restype  = ctypes.c_double

# Unweighted distance correlation
CFUNC_DCORS_DLL.dcor.argtypes = (
        ctypes.POINTER(ctypes.c_double), # x
        ctypes.POINTER(ctypes.c_double), # y
        ctypes.c_int,                    # n                 
        )
CFUNC_DCORS_DLL.dcor.restype  = ctypes.c_double


#######################
"""Feature selectors"""
#######################

# Lambda function used in approx_wdcor function
MEAN = lambda z: sum(z)/float(len(z))


@autojit(cache=True, nopython=True, nogil=True)
def pcor(x, y):
    """ADD HERE
    
    Parameters
    ----------
    
    Returns
    -------
    """
    if x.ndim > 1: x = x.ravel()
    if y.ndim > 1: y = y.ravel()

    # Define variables for looping
    n, mu_x, mu_y, cov_xy, var_x, var_y = len(x), 0.0, 0.0, 0.0, 0.0, 0.0

    # Means
    for i in xrange(n):
        mu_x += x[i]
        mu_y += y[i]

    mu_x /= float(n)
    mu_y /= float(n)
    
    # Covariances
    for i in xrange(n):
        e_x     = x[i]-mu_x
        e_y     = y[i]-mu_y
        cov_xy += e_x*e_y
        var_x  += e_x*e_x
        var_y  += e_y*e_y

    # Catch division by zero errors
    if var_x == 0 or var_y == 0:
        return 0.0
    else:
        return cov_xy/np.sqrt(var_x*var_y)


@autojit(cache=True, nopython=True, nogil=True)
def py_wdcor(x, y, n, weights):
    """Python port of C function for distance correlation

    Note: Version is optimized for use with Numba

    Parameters
    ----------
    x : 1d array-like
        Array of length n

    y : 1d array-like
        Array of length n

    n : int
        Sample size

    weights : 1d array-like
        Weight vector that sums to 1

    Returns
    -------
    cor : float
        Distance correlation
    """
    # Define initial variables
    s   = int(n*(n-1)/2.)
    Edx = np.zeros(n)
    Edy = np.zeros(n)
    DMY = np.zeros(s)
    DMX = np.zeros(s)
    F   = np.zeros(s)
    S1  = 0
    S2  = 0
    S3  = 0
    S2a = 0
    S2b = 0
    S1X = 0
    S1Y = 0
    S2X = 0
    S2Y = 0
    S3X = 0
    S3Y = 0
    k   = 0

    for i in range(n-1):
        for j in range(i+1, n):

            # Distance matrices
            DMX[k]  = np.fabs(x[i]-x[j])
            DMY[k]  = np.fabs(y[i]-y[j])
            F[k]    = weights[i]*weights[j]
            S1     += DMX[k]*DMY[k]*F[k]
            S1X    += DMX[k]*DMX[k]*F[k]
            S1Y    += DMY[k]*DMY[k]*F[k]
            Edx[i] += DMX[k]*weights[j]
            Edy[j] += DMY[k]*weights[i]
            Edx[j] += DMX[k]*weights[i]
            Edy[i] += DMY[k]*weights[j]
            k      += 1
    
    # Means
    for i in range(n):
        S3  += Edx[i]*Edy[i]*weights[i]
        S2a += Edy[i]*weights[i] 
        S2b += Edx[i]*weights[i]
        S3X += Edx[i]*Edx[i]*weights[i]
        S3Y += Edy[i]*Edy[i]*weights[i]

    # Variance and covariance terms
    S1  = 2*S1
    S1Y = 2*S1Y
    S1X = 2*S1X
    S2  = S2a*S2b
    S2X = S2b*S2b
    S2Y = S2a*S2a

    if S1X == 0 or S2X == 0 or S3X == 0 or S1Y == 0 or S2Y == 0 or S3Y == 0:
        return 0
    else:
        return np.sqrt( (S1+S2-2*S3) / np.sqrt( (S1X+S2X-2*S3X)*(S1Y+S2Y-2*S3Y) ))


@autojit(cache=True, nopython=True, nogil=True)
def py_dcor(x, y, n):
    """Python port of C function for distance correlation

    Note: Version is optimized for use with Numba

    Parameters
    ----------
    x : 1d array-like
        Array of length n

    y : 1d array-like
        Array of length n

    n : int
        Sample size

    Returns
    -------
    cor : float
        Distance correlation
    """
    s   = int(n*(n-1)/2.)
    n2  = n*n
    n3  = n2*n
    n4  = n3*n
    Edx = np.zeros(n)
    Edy = np.zeros(n)
    DMY = np.zeros(s)
    DMX = np.zeros(s)
    S1  = 0
    S2  = 0
    S3  = 0
    S2a = 0
    S2b = 0
    S1X = 0
    S1Y = 0
    S2X = 0
    S2Y = 0
    S3X = 0
    S3Y = 0
    k   = 0

    for i in range(n-1):
        for j in range(i+1, n):

            # Distance matrices
            DMX[k]  = np.fabs(x[i]-x[j])
            DMY[k]  = np.fabs(y[i]-y[j])
            S1     += DMX[k]*DMY[k]
            S1X    += DMX[k]*DMX[k]
            S1Y    += DMY[k]*DMY[k]
            Edx[i] += DMX[k]
            Edy[j] += DMY[k]
            Edx[j] += DMX[k]
            Edy[i] += DMY[k]
            k      += 1
    
    # Means
    for i in range(n):
        S3  += Edx[i]*Edy[i]
        S2a += Edy[i] 
        S2b += Edx[i]
        S3X += Edx[i]*Edx[i]
        S3Y += Edy[i]*Edy[i]

    # Variance and covariance terms
    S1   = (2*S1)/float(n2)
    S1Y  = (2*S1Y)/float(n2)
    S1X  = (2*S1X)/float(n2)
    S2   = S2a*S2b/float(n4)
    S2X  = S2b*S2b/float(n4)
    S2Y  = S2a*S2a/float(n4)
    S3  /= float(n3)
    S3X /= float(n3)
    S3Y /= float(n3)

    if S1X == 0 or S2X == 0 or S3X == 0 or S1Y == 0 or S2Y == 0 or S3Y == 0:
        return 0
    else:
        return np.sqrt( (S1+S2-2*S3) / np.sqrt( (S1X+S2X-2*S3X)*(S1Y+S2Y-2*S3Y) ))


def approx_wdcor(x, y, n):
    """ADD DESCRIPTION HERE

    NOTE: Code ported from R function approx.dcor at: 
        https://rdrr.io/cran/extracat/src/R/wdcor.R
    
    Parameters
    ----------
    x : 1d array-like
        Array of length n

    y : 1d array-like
        Array of length n

    n : int
        Sample size
    
    Returns
    -------
    dcor : float
        ADD
    """
    # Equal cuts and then create dataframe
    cx = pd.cut(x, n, include_lowest=True)
    cy = pd.cut(y, n, include_lowest=True)
    df = pd.DataFrame(
            np.column_stack([x, y, cx, cy]), columns=['x', 'y', 'cx', 'cy']
        )

    # Average values in interval
    vx = df['x'].groupby(df['cx'], sort=False).agg(MEAN).values
    vy = df['y'].groupby(df['cy'], sort=False).agg(MEAN).values

    # Calculate frequencies based on groupings
    f = df[['x', 'y']].groupby([df['cx'], df['cy']], sort=False).size()

    # Normalize weights and calculate weighted distance correlation
    w = f.values/float(f.values.sum())

    # Recompute n and s
    n = len(w)
    s = int(n*(n-1)/2.)

    # Call either the Python or C version based on array length
    if len(w) > 5000:
        return c_wdcor(vx[f.index.labels[0]], vy[f.index.labels[1]], n, w)
    else:
        return py_wdcor(vx[f.index.labels[0]], vy[f.index.labels[1]], n, w)


def c_wdcor(x, y, n, weights):
    """Wrapper for C version of weighted distance correlation

    Parameters
    ----------
    x : 1d array-like
        Array of length n

    y : 1d array-like
        Array of length n

    n : int
        Sample size

    weights : 1d array-like
        Weight vector that sums to 1

    Returns
    -------
    cor : float
        Distance correlation
    """
    array_type = ctypes.c_double*n
    return CFUNC_DCORS_DLL.wdcor(array_type(*x),
                                 array_type(*y),
                                 ctypes.c_int(n),
                                 array_type(*weights))

def c_dcor(x, y, n):
    """Wrapper for C version of distance correlation

    Parameters
    ----------
    x : 1d array-like
        Array of length n

    y : 1d array-like
        Array of length n

    n : int
        Sample size

    Returns
    -------
    cor : float
        Distance correlation
    """
    array_type = ctypes.c_double*n
    return CFUNC_DCORS_DLL.dcor(array_type(*x),
                                array_type(*y),
                                ctypes.c_int(n))

#####################
"""Split selectors"""
#####################


@autojit(cache=True, nopython=True, nogil=True)
def gini_index(y, labels):
    """ADD
    
    Parameters
    ----------
    
    Returns
    -------
    """
    # Gini index for each label
    n, gini = len(y), 0.0
    for label in labels:

        # Proportion of each label
        p = np.mean(y == label)

        # Only square if greater than 0
        if p > 0: gini += p*p

    # Weighted by class node size
    return 1-gini


if __name__ == '__main__':
    n = 10000
    s = int(n*(n-1)/2.)
    x=np.random.normal(0, 1, n)
    w=np.ones(n)/float(n)

    import time

    for i in xrange(10):
        start=time.time()
        py_wdcor(x, x, n, w)
        print("\nPYTHON", time.time()-start)

        start=time.time()
        approx_wdcor(x, x, n)
        print("PANDAS", time.time()-start)

        start=time.time()
        c_wdcor(x, x, n, w)
        print("C", time.time()-start)

        time.sleep(2)

# def get_variance(target_value_statistics_list):
#     """
#     :param target_value_statistics_list: a list of number
#     :return: variance
#     Example:
#     >>> get_variance([1, 1])
#     0.0
#     >>> get_variance([])
#     0.0
#     >>> get_variance([1,2])
#     0.5
#     """

#     count = len(target_value_statistics_list)
#     if count == 0:
#         return 0.0
#     average = 0
#     for val in target_value_statistics_list:
#         average += val * 1.0 / count

#     s_diff = 0
#     for val in target_value_statistics_list:
#         s_diff += (val - average) ** 2

#     return s_diff

