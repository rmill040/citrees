from __future__ import division, print_function

import ctypes
from numba import njit
import numpy as np
from os.path import dirname, join
import pandas as pd
from scipy.stats import rankdata as rank

from externals.six.moves import range


#######################
"""CREATE C WRAPPERS"""
#######################

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


###################################
"""FEATURE SELECTORS: CONTINUOUS"""
###################################

@njit(cache=True, nogil=True, fastmath=True)
def pcor(x, y):
    """Pearson correlation
    
    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements
    
    Returns
    -------
    cor : float
        Pearson correlation
    """
    if x.ndim > 1: x = x.ravel()
    if y.ndim > 1: y = y.ravel()

    # Define variables for looping
    n, sx, sy, sx2, sy2, sxy = len(x), 0.0, 0.0, 0.0, 0.0, 0.0

    # Loop to calculate statistics
    for i in range(n):
        xi   = x[i]
        yi   = y[i]
        sx  += xi
        sx2 += xi*xi
        sy  += yi
        sy2 += yi*yi
        sxy += xi*yi

    # Covariance terms
    cov = n*sxy - sx*sy
    ssx = n*sx2 - sx*sx
    ssy = n*sy2 - sy*sy

    # Catch division by zero errors
    if ssx == 0.0 or ssy == 0.0:
        return 0.0
    else:
        return cov/np.sqrt(ssx*ssy)


def cca(X, Y):
    """Largest canonical correlation
    
    Parameters
    ----------
    X : 2d array-like
        Array of n elements

    Y : 2d array-like
        Array of n elements
    
    Returns
    -------
    cor : float
        Largest canonical correlation between X and Y
    """    
    # Columns for X and Y
    Xp = X.shape[1]
    Yp = Y.shape[1]

    # Center X and Y and then QR decomposition
    X      = X-X.mean(axis=0)
    Y      = Y-Y.mean(axis=0)
    Qx, Rx = np.linalg.qr(X)
    Qy, Ry = np.linalg.qr(Y)

    # Check rank for X
    rankX = np.linalg.matrix_rank(Rx)
    if rankX == 0:
        return [0.0]
    elif rankX < Xp:
        Qx = Qx[:, 0:rankX]  
        Rx = Rx[0:rankX, 0:rankX]

    # Check rank for Y
    rankY = np.linalg.matrix_rank(Ry)
    if rankY == 0:
        return [0.0]
    elif rankY < Yp:
        Qy = Qy[:, 0:rankY]
        Ry = Ry[0:rankY, 0:rankY]

    # SVD then clip top eigenvalue
    QxQy    = np.dot(Qx.T, Qy)
    _, cor, _ = np.linalg.svd(QxQy)

    return np.clip(cor[0], 0, 1)


def rdc(X, Y, k=10, s=1.0/6.0, f=np.sin):
    """Randomized dependence coefficient
    
    Parameters
    ----------
    X : 2d array-like
        Array of n elements

    Y : 2d array-like
        Array of n elements

    k : int
        Number of random projections

    s : float
        Variance of Gaussian random variables

    f : function
        Non-linear function
    
    Returns
    -------
    cor : float
        Randomized dependence coefficient between X and Y
    """
    if X.ndim < 2: X = X.reshape(-1, 1)
    if Y.ndim < 2: Y = Y.reshape(-1, 1)

    # Shape of random vectors
    Xn, Xp = X.shape
    Yn, Yp = Y.shape

    # X data
    X_ones = np.ones((Xn, 1))
    X_     = np.array([rank(X[:, j])/float(Xn) for j in range(Xp)]).reshape(Xn, Xp)
    X_     = (s/X_.shape[1])*np.column_stack([X_, X_ones])
    X_     = X_.dot(np.random.randn(X_.shape[1], k))

    # Y data
    Y_ones = np.ones((Yn, 1))
    Y_     = np.array([rank(Y[:, j])/float(Yn) for j in range(Yp)]).reshape(Yn, Yp)
    Y_     = (s/Y_.shape[1])*np.column_stack([Y_, Y_ones])
    Y_     = Y_.dot(np.random.randn(Y_.shape[1], k))

    # Apply canonical correlation
    X_ = np.column_stack([f(X_), X_ones])
    Y_ = np.column_stack([f(Y_), Y_ones])
    
    return cca(X_, Y_)


@njit(cache=True, nogil=True, fastmath=True)
def cca_fast(X, Y):
    """Largest canonical correlation
    
    Parameters
    ----------
    X : 2d array-like
        Array of n elements

    Y : 2d array-like
        Array of n elements
    
    Returns
    -------
    cor : float
        Largest correlation between X and Y
    """    
    # Columns for X and Y
    Xp = X.shape[1]
    Yp = Y.shape[1]

    # Center X and Y and then QR decomposition
    mu_x   = np.array([np.mean(X[:, j]) for j in range(Xp)])
    mu_y   = np.array([np.mean(Y[:, j]) for j in range(Yp)])
    X      = X-mu_x
    Y      = Y-mu_y
    Qx, Rx = np.linalg.qr(X)
    Qy, Ry = np.linalg.qr(Y)

    # Check rank for X
    rankX = np.linalg.matrix_rank(Rx)
    if rankX == 0:
        return np.array([0.0])
    elif rankX < Xp:
        Qx = Qx[:, 0:rankX]  
        Rx = Rx[0:rankX, 0:rankX]

    # Check rank for Y
    rankY = np.linalg.matrix_rank(Ry)
    if rankY == 0:
        return np.array([0.0])
    elif rankY < Yp:
        Qy = Qy[:, 0:rankY]
        Ry = Ry[0:rankY, 0:rankY]

    # SVD then clip top eigenvalue
    QxQy    = np.dot(Qx.T, Qy)
    _, cor, _ = np.linalg.svd(QxQy)
    return cor



@njit(cache=True, nogil=True, fastmath=True)
def rdc_fast(x, y, k=10, s=1.0/6.0, f=np.sin):
    """Randomized dependence coefficient
    
    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    k : int
        Number of random projections

    s : float
        Variance of Gaussian random variables

    f : function
        Non-linear function
    
    Returns
    -------
    cor : float
        Randomized dependence coefficient between x and y
    """
    # Shape of random vectors
    xn = x.shape[0]
    yn = y.shape[0]

    # X data
    x_ones = np.ones((xn, 1))
    X_     = np.argsort(x)/float(xn)
    X_     = 0.5*s*np.column_stack((X_, x_ones))
    X_     = np.dot(X_, np.random.randn(2, k))

    # Y data
    y_ones = np.ones((yn, 1))
    Y_     = np.argsort(y)/float(yn)
    Y_     = 0.5*s*np.column_stack((Y_, y_ones))
    Y_     = np.dot(Y_, np.random.randn(2, k))

    # Apply canonical correlation
    X_ = np.column_stack((f(X_), x_ones))
    Y_ = np.column_stack((f(Y_), y_ones))
    
    cor = cca_fast(X_, Y_)[0]
    if cor < 0.0:
        return 0.0
    elif cor > 1.0:
        return 1.0
    else:
        return cor


@njit(cache=True, nogil=True, fastmath=True)
def py_wdcor(x, y, weights):
    """Python port of C function for distance correlation

    Note: Version is optimized for use with Numba

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    weights : 1d array-like
        Weight vector that sums to 1

    Returns
    -------
    dcor : float
        Distance correlation
    """
    # Define initial variables
    n   = x.shape[0]
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


@njit(cache=True, nogil=True, fastmath=True)
def py_dcor(x, y):
    """Python port of C function for distance correlation

    Note: Version is optimized for use with Numba

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    Returns
    -------
    dcor : float
        Distance correlation
    """
    n   = x.shape[0]
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


# Lambda function used in approx_wdcor function
MEAN = lambda z: sum(z)/float(len(z))

def approx_wdcor(x, y):
    """Approximate distance correlation by binning arrays

    NOTE: Code ported from R function approx.dcor at: 
        https://rdrr.io/cran/extracat/src/R/wdcor.R
    
    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    Returns
    -------
    dcor : float
        Distance correlation
    """
    # Equal cuts and then create dataframe
    n  = x.shape[0]
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

    # Recompute n
    n = len(w)

    # Call either the Python or C version based on array length
    if n > 5000:
        return c_wdcor(vx[f.index.labels[0]], vy[f.index.labels[1]], w)
    else:
        return py_wdcor(vx[f.index.labels[0]], vy[f.index.labels[1]], w)


def c_wdcor(x, y, weights):
    """Wrapper for C version of weighted distance correlation

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    weights : 1d array-like
        Weight vector that sums to 1

    Returns
    -------
    dcor : float
        Distance correlation
    """
    n = x.shape[0]
    array_type = ctypes.c_double*n
    return CFUNC_DCORS_DLL.wdcor(array_type(*x),
                                 array_type(*y),
                                 ctypes.c_int(n),
                                 array_type(*weights))


def c_dcor(x, y):
    """Wrapper for C version of distance correlation

    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    Returns
    -------
    dcor : float
        Distance correlation
    """
    n          = x.shape[0]
    array_type = ctypes.c_double*n
    return CFUNC_DCORS_DLL.dcor(array_type(*x),
                                array_type(*y),
                                ctypes.c_int(n))


#################################
"""FEATURE SELECTORS: DISCRETE"""
#################################

@njit(cache=True, nogil=True, fastmath=True)
def mc_fast(x, y, n_classes):
    """Multiple correlation
    
    Parameters
    ----------
    x : 1d array-like
        Array of n elements

    y : 1d array-like
        Array of n elements

    n_classes : int
        Number of classes

    Returns
    -------
    cor : float
        Multiple correlation coefficient between x and y
    """
    ssb, mu = 0.0, x.mean()

    # Sum of squares total
    sst = np.sum((x-mu)*(x-mu))  
    if sst == 0.0: return 0.0

    for j in range(n_classes):

        # Grab data for current class and if empty skip
        group = x[y==j]
        if group.shape[0] == 0: continue

        # Sum of squares between
        mu_j  = group.mean()
        n_j   = group.shape[0]
        ssb  += n_j*(mu_j-mu)*(mu_j-mu)

    return np.sqrt(ssb/sst)


###############################
"""SPLIT SELECTORS: DISCRETE"""
###############################

@njit(cache=True, nogil=True, fastmath=True)
def gini_index(y, labels):
    """Weighted gini index for classification

    Note: Despite being jitted, this function is still slow and a bottleneck
          in the actual training phase. Sklearn's Cython version is used to
          find the best split and this function is then called on the parent node
          and two child nodes to calculate feature importances using the mean
          decrease impurity formula
    
    Parameters
    ----------
    y : 1d array-like
        Array of labels

    labels : 1d array-like
        Unique labels
    
    Returns
    -------
    gini : float
        Weighted gini index
    """
    # Gini index for each label
    n, gini = len(y), 0.0
    for label in labels:

        # Proportion of each label
        p = np.mean(y == label)

        # Only square if greater than 0
        if p > 0: gini += p*p

    # Weighted by class node size
    return 1 - gini
