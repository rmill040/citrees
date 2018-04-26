#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* 
Compiling example:
    gcc -Ofast -march=native -ffast-math -fPIC -shared -o dcor.so dcor.c
*/

double wdcor(double* x, double* y, int n, int s, double* w) {
    /* Distance correlation allowing weights

    Parameters
    ----------
    x : 1d double array pointer
        Pointer to array of length n

    y : 1d double array pointer
        Pointer to array of length n

    n : int
        Number of samples

    s : int
        Defined as int(n*(n-1)/2)

    w : 1d double array pointer
        Pointer to array of weights that sum to 1

    Returns
    -------
    wdcor : float
        Weighted distance correlation
    */
    // Initialize integers and doubles
    int i, j, k;
    double S1  = 0;
    double S2  = 0;
    double S3  = 0;
    double S2a = 0;
    double S2b = 0;
    double S1X = 0;
    double S1Y = 0;
    double S2X = 0;
    double S2Y = 0;
    double S3X = 0;
    double S3Y = 0;

    // Allocate arrays
    double DMY[s], DMX[s], F[s], Edx[n], Edy[n];
    for (i=0; i<s; i++) DMY[i] = DMX[i] = F[i] = 0;
    for (i=0; i<n; i++) Edx[i] = Edy[i] = 0;

    // Begin calculations
    k = 0;
    for (i=0; i<n-1; i++) {
        for (j=i+1; j<n; j++) {

            // Distance matrices
            DMX[k]  = fabs(x[i]-x[j]);
            DMY[k]  = fabs(y[i]-y[j]);
            F[k]    = w[i]*w[j];
            S1     += DMX[k]*DMY[k]*F[k];
            S1X    += DMX[k]*DMX[k]*F[k];
            S1Y    += DMY[k]*DMY[k]*F[k];
            Edx[i] += DMX[k]*w[j];
            Edy[j] += DMY[k]*w[i];
            Edx[j] += DMX[k]*w[i];
            Edy[i] += DMY[k]*w[j];
            k++;
        }
    }

    // Means
    for (i=0; i<n; i++) {
        S3  += Edx[i]*Edy[i]*w[i];
        S2a += Edy[i]*w[i]; 
        S2b += Edx[i]*w[i];
        S3X += Edx[i]*Edx[i]*w[i];
        S3Y += Edy[i]*Edy[i]*w[i];
    }

    // Variance and covariance terms
    S1  = 2*S1;
    S1Y = 2*S1Y;
    S1X = 2*S1X;
    S2  = S2a*S2b;
    S2X = S2b*S2b;
    S2Y = S2a*S2a;

    // Calculate result
    if (S1X == 0 || S2X == 0 || S3X == 0 || S1Y == 0 || S2Y == 0 || S3Y == 0){
        return 0.0;
    } else {
        return pow((S1+S2-2*S3)/pow((S1X+S2X-2*S3X)*(S1Y+S2Y-2*S3Y), 0.5), 0.5);

    }
}


double dcor(double* x, double* y, int n, int s) {
    /* Distance correlation

    Parameters
    ----------
    x : 1d double array pointer
        Pointer to array of length n

    y : 1d double array pointer
        Pointer to array of length n

    n : int
        Number of samples

    s : int
        Defined as int(n*(n-1)/2)

    Returns
    -------
    dcor : float
        Distance correlation
    */
    // Initialize integers and doubles
    int i, j, k;
    long n2    = n*n;   // These numbers can blow up quick so keep as long type
    long n3    = n2*n;
    long n4    = n3*n;
    double S1  = 0;
    double S2  = 0;
    double S3  = 0;
    double S2a = 0;
    double S2b = 0;
    double S1X = 0;
    double S1Y = 0;
    double S2X = 0;
    double S2Y = 0;
    double S3X = 0;
    double S3Y = 0;

    // Allocate arrays
    double DMY[s], DMX[s], Edx[n], Edy[n];
    for (i=0; i<s; i++) DMY[i] = DMX[i] = 0;
    for (i=0; i<n; i++) Edx[i] = Edy[i] = 0;

    // Begin calculations
    k = 0;
    for (i=0; i<n-1; i++) {
        for (j=i+1; j<n; j++) {

            // Distance matrices
            DMX[k]  = fabs(x[i]-x[j]);
            DMY[k]  = fabs(y[i]-y[j]);
            S1     += DMX[k]*DMY[k];
            S1X    += DMX[k]*DMX[k];
            S1Y    += DMY[k]*DMY[k];
            Edx[i] += DMX[k];
            Edy[j] += DMY[k];
            Edx[j] += DMX[k];
            Edy[i] += DMY[k];
            k++;
        }
    }

    // Means
    for (i=0; i<n; i++) {
        S3  += Edx[i]*Edy[i];
        S2a += Edy[i]; 
        S2b += Edx[i];
        S3X += Edx[i]*Edx[i];
        S3Y += Edy[i]*Edy[i];
    }

    // Variance and covariance terms
    S1   = (2*S1)/n2;
    S1Y  = (2*S1Y)/n2;
    S1X  = (2*S1X)/n2;
    S2   = (S2a*S2b)/n4;
    S2X  = (S2b*S2b)/n4;
    S2Y  = (S2a*S2a)/n4;
    S3  /= n3;
    S3X /= n3;
    S3Y /= n3;

    // Calculate result
    if (S1X == 0 || S2X == 0 || S3X == 0 || S1Y == 0 || S2Y == 0 || S3Y == 0){
        return 0.0;
    } else {
        return pow((S1+S2-2*S3)/pow((S1X+S2X-2*S3X)*(S1Y+S2Y-2*S3Y), 0.5), 0.5);

    }
}
