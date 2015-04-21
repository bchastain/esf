# -*- coding: utf-8 -*-
"""Python Eigenvector Spatial Filtering
This module provides a port of the SpatialFiltering function from the
R spdep library, written by Michael Tiefelsdorf, Yongwan Chun and
Roger Bivand ((c) 2005) and distributed  under the terms of the GNU
General Public License, version 2.

References:
    Roger Bivand, Gianfranco Piras (2015). Comparing Implementations of
    Estimation Methods for Spatial Econometrics. Journal of Statistical
    Software, 63(18), 1-36. URL http://www.jstatsoft.org/v63/i18/.

    Bivand, R. S., Hauke, J., and Kossowski, T. (2013). Computing the
    Jacobian in Gaussian spatial autoregressive models: An illustrated
    comparison of available methods. Geographical Analysis, 45(2),
    150-179.

    Tiefelsdorf M, Griffith DA. (2007) Semiparametric Filtering of
    Spatial Autocorrelation: The Eigenvector Approach. Environment and
    Planning A, 39 (5) 1193 - 1221. http://www.spatialfiltering.com
"""

__author__ = "Bryan Chastain <chastain@utdallas.edu>"

import math

import scipy.stats as stat
import numpy as np
import numpy.linalg as LA
import pysal

def _getmoranstat(MSM, degfree):
    #Internal function for calculating Moran's I, given MSM matrix and d.f.
    t1 = np.sum(np.diag(MSM))
    t2 = np.sum(np.diag(MSM * MSM))
    E = t1 / degfree
    V = 2 * (degfree*t2 - t1*t1) / (degfree * degfree * (degfree+2))
    return E, V


def _altfunction(ZI, alternative):
    #Internal function for returning p-value based on user-selected tail(s)
    if(alternative == "two.sided"):
        return 2 * (1 - stat.norm.cdf(abs(ZI)))
    elif(alternative == "greater"):
        return (1 - stat.norm.cdf(ZI))
    else:
        return stat.norm.cdf(ZI)


def spatialfiltering(
        depvar,
        indepvars,
        spatiallagvars,
        data,
        nb,
        style="d",
        zeropolicy=False,
        tol=0.1,
        zerovalue=0.0001,
        ExactEV=False,
        symmetric=True,
        alpha=None,
        alternative="two.sided",
        verbose=False):
    """This function uses the Tiefelsdorf & Griffith (2007) method for
    performing a semi-parametric spatial filtering approach for removing
    spatial dependence from linear models. A brute-force selection
    method is employed for finding eigenvectors that reduce the Moran's
    I value for regression residuals the most, and it continues until
    no remaining candidate eigenvectors can reduce the value by more
    than "tol". The function returns a summary table of the selection
    process as well as a matrix of the final selected eigenvectors.

    Args:
        depvar (str):
        indepvars (list of str):
        spatiallagvars (list of str):
        data (str):
        nb (str):
        style (str):
        zeropolicy (bool):
        tol (float):
        zerovalue (float):
        ExactEV (bool):
        symmetric (bool):
        alpha (float):
        alternative (str):
        verbose (bool):

    Returns:
        A tuple comprised of a summary table of the selection process
        as well as a matrix of the final selected eigenvectors.
    """

    if nb == "":
        raise Exception("Neighbour list argument missing")
    if depvar == "":
        raise Exception("Missing dependent variable")
    if len(indepvars) == 0:
        raise Exception("Missing independent variable(s)")

    # supplement given neighbors list with spatial weights for given coding
    # scheme (r=row-standardized, d=double standardized, b=binary, v=variance
    # stabilized)
    w = pysal.open(nb).read()
    w.transform = style

    S, ids = pysal.full(w)

    # if symmetric=true, constructs a weights list object corresponding to the
    # sparse matrix 1/2 (W + W')
    if symmetric:
        S = 0.5 * (S + S.T)

    S = w.s0 / S.shape[0] * S

    nofreg = S.shape[0]  # number of observations

    # Generate Eigenvectors if eigen vectors are not given
    # (M1 for no SAR, MX for SAR)

    db = pysal.open(data, 'r')
    y = np.array(db.by_col(depvar))
    if(np.count_nonzero(np.isnan(y)) > 0):
        raise Exception("NAs in dependent variable")

    xsar = []
    xsar.append([1] * nofreg)
    for indep in indepvars:
        xsar.append(db.by_col(indep))
    xsar = np.matrix(xsar).T
    if(np.count_nonzero(np.isnan(xsar)) > 0):
        raise Exception("NAs in independent variable(s)")

    if(xsar.shape[0] != S.shape[0]):
        raise Exception(
            "Input data and neighbourhood list have different dimensions")

    q, r = LA.qr(np.transpose(xsar) * xsar)
    p = np.dot(q.T, np.transpose(xsar))
    qrsolve = np.dot(LA.inv(r), p)
    mx = np.identity(nofreg) - xsar*qrsolve
    S = mx * S * mx

    # Get EigenVectors and EigenValues
    v, d = LA.eig(S)
    sortid = v.argsort()[::-1]
    v = v[sortid]
    d = d[:, sortid]

    if len(spatiallagvars) == 0:
        X = xsar
    else:
        X = xsar
        for lag in spatiallagvars:
            X = np.hstack((X, np.matrix(db.by_col(lag)).T))

    y.shape = (y.shape[0], 1)
	coll_test = pysal.spreg.OLS(np.array(y), np.array(X[:, 1:]))
    if(np.count_nonzero(np.isnan(coll_test.betas)) > 0):
        raise Exception("Collinear RHS variable detected")
    # X will be augmented by the selected eigenvectors

    # Total sum of squares for R2
    TSS = np.sum(np.asarray(y - np.mean(y))**2)

    # Compute first Moran Expectation and Variance
    nofexo = X.shape[1]
    degfree = nofreg - nofexo
    M = (np.identity(nofreg) - X *
         LA.solve((np.transpose(X) * X), np.transpose(X)))
    MSM = M * S * M
    E, V = _getmoranstat(MSM, degfree)

    y = np.matrix(y)
    # Matrix storing the iteration history:
    #   [1] Step counter of the selection procedure
    #   [2] number of selected eigenvector (sorted descending)
    #   [3] its associated eigenvalue
    #   [4] value Moran's I for residual autocorrelation
    #   [5] standardized value of Moran's I assuming a normal approximation
    #   [6] p-value of [5] for given alternative
    #   [7] R^2 of the model including exogenous variables and eigenvectors
    #   c("Step","SelEvec","Eval","MinMi","ZMinMi","R2","gamma")
    # Store the results at Step 0 (i.e., no eigenvector selected yet)
    cyMy = (y.T * M) * y
    cyMSMy = (y.T * MSM) * y
    IthisTime = cyMSMy / cyMy
    zIthisTime = (IthisTime - E) / math.sqrt(V)

    PrI = _altfunction(zIthisTime, alternative)

    Aout = np.matrix([0, 0, 0, IthisTime, zIthisTime, PrI, 1 - (cyMy/TSS)])
    if verbose:
        print("Step", Aout[0, 0], "SelEvec", Aout[0, 1], "MinMi", Aout[0, 3],
              "ZMinMi", Aout[0, 4], "Pr(ZI)", Aout[0, 5])
    # Define search eigenvalue range
    # The search range is restricted into a sign range based on Moran's I
    # Put a sign for eigenvectors associated with their eigenvalues
    # if val > zerovalue (e.g. if val > 0.0001), then 1
    # if val < zerovalue (e.g. if val < -0.0001), then -1
    # otherwise 0

    sel = np.vstack((np.r_[1:nofreg + 1], v, np.zeros(nofreg))).T
    sel[:, 2] = ((v > abs(zerovalue)).astype(int) -
                 (v < -abs(zerovalue)).astype(int))

    # Compute the Moran's I of the aspatial model (without any eigenvector)
    # i.e., the sign of autocorrelation
    # if MI is positive, then acsign = 1
    # if MI is negative, then acsign = -1
    res = y - X*LA.solve((np.transpose(X) * X), (np.transpose(X) * y))
    acsign = 1
    if(((np.transpose(res) * S) * res) / (np.transpose(res) * res) < 0):
        acsign = -1

    # If only sar model is applied or just the intercept,
    # Compute and save coefficients for all eigenvectors
    onlysar = False
    # if (missing(xlag) & !missing(xsar))
    if len(spatiallagvars) == 0:
        onlysar = True
        Xcoeffs = LA.solve((np.transpose(X) * X), (np.transpose(X) * y))
        gamma4eigenvec = np.vstack((np.r_[1:nofreg + 1], np.zeros(nofreg))).T
    # Only SAR the first parameter estimation for all eigenvectors
    # Due to orthogonality each coefficient can be estimate individually
        for j in range(0, nofreg):
            if (sel[j, 2] == acsign):  # Use only feasible unselected evecs
                gamma4eigenvec[j, 1] = LA.solve(
                    np.transpose(d[:, j]) * d[:, j], np.transpose(d[:, j]) * y)

    # Here the actual search starts - The inner loop check each candidate -
    # The outer loop selects eigenvectors until the residual autocorrelation
    # falls below 'tol'
    # Loop over all eigenvectors with positive or negative eigenvalue
    oldZMinMi = float("inf")
    for i in range(0, nofreg):  # Outer Loop
        z = float("inf")
        idx = -1
        for j in range(0, nofreg):  # Inner Loop - Find next eigenvector
            if(sel[j, 2] == acsign):  # Use only feasible unselected evecs
                xe = np.hstack((X, d[:, j]))  # Add test eigenvector
                # Based on whether it is an only SAR model or not
                if onlysar:
                    res = y - xe * np.vstack((Xcoeffs, gamma4eigenvec[j, 1]))
                else:
                    res = y - xe * LA.solve(np.transpose(xe) *
                                            xe, np.transpose(xe) * y)

                mi = (((np.transpose(res) * S) * res) /
                      (np.transpose(res) * res))

                if ExactEV:
                    ident = np.identity(nofreg)
                    M = (ident - xe *
                         LA.solve(np.transpose(xe) * xe, np.transpose(xe)))
                    degfree = nofreg - xe.shape[1]
                    MSM = M * S * M
                    E, V = _getmoranstat(MSM, degfree)

                if(abs((mi - E) / math.sqrt(V)) < z):  # Identify min z(Moran)
                    MinMi = mi
                    z = (MinMi - E) / math.sqrt(V)
                    idx = j + 1

        # Update design matrix permanently by selected eigenvector
        if(idx > 0):
            X = np.hstack((X, d[:, idx - 1]))
            if onlysar:
                Xcoeffs = np.vstack((Xcoeffs, gamma4eigenvec[idx - 1, 1]))

            M = (np.identity(nofreg) - X *
                 LA.solve(np.transpose(X) * X, np.transpose(X)))
            degfree = nofreg - X.shape[1]
            MSM = M * S * M
            E, V = _getmoranstat(MSM, degfree)
            ZMinMi = ((MinMi - E) / math.sqrt(V))
            out = [i + 1,
                   idx,
                   v[idx - 1],
                   MinMi[0, 0],
                   ZMinMi[0, 0],
                   _altfunction(ZMinMi, alternative)[0, 0],
                   (1 - ((np.transpose(y) * M) * y / TSS))]
            if verbose:
                print("Step", out[0], "SelEvec", out[1], "MinMi", out[3],
                      "ZMinMi", out[4], "Pr(ZI)", out[5])
            Aout = np.vstack((Aout, out))
            sel[idx - 1, 2] = 0
            if not alpha:
                if(abs(ZMinMi) < tol):
                    break
                elif(abs(ZMinMi) > abs(oldZMinMi)):
                    if not ExactEV:
                        out = "An inversion has been detected. The procedure "
                        out += "will terminate now.\nIt is suggested to use "
                        out += "the exact expectation and variance of Moran's "
                        out += "I\nby setting the option ExactEV to TRUE.\n"
                        print out
                    break
            else:
                if(_altfunction(ZMinMi, alternative) >= alpha):
                    break

            if not ExactEV:
                if (abs(ZMinMi) > abs(oldZMinMi)):
                    out = "An inversion has been detected. The procedure "
                    out += "will terminate now.\nIt is suggested to use "
                    out += "the exact expectation and variance of Moran's "
                    out += "I\nby setting the option ExactEV to TRUE.\n"
                    print out
                    break

            oldZMinMi = ZMinMi

    betagam = LA.solve(np.transpose(X) * X, np.transpose(X) * y)
    gammas = betagam[nofexo:betagam.shape[0]]

    gammas = np.vstack((0, gammas))
    out = np.hstack((Aout, gammas))

    eiglist = (np.array(out[1:, 1].T)[0] - 1).tolist()
    selVec = d[:, eiglist]
    return out, selVec


if __name__ == "__main__":
    depvar = "LOGB_WM_P2"
    indepvars = ["LOGPOPDEN", "LOGL_WM_P1"]
    spatiallag = []
    nb = "C:\\SEA.GAL"
    data = "C:\\SEA.DBF"
    style = "v"
    zeropolicy = False
    tol = 0.1
    zerovalue = 0.0001
    ExactEV = False
    symmetric = True
    alpha = None
    alternative = "two.sided"
    verbose = True
    spatialfiltering(
        depvar,
        indepvars,
        spatiallag,
        data,
        nb,
        style,
        zeropolicy,
        tol,
        zerovalue,
        ExactEV,
        symmetric,
        alpha,
        alternative,
        verbose)
