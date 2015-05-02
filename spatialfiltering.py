"""Python Eigenvector Spatial Filtering
This module provides a port of the SpatialFiltering function from the
R spdep library, written by Michael Tiefelsdorf, Yongwan Chun and
Roger Bivand ((c) 2005) and distributed  under the terms of the GNU
General Public License, version 2.

References
----------
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

import math

import scipy.stats as stat
import numpy as np
import numpy.linalg as LA
import pysal

__author__ = "Bryan Chastain <chastain@utdallas.edu>"


def _getmoranstat(MSM, degfree):
    # Internal function for calculating Moran's I expected value and variance,
    # given MSM matrix and d.f.
    trace_MSM = np.sum(np.diag(MSM))
    trace_MSM2 = np.sum(np.diag(MSM * MSM))
    expected = trace_MSM / degfree
    variance = (2 * (degfree*trace_MSM2 - trace_MSM*trace_MSM) /
                (degfree * degfree * (degfree+2)))
    return expected, variance


def _altfunction(ZI, alternative):
    # Internal function for returning p-value based on user-selected tail(s)
    if alternative == "two.sided":
        return 2 * (1 - stat.norm.cdf(abs(ZI)))
    elif alternative == "greater":
        return 1 - stat.norm.cdf(ZI)
    else:
        return stat.norm.cdf(ZI)


def spatialfiltering(
        dependent_var, independent_vars, spatial_lag_vars,
        data, neighbor_list, style="d", zero_policy=False,
        tolerance=0.1, zero_value=0.0001, exact_EV=False,
        symmetric=True, alpha=None, alternative="two.sided",
        verbose=False):
    """This function uses the Tiefelsdorf & Griffith (2007) method for
    performing a semi-parametric spatial filtering approach for removing
    spatial dependence from linear models. A brute-force selection
    method is employed for finding eigenvectors that reduce the Moran's
    I value for regression residuals the most, and it continues until
    no remaining candidate eigenvectors can reduce the value by more
    than "tolerance". The function returns a summary table of the selection
    process as well as a matrix of the final selected eigenvectors.

    Parameters
    ----------
    dependent_var : str
        Name of the response variable column in dataset
    independent_vars : list of str
        Names of indep. variable columns
    spatial_lag_vars : list of str
            Names of lagged variabled columns
    data : str
        Filename of dataset (.dbf)
    neighbor_list : str
        Filename of neighbor list file (.gal)
    style : str
        Style of spatial weights coding to be used - r=row-standardized,
        d=double standardized, b=binary, v=variance stabilized
    zero_policy : bool
        If False, stop with error for any empty neighbor sets, if True
        permit the weights list to be formed with zero-length weights vectors
    tolerance : float
            Tolerance value for convergence of spatial filtering
    zero_value : float
        Eigenvectors with eigenvalues of an absolute value smaller than
        zero_value will be excluded in eigenvector search
    exact_EV : bool
        Set exact_EV=True to use exact expectations and variances rather than
        the expectation and variance of Moran's I from the previous iteration
    symmetric : bool
        If True, spatial weights matrix forced to symmetry
    alpha : float
        If not None, used instead of the tolerance argument as a stopping
        rule to choose all eigenvectors up to and including the one with a
        probability value exceeding alpha.
    alternative : str
        String for specifying alternative hypothesis - "greater", "less" or
        "two.sided"
    verbose : bool
        If True, reports update on eigenvector selection during the
        brute-force search.

    Returns
    -------
    out, selVec : tuple of numpy.matrix
        A tuple comprised of a numpy matrix summary table of the selection
        process as well as a numpy matrix of the final selected eigenvectors.
        The summary table includes the following columns:

        Step: Step counter of the selection procedure

        SelEvec: number of selected eigenvector (sorted descending)

        Eval: its associated eigenvalue

        MinMi: value Moran's I for residual autocorrelation

        ZMinMi: standardized value of Moran's I assuming a normal
        approximation

        pr(ZI): probability value of the permutation-based
        standardized deviate for the given value of the alternative
        argument

        R2: R^2 of the model including exogenous variables and
        eigenvectors

        gamma: regression coefficient of selected eigenvector in
        fit

    Examples
    --------
    From the R SpatialFiltering example:

    >>> import numpy as np
    >>> import pysal
    >>> from esf.spatialfiltering import spatialfiltering
    >>> neighbor_list = pysal.examples.get_path("columbus.gal")
    >>> data = pysal.examples.get_path("columbus.dbf")
    >>> dependent_var = "CRIME"
    >>> independent_vars = ["INC", "HOVAL"]
    >>> spatiallag = None
    >>> style = "r"
    >>> zero_policy = False
    >>> tolerance = 0.1
    >>> zero_value = 0.0001
    >>> exact_EV = True
    >>> symmetric = True
    >>> alpha = None
    >>> alternative = "two.sided"
    >>> verbose = False
    >>> out, selVec = spatialfiltering(
    >>>     dependent_var, independent_vars, spatiallag, data,
    >>>     neighbor_list, style, zero_policy, tolerance, zero_value,
    >>>     exact_EV, symmetric, alpha, alternative, verbose
    >>> )
    >>> np.set_printoptions(precision=3, suppress=True)
    >>> hdr = "    Step    SelEvec Eval    MinMi"
    >>> hdr += "   ZMinMi  Pr(ZI)  R2     tgamma"
    >>> print hdr
    >>> print np.array_str(np.array(out))
        Step    SelEvec Eval    MinMi   ZMinMi  Pr(ZI)  R2     tgamma
    [[  0.      0.      0.      0.222   2.839   0.005   0.552   0.   ]
     [  1.      5.      0.706   0.126   1.971   0.049   0.627 -31.624]
     [  2.      3.      0.84    0.058   1.485   0.138   0.659  20.777]
     [  3.      1.      1.004  -0.021   0.906   0.365   0.685  18.88 ]
     [  4.     10.      0.341  -0.073   0.395   0.693   0.725 -22.972]
     [  5.     14.      0.188  -0.116  -0.045   0.964   0.764 -22.941]]

    """

    if neighbor_list == "":
        raise Exception("Neighbour list argument missing")
    if dependent_var == "":
        raise Exception("Missing dependent variable")
    if len(independent_vars) == 0 and len(spatial_lag_vars) == 0:
        raise Exception("Missing independent variable(s)")

    # Supplement given neighbors list with spatial weights for given coding
    # scheme (r=row-standardized, d=double standardized, b=binary, v=variance
    # stabilized).
    w = pysal.open(neighbor_list).read()
    w.transform = style

    # Return the full numpy array of the weights matrix.
    S, ids = pysal.full(w)

    # If symmetric=true, constructs a weights list object corresponding to the
    # sparse matrix 1/2 (W + W').
    if symmetric:
        S = 0.5 * (S + S.T)

    S = w.s0 / S.shape[0] * S

    # number of observations
    nofreg = S.shape[0]

    # Open the data file and store the dependent variable as a numpy array.
    db = pysal.open(data, 'r')
    y = np.array(db.by_col(dependent_var))
    # Check for missing values.
    if np.count_nonzero(np.isnan(y)) > 0:
        raise Exception("NAs in dependent variable")

    xsar = []
    # Add intercept.
    xsar.append([1] * nofreg)
    # Add data for each independent variable.
    for indep in independent_vars:
        xsar.append(db.by_col(indep))
    xsar = np.matrix(xsar).T
    # Check for missing values.
    if np.count_nonzero(np.isnan(xsar)) > 0:
        raise Exception("NAs in independent variable(s)")

    # Ensure data and spatial weights have the same dimensions.
    if xsar.shape[0] != S.shape[0]:
        raise Exception(
            "Input data and neighbourhood list have different dimensions")

    # Construct the MSM matrix.
    q, r = LA.qr(np.transpose(xsar) * xsar)
    p = np.dot(q.T, np.transpose(xsar))
    qrsolve = np.dot(LA.inv(r), p)
    mx = np.identity(nofreg) - xsar*qrsolve
    S = mx * S * mx

    # Calculate eigenvectors (v) and eigenvalues (d).
    v, d = LA.eig(S)

    # Sort eigenvalues - this is not necessary, but is included here in order
    # to compare results with R, which provides sorted eigenvalues by default.
    # For increased performance, the following 3 lines may be commented out.
    sortid = v.argsort()[::-1]
    v = np.real(v[sortid])
    d = np.real(d[:, sortid])

    # If not using spatial lag variables, just use independent variables
    if spatial_lag_vars is None or len(spatial_lag_vars) == 0:
        X = xsar
    else:
        # If using lagged variables, add them in now.
        X = xsar
        for lag in spatial_lag_vars:
            X = np.hstack((X, np.matrix(db.by_col(lag)).T))

    y.shape = (y.shape[0], 1)
    coll_test = pysal.spreg.OLS(np.array(y), np.array(X[:, 1:]))
    # Check for collinearity.
    if np.count_nonzero(np.isnan(coll_test.betas)) > 0:
        raise Exception("Collinear RHS variable detected")

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
    # if val > zero_value (e.g. if val > 0.0001), then 1
    # if val < zero_value (e.g. if val < -0.0001), then -1
    # otherwise 0

    sel = np.vstack((np.r_[1:nofreg + 1], v, np.zeros(nofreg))).T
    sel[:, 2] = ((v > abs(zero_value)).astype(int) -
                 (v < -abs(zero_value)).astype(int))
    # Compute the Moran's I of the aspatial model (without any eigenvector)
    # i.e., the sign of autocorrelation
    # if MI is positive, then acsign = 1
    # if MI is negative, then acsign = -1
    res = y - X*LA.solve((np.transpose(X) * X), (np.transpose(X) * y))
    acsign = 1
    if ((np.transpose(res) * S) * res) / (np.transpose(res) * res) < 0:
        acsign = -1

    # If only sar model is applied or just the intercept,
    # Compute and save coefficients for all eigenvectors
    onlysar = False
    # if (missing(xlag) & !missing(xsar))
    if spatial_lag_vars is None or len(spatial_lag_vars) == 0:
        onlysar = True
        Xcoeffs = LA.solve((np.transpose(X) * X), (np.transpose(X) * y))
        gamma4eigenvec = np.vstack((np.r_[1:nofreg + 1], np.zeros(nofreg))).T
    # Only SAR the first parameter estimation for all eigenvectors
    # Due to orthogonality each coefficient can be estimate individually
        for j in range(0, nofreg):
            if sel[j, 2] == acsign:  # Use only feasible unselected evecs
                gamma4eigenvec[j, 1] = LA.solve(
                    np.transpose(d[:, j]) * d[:, j], np.transpose(d[:, j]) * y)

    # Here the actual search starts - The inner loop check each candidate -
    # The outer loop selects eigenvectors until the residual autocorrelation
    # falls below 'tolerance'
    # Loop over all eigenvectors with positive or negative eigenvalue
    oldZMinMi = float("inf")
    for i in range(0, nofreg):  # Outer Loop
        z = float("inf")
        idx = -1
        for j in range(0, nofreg):  # Inner Loop - Find next eigenvector
            if sel[j, 2] == acsign:  # Use only feasible unselected evecs
                xe = np.hstack((X, d[:, j]))  # Add test eigenvector
                # Based on whether it is an only SAR model or not
                if onlysar:
                    res = y - xe * np.vstack((Xcoeffs, gamma4eigenvec[j, 1]))
                else:
                    res = y - xe * LA.solve(np.transpose(xe) *
                                            xe, np.transpose(xe) * y)

                mi = (((np.transpose(res) * S) * res) /
                      (np.transpose(res) * res))

                if exact_EV:
                    ident = np.identity(nofreg)
                    M = (ident - xe *
                         LA.solve(np.transpose(xe) * xe, np.transpose(xe)))
                    degfree = nofreg - xe.shape[1]
                    MSM = M * S * M
                    E, V = _getmoranstat(MSM, degfree)

                # Identify min z(Moran)
                if abs((mi - E) / math.sqrt(V)) < z:
                    MinMi = mi
                    z = (MinMi - E) / math.sqrt(V)
                    idx = j + 1

        # Update design matrix permanently by selected eigenvector
        if idx > 0:
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
                if abs(ZMinMi) < tolerance:
                    break
                elif abs(ZMinMi) > abs(oldZMinMi):
                    if not exact_EV:
                        out = "An inversion has been detected. The procedure "
                        out += "will terminate now.\nIt is suggested to use "
                        out += "the exact expectation and variance of Moran's "
                        out += "I\nby setting the option exact_EV to TRUE.\n"
                        print out
                    break
            else:
                if _altfunction(ZMinMi, alternative) >= alpha:
                    break

            if not exact_EV:
                if abs(ZMinMi) > abs(oldZMinMi):
                    out = "An inversion has been detected. The procedure "
                    out += "will terminate now.\nIt is suggested to use "
                    out += "the exact expectation and variance of Moran's "
                    out += "I\nby setting the option exact_EV to TRUE.\n"
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
