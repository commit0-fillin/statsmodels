"""
Miscellaneous utility code for VAR estimation
"""
from statsmodels.compat.pandas import frequencies
from statsmodels.compat.python import asbytes
from statsmodels.tools.validation import array_like, int_like
import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.tsa.tsatools as tsa

def get_var_endog(y, lags, trend='c', has_constant='skip'):
    """
    Make predictor matrix for VAR(p) process

    Z := (Z_0, ..., Z_T).T (T x Kp)
    Z_t = [1 y_t y_{t-1} ... y_{t - p + 1}] (Kp x 1)

    Ref: Lütkepohl p.70 (transposed)

    has_constant can be 'raise', 'add', or 'skip'. See add_constant.
    """
    pass

def make_lag_names(names, lag_order, trendorder=1, exog=None):
    """
    Produce list of lag-variable names. Constant / trends go at the beginning

    Examples
    --------
    >>> make_lag_names(['foo', 'bar'], 2, 1)
    ['const', 'L1.foo', 'L1.bar', 'L2.foo', 'L2.bar']
    """
    pass

def comp_matrix(coefs):
    """
    Return compansion matrix for the VAR(1) representation for a VAR(p) process
    (companion form)

    A = [A_1 A_2 ... A_p-1 A_p
         I_K 0       0     0
         0   I_K ... 0     0
         0 ...       I_K   0]
    """
    pass

def parse_lutkepohl_data(path):
    """
    Parse data files from Lütkepohl (2005) book

    Source for data files: www.jmulti.de
    """
    pass

def varsim(coefs, intercept, sig_u, steps=100, initial_values=None, seed=None, nsimulations=None):
    """
    Simulate VAR(p) process, given coefficients and assuming Gaussian noise

    Parameters
    ----------
    coefs : ndarray
        Coefficients for the VAR lags of endog.
    intercept : None or ndarray 1-D (neqs,) or (steps, neqs)
        This can be either the intercept for each equation or an offset.
        If None, then the VAR process has a zero intercept.
        If intercept is 1-D, then the same (endog specific) intercept is added
        to all observations.
        If intercept is 2-D, then it is treated as an offset and is added as
        an observation specific intercept to the autoregression. In this case,
        the intercept/offset should have same number of rows as steps, and the
        same number of columns as endogenous variables (neqs).
    sig_u : ndarray
        Covariance matrix of the residuals or innovations.
        If sig_u is None, then an identity matrix is used.
    steps : {None, int}
        number of observations to simulate, this includes the initial
        observations to start the autoregressive process.
        If offset is not None, then exog of the model are used if they were
        provided in the model
    initial_values : array_like, optional
        Initial values for use in the simulation. Shape should be
        (nlags, neqs) or (neqs,). Values should be ordered from less to
        most recent. Note that this values will be returned by the
        simulation as the first values of `endog_simulated` and they
        will count for the total number of steps.
    seed : {None, int}
        If seed is not None, then it will be used with for the random
        variables generated by numpy.random.
    nsimulations : {None, int}
        Number of simulations to perform. If `nsimulations` is None it will
        perform one simulation and return value will have shape (steps, neqs).

    Returns
    -------
    endog_simulated : nd_array
        Endog of the simulated VAR process. Shape will be (nsimulations, steps, neqs)
        or (steps, neqs) if `nsimulations` is None.
    """
    pass

def eigval_decomp(sym_array):
    """
    Returns
    -------
    W: array of eigenvectors
    eigva: list of eigenvalues
    k: largest eigenvector
    """
    pass

def vech(A):
    """
    Simple vech operator
    Returns
    -------
    vechvec: vector of all elements on and below diagonal
    """
    pass

def seasonal_dummies(n_seasons, len_endog, first_period=0, centered=False):
    """

    Parameters
    ----------
    n_seasons : int >= 0
        Number of seasons (e.g. 12 for monthly data and 4 for quarterly data).
    len_endog : int >= 0
        Total number of observations.
    first_period : int, default: 0
        Season of the first observation. As an example, suppose we have monthly
        data and the first observation is in March (third month of the year).
        In this case we pass 2 as first_period. (0 for the first season,
        1 for the second, ..., n_seasons-1 for the last season).
        An integer greater than n_seasons-1 are treated in the same way as the
        integer modulo n_seasons.
    centered : bool, default: False
        If True, center (demean) the dummy variables. That is useful in order
        to get seasonal dummies that are orthogonal to the vector of constant
        dummy variables (a vector of ones).

    Returns
    -------
    seasonal_dummies : ndarray (len_endog x n_seasons-1)
    """
    pass