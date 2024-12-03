"""
Statistical tools for time series analysis
"""
from __future__ import annotations
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import Literal, lzip
from statsmodels.compat.scipy import _next_regular
from typing import Union, List
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import CollinearityWarning, InfeasibleTestError, InterpolationWarning, MissingDataError, ValueWarning
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import array_like, bool_like, dict_like, float_like, int_like, string_like
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
ArrayLike1D = Union[np.ndarray, pd.Series, List[float]]
__all__ = ['acovf', 'acf', 'pacf', 'pacf_yw', 'pacf_ols', 'ccovf', 'ccf', 'q_stat', 'coint', 'arma_order_select_ic', 'adfuller', 'kpss', 'bds', 'pacf_burg', 'innovations_algo', 'innovations_filter', 'levinson_durbin_pacf', 'levinson_durbin', 'zivot_andrews', 'range_unit_root_test']
SQRTEPS = np.sqrt(np.finfo(np.double).eps)

def _autolag(mod, endog, exog, startlag, maxlag, method, modargs=(), fitargs=(), regresults=False):
    """
    Returns the results for the lag length that maximizes the info criterion.

    Parameters
    ----------
    mod : Model class
        Model estimator class
    endog : array_like
        nobs array containing endogenous variable
    exog : array_like
        nobs by (startlag + maxlag) array containing lags and possibly other
        variables
    startlag : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : {"aic", "bic", "t-stat"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag
    modargs : tuple, optional
        args to pass to model.  See notes.
    fitargs : tuple, optional
        args to pass to fit.  See notes.
    regresults : bool, optional
        Flag indicating to return optional return results

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that maximizes the information criterion.
    results : dict, optional
        Dictionary containing all estimation results

    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    results = {}
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit(*fitargs)

    if method == "aic":
        criteria = [r.aic for r in results.values()]
    elif method == "bic":
        criteria = [r.bic for r in results.values()]
    elif method == "t-stat":
        criteria = [abs(r.tvalues[-1]) for r in results.values()]
    else:
        raise ValueError("method must be 'aic', 'bic', or 't-stat'")

    icbest, bestlag = max(zip(criteria, results.keys()))

    if regresults:
        return icbest, bestlag, results
    else:
        return icbest, bestlag

def adfuller(x, maxlag: int | None=None, regression='c', autolag='AIC', store=False, regresults=False):
    """
    Augmented Dickey-Fuller unit root test.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    maxlag : {None, int}
        Maximum lag which is included in test, default value of
        12*(nobs/100)^{1/4} is used when ``None``.
    regression : {"c","ct","ctt","n"}
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "n" : no constant, no trend.

    autolag : {"AIC", "BIC", "t-stat", None}
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.

        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
        * If None, then the number of included lags is set to maxlag.
    store : bool
        If True, then a result instance is returned additionally to
        the adf statistic. Default is False.
    regresults : bool, optional
        If True, the full regression results are returned. Default is False.

    Returns
    -------
    adf : float
        The test statistic.
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
    usedlag : int
        The number of lags used.
    nobs : int
        The number of observations used for the ADF regression and calculation
        of the critical values.
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010).
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes.

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    The autolag option and maxlag for it are described in Greene.

    See the notebook `Stationarity and detrending (ADF/KPSS)
    <../examples/notebooks/generated/stationarity_detrending_adf_kpss.html>`__
    for an overview.

    References
    ----------
    .. [1] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

    .. [2] Hamilton, J.D.  "Time Series Analysis".  Princeton, 1994.

    .. [3] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    .. [4] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen"s
        University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    from statsmodels.regression.linear_model import OLS

    x = np.asarray(x)
    nobs = x.shape[0]

    if maxlag is None:
        maxlag = int(np.ceil(12 * np.power(nobs / 100, 1 / 4)))

    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim='both', original='in')
    nobs = xdall.shape[0]

    xdall[:, 0] = x[-nobs - 1:-1]  # replace 0 xdiff with level of x
    xdshort = xdall[:, 0]

    if regression != 'n':
        fullregs = add_trend(xdall[:, 1:], regression, prepend=True)
    else:
        fullregs = xdall[:, 1:]

    startlag = fullregs.shape[1] - xdall.shape[1] + 1

    # Select lag length
    if isinstance(autolag, str):
        icbest, bestlag = _autolag(OLS, xdshort, fullregs, startlag,
                                   maxlag, autolag)
    elif autolag is None:
        bestlag = maxlag
        icbest = None
    else:
        raise ValueError('autolag must be either a string or None')

    # Compute ADF test statistic
    bestmod = OLS(xdshort, add_trend(xdall[:, :bestlag + 1], regression)).fit()
    adf = bestmod.tvalues[0]

    # Get approx p-value and critical values
    pvalue = mackinnonp(adf, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {"1%" : critvalues[0], "5%" : critvalues[1], "10%" : critvalues[2]}

    if store:
        from statsmodels.tools.sm_exceptions import ResultsStore
        resstore = ResultsStore()
        resstore.adf = adf
        resstore.pvalue = pvalue
        resstore.usedlag = bestlag
        resstore.nobs = nobs
        resstore.critical_values = critvalues
        resstore.icbest = icbest
        return adf, pvalue, bestlag, nobs, critvalues, icbest, resstore
    else:
        if not regresults:
            return adf, pvalue, bestlag, nobs, critvalues, icbest
        else:
            return adf, pvalue, bestlag, nobs, critvalues, icbest, bestmod

@deprecate_kwarg('unbiased', 'adjusted')
def acovf(x, adjusted=False, demean=True, fft=True, missing='none', nlag=None):
    """
    Estimate autocovariances.

    Parameters
    ----------
    x : array_like
        Time series data. Must be 1d.
    adjusted : bool, default False
        If True, then denominators is n-k, otherwise n.
    demean : bool, default True
        If True, then subtract the mean x from each element of x.
    fft : bool, default True
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    missing : str, default "none"
        A string in ["none", "raise", "conservative", "drop"] specifying how
        the NaNs are to be treated. "none" performs no checks. "raise" raises
        an exception if NaN values are found. "drop" removes the missing
        observations and then estimates the autocovariances treating the
        non-missing as contiguous. "conservative" computes the autocovariance
        using nan-ops so that nans are removed when computing the mean
        and cross-products that are used to estimate the autocovariance.
        When using "conservative", n is set to the number of non-missing
        observations.
    nlag : {int, None}, default None
        Limit the number of autocovariances returned.  Size of returned
        array is nlag + 1.  Setting nlag when fft is False uses a simple,
        direct estimator of the autocovariances that only computes the first
        nlag + 1 values. This can be much faster when the time series is long
        and only a small number of autocovariances are needed.

    Returns
    -------
    ndarray
        The estimated autocovariances.

    References
    ----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
           and amplitude modulation. Sankhya: The Indian Journal of
           Statistics, Series A, pp.383-392.
    """
    x = np.array(x)
    if x.ndim > 1:
        raise ValueError("x must be 1d")

    if missing != 'none':
        mask = np.isnan(x)
        if missing == 'raise' and np.any(mask):
            raise MissingDataError("NaNs found in input array")
        elif missing == 'drop':
            x = x[~mask]
        elif missing == 'conservative':
            x = ma.array(x, mask=mask)
        else:
            raise ValueError("missing option %s not understood" % missing)

    if demean:
        x = x - x.mean()

    n = len(x)
    if nlag is None:
        nlag = n - 1

    if adjusted:
        d = np.arange(1, nlag + 1)
        acov = np.correlate(x, x, 'full')[n - 1:] / (n - d)
    elif fft:
        nobs = len(x)
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(x, n=n)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / nobs
        acov = acov.real
    else:
        acov = np.correlate(x, x, 'full')[n - 1:] / n

    return acov[:nlag + 1]

def q_stat(x, nobs):
    """
    Compute Ljung-Box Q Statistic.

    Parameters
    ----------
    x : array_like
        Array of autocorrelation coefficients.  Can be obtained from acf.
    nobs : int, optional
        Number of observations in the entire sample (ie., not just the length
        of the autocorrelation function results.

    Returns
    -------
    q-stat : ndarray
        Ljung-Box Q-statistic for autocorrelation parameters.
    p-value : ndarray
        P-value of the Q statistic.

    See Also
    --------
    statsmodels.stats.diagnostic.acorr_ljungbox
        Ljung-Box Q-test for autocorrelation in time series based
        on a time series rather than the estimated autocorrelation
        function.

    Notes
    -----
    Designed to be used with acf.
    """
    pass

def acf(x, adjusted=False, nlags=None, qstat=False, fft=True, alpha=None, bartlett_confint=True, missing='none'):
    """
    Calculate the autocorrelation function.

    Parameters
    ----------
    x : array_like
       The time series data.
    adjusted : bool, default False
       If True, then denominators for autocovariance are n-k, otherwise n.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1). The returned value
        includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
    qstat : bool, default False
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, default True
        If True, computes the ACF via FFT.
    alpha : scalar, default None
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett"s formula.
    bartlett_confint : bool, default True
        Confidence intervals for ACF values are generally placed at 2
        standard errors around r_k. The formula used for standard error
        depends upon the situation. If the autocorrelations are being used
        to test for randomness of residuals as part of the ARIMA routine,
        the standard errors are determined assuming the residuals are white
        noise. The approximate formula for any lag is that standard error
        of each r_k = 1/sqrt(N). See section 9.4 of [2] for more details on
        the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2
        in [3].
        For the ACF of raw data, the standard error at a lag k is
        found as if the right model was an MA(k-1). This allows the possible
        interpretation that if all autocorrelations past a certain lag are
        within the limits, the model might be an MA of order defined by the
        last significant autocorrelation. In this case, a moving average
        model is assumed for the data and the standard errors for the
        confidence intervals should be generated using Bartlett's formula.
        For more details on Bartlett formula result, see section 7.2 in [2].
    missing : str, default "none"
        A string in ["none", "raise", "conservative", "drop"] specifying how
        the NaNs are to be treated. "none" performs no checks. "raise" raises
        an exception if NaN values are found. "drop" removes the missing
        observations and then estimates the autocovariances treating the
        non-missing as contiguous. "conservative" computes the autocovariance
        using nan-ops so that nans are removed when computing the mean
        and cross-products that are used to estimate the autocovariance.
        When using "conservative", n is set to the number of non-missing
        observations.

    Returns
    -------
    acf : ndarray
        The autocorrelation function for lags 0, 1, ..., nlags. Shape
        (nlags+1,).
    confint : ndarray, optional
        Confidence intervals for the ACF at lags 0, 1, ..., nlags. Shape
        (nlags + 1, 2). Returned if alpha is not None.
    qstat : ndarray, optional
        The Ljung-Box Q-Statistic for lags 1, 2, ..., nlags (excludes lag
        zero). Returned if q_stat is True.
    pvalues : ndarray, optional
        The p-values associated with the Q-statistics for lags 1, 2, ...,
        nlags (excludes lag zero). Returned if q_stat is True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    For very long time series it is recommended to use fft convolution instead.
    When fft is False uses a simple, direct estimator of the autocovariances
    that only computes the first nlag + 1 values. This can be much faster when
    the time series is long and only a small number of autocovariances are
    needed.

    If adjusted is true, the denominator for the autocovariance is adjusted
    for the loss of data.

    References
    ----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
       and amplitude modulation. Sankhya: The Indian Journal of
       Statistics, Series A, pp.383-392.
    .. [2] Brockwell and Davis, 1987. Time Series Theory and Methods
    .. [3] Brockwell and Davis, 2010. Introduction to Time Series and
       Forecasting, 2nd edition.
    """
    pass

def pacf_yw(x: ArrayLike1D, nlags: int | None=None, method: Literal['adjusted', 'mle']='adjusted') -> np.ndarray:
    """
    Partial autocorrelation estimated with non-recursive yule_walker.

    Parameters
    ----------
    x : array_like
        The observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    method : {"adjusted", "mle"}, default "adjusted"
        The method for the autocovariance calculations in yule walker.

    Returns
    -------
    ndarray
        The partial autocorrelations, maxlag+1 elements.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg"s method.

    Notes
    -----
    This solves yule_walker for each desired lag and contains
    currently duplicate calculations.
    """
    pass

def pacf_burg(x: ArrayLike1D, nlags: int | None=None, demean: bool=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Burg"s partial autocorrelation estimator.

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    demean : bool, optional
        Flag indicating to demean that data. Set to False if x has been
        previously demeaned.

    Returns
    -------
    pacf : ndarray
        Partial autocorrelations for lags 0, 1, ..., nlag.
    sigma2 : ndarray
        Residual variance estimates where the value in position m is the
        residual variance in an AR model that includes m lags.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
         Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.
    """
    pass

@deprecate_kwarg('unbiased', 'adjusted')
def pacf_ols(x: ArrayLike1D, nlags: int | None=None, efficient: bool=True, adjusted: bool=False) -> np.ndarray:
    """
    Calculate partial autocorrelations via OLS.

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    efficient : bool, optional
        If true, uses the maximum number of available observations to compute
        each partial autocorrelation. If not, uses the same number of
        observations to compute all pacf values.
    adjusted : bool, optional
        Adjust each partial autocorrelation by n / (n - lag).

    Returns
    -------
    ndarray
        The partial autocorrelations, (maxlag,) array corresponding to lags
        0, 1, ..., maxlag.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
         Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg"s method.

    Notes
    -----
    This solves a separate OLS estimation for each desired lag using method in
    [1]_. Setting efficient to True has two effects. First, it uses
    `nobs - lag` observations of estimate each pacf.  Second, it re-estimates
    the mean in each regression. If efficient is False, then the data are first
    demeaned, and then `nobs - maxlag` observations are used to estimate each
    partial autocorrelation.

    The inefficient estimator appears to have better finite sample properties.
    This option should only be used in time series that are covariance
    stationary.

    OLS estimation of the pacf does not guarantee that all pacf values are
    between -1 and 1.

    References
    ----------
    .. [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
       Time series analysis: forecasting and control. John Wiley & Sons, p. 66
    """
    pass

def pacf(x: ArrayLike1D, nlags: int | None=None, method: Literal['yw', 'ywadjusted', 'ols', 'ols-inefficient', 'ols-adjusted', 'ywm', 'ywmle', 'ld', 'ldadjusted', 'ldb', 'ldbiased', 'burg']='ywadjusted', alpha: float | None=None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Partial autocorrelation estimate.

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs // 2 - 1). The returned value
        includes lag 0 (ie., 1) so size of the pacf vector is (nlags + 1,).
    method : str, default "ywunbiased"
        Specifies which method for the calculations to use.

        - "yw" or "ywadjusted" : Yule-Walker with sample-size adjustment in
          denominator for acovf. Default.
        - "ywm" or "ywmle" : Yule-Walker without adjustment.
        - "ols" : regression of time series on lags of it and on constant.
        - "ols-inefficient" : regression of time series on lags using a single
          common sample to estimate all pacf coefficients.
        - "ols-adjusted" : regression of time series on lags with a bias
          adjustment.
        - "ld" or "ldadjusted" : Levinson-Durbin recursion with bias
          correction.
        - "ldb" or "ldbiased" : Levinson-Durbin recursion without bias
          correction.
        - "burg" :  Burg"s partial autocorrelation estimator.

    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x)).

    Returns
    -------
    pacf : ndarray
        The partial autocorrelations for lags 0, 1, ..., nlags. Shape
        (nlags+1,).
    confint : ndarray, optional
        Confidence intervals for the PACF at lags 0, 1, ..., nlags. Shape
        (nlags + 1, 2). Returned if alpha is not None.

    See Also
    --------
    statsmodels.tsa.stattools.acf
        Estimate the autocorrelation function.
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
         Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg"s method.

    Notes
    -----
    Based on simulation evidence across a range of low-order ARMA models,
    the best methods based on root MSE are Yule-Walker (MLW), Levinson-Durbin
    (MLE) and Burg, respectively. The estimators with the lowest bias included
    included these three in addition to OLS and OLS-adjusted.

    Yule-Walker (adjusted) and Levinson-Durbin (adjusted) performed
    consistently worse than the other options.
    """
    pass

@deprecate_kwarg('unbiased', 'adjusted')
def ccovf(x, y, adjusted=True, demean=True, fft=True):
    """
    Calculate the cross-covariance between two series.

    Parameters
    ----------
    x, y : array_like
       The time series data to use in the calculation.
    adjusted : bool, optional
       If True, then denominators for cross-covariance are n-k, otherwise n.
    demean : bool, optional
        Flag indicating whether to demean x and y.
    fft : bool, default True
        If True, use FFT convolution.  This method should be preferred
        for long time series.

    Returns
    -------
    ndarray
        The estimated cross-covariance function: the element at index k
        is the covariance between {x[k], x[k+1], ..., x[n]} and {y[0], y[1], ..., y[m-k]},
        where n and m are the lengths of x and y, respectively.
    """
    pass

@deprecate_kwarg('unbiased', 'adjusted')
def ccf(x, y, adjusted=True, fft=True, *, nlags=None, alpha=None):
    """
    The cross-correlation function.

    Parameters
    ----------
    x, y : array_like
        The time series data to use in the calculation.
    adjusted : bool
        If True, then denominators for cross-correlation are n-k, otherwise n.
    fft : bool, default True
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    nlags : int, optional
        Number of lags to return cross-correlations for. If not provided,
        the number of lags equals len(x).
    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x)).

    Returns
    -------
    ndarray
        The cross-correlation function of x and y: the element at index k
        is the correlation between {x[k], x[k+1], ..., x[n]} and {y[0], y[1], ..., y[m-k]},
        where n and m are the lengths of x and y, respectively.
    confint : ndarray, optional
        Confidence intervals for the CCF at lags 0, 1, ..., nlags-1 using the level given by
        alpha and the standard deviation calculated as 1/sqrt(len(x)) [1]. Shape (nlags, 2).
        Returned if alpha is not None.

    Notes
    -----
    If adjusted is True, the denominator for the cross-correlation is adjusted.

    References
    ----------
    .. [1] Brockwell and Davis, 2016. Introduction to Time Series and
       Forecasting, 3rd edition, p. 242.
    """
    pass

def levinson_durbin(s, nlags=10, isacov=False):
    """
    Levinson-Durbin recursion for autoregressive processes.

    Parameters
    ----------
    s : array_like
        If isacov is False, then this is the time series. If iasacov is true
        then this is interpreted as autocovariance starting with lag 0.
    nlags : int, optional
        The largest lag to include in recursion or order of the autoregressive
        process.
    isacov : bool, optional
        Flag indicating whether the first argument, s, contains the
        autocovariances or the data series.

    Returns
    -------
    sigma_v : float
        The estimate of the error variance.
    arcoefs : ndarray
        The estimate of the autoregressive coefficients for a model including
        nlags.
    pacf : ndarray
        The partial autocorrelation function.
    sigma : ndarray
        The entire sigma array from intermediate result, last value is sigma_v.
    phi : ndarray
        The entire phi array from intermediate result, last column contains
        autoregressive coefficients for AR(nlags).

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the
    sample autocovariance function is calculated with the default options
    (biased, no fft).
    """
    pass

def levinson_durbin_pacf(pacf, nlags=None):
    """
    Levinson-Durbin algorithm that returns the acf and ar coefficients.

    Parameters
    ----------
    pacf : array_like
        Partial autocorrelation array for lags 0, 1, ... p.
    nlags : int, optional
        Number of lags in the AR model.  If omitted, returns coefficients from
        an AR(p) and the first p autocorrelations.

    Returns
    -------
    arcoefs : ndarray
        AR coefficients computed from the partial autocorrelations.
    acf : ndarray
        The acf computed from the partial autocorrelations. Array returned
        contains the autocorrelations corresponding to lags 0, 1, ..., p.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.
    """
    pass

def breakvar_heteroskedasticity_test(resid, subset_length=1 / 3, alternative='two-sided', use_f=True):
    """
    Test for heteroskedasticity of residuals

    Tests whether the sum-of-squares in the first subset of the sample is
    significantly different than the sum-of-squares in the last subset
    of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
    is of no heteroskedasticity.

    Parameters
    ----------
    resid : array_like
        Residuals of a time series model.
        The shape is 1d (nobs,) or 2d (nobs, nvars).
    subset_length : {int, float}
        Length of the subsets to test (h in Notes below).
        If a float in 0 < subset_length < 1, it is interpreted as fraction.
        Default is 1/3.
    alternative : str, 'increasing', 'decreasing' or 'two-sided'
        This specifies the alternative for the p-value calculation. Default
        is two-sided.
    use_f : bool, optional
        Whether or not to compare against the asymptotic distribution
        (chi-squared) or the approximate small-sample distribution (F).
        Default is True (i.e. default is to compare against an F
        distribution).

    Returns
    -------
    test_statistic : {float, ndarray}
        Test statistic(s) H(h).
    p_value : {float, ndarray}
        p-value(s) of test statistic(s).

    Notes
    -----
    The null hypothesis is of no heteroskedasticity. That means different
    things depending on which alternative is selected:

    - Increasing: Null hypothesis is that the variance is not increasing
        throughout the sample; that the sum-of-squares in the later
        subsample is *not* greater than the sum-of-squares in the earlier
        subsample.
    - Decreasing: Null hypothesis is that the variance is not decreasing
        throughout the sample; that the sum-of-squares in the earlier
        subsample is *not* greater than the sum-of-squares in the later
        subsample.
    - Two-sided: Null hypothesis is that the variance is not changing
        throughout the sample. Both that the sum-of-squares in the earlier
        subsample is not greater than the sum-of-squares in the later
        subsample *and* that the sum-of-squares in the later subsample is
        not greater than the sum-of-squares in the earlier subsample.

    For :math:`h = [T/3]`, the test statistic is:

    .. math::

        H(h) = \\sum_{t=T-h+1}^T  \\tilde v_t^2
        \\Bigg / \\sum_{t=1}^{h} \\tilde v_t^2

    This statistic can be tested against an :math:`F(h,h)` distribution.
    Alternatively, :math:`h H(h)` is asymptotically distributed according
    to :math:`\\chi_h^2`; this second test can be applied by passing
    `use_f=False` as an argument.

    See section 5.4 of [1]_ for the above formula and discussion, as well
    as additional details.

    References
    ----------
    .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
            *Models and the Kalman Filter.* Cambridge University Press.
    """
    pass

def grangercausalitytests(x, maxlag, addconst=True, verbose=None):
    """
    Four tests for granger non causality of 2 time series.

    All four tests give similar results. `params_ftest` and `ssr_ftest` are
    equivalent based on F test which is identical to lmtest:grangertest in R.

    Parameters
    ----------
    x : array_like
        The data for testing whether the time series in the second column Granger
        causes the time series in the first column. Missing values are not
        supported.
    maxlag : {int, Iterable[int]}
        If an integer, computes the test for all lags up to maxlag. If an
        iterable, computes the tests only for the lags in maxlag.
    addconst : bool
        Include a constant in the model.
    verbose : bool
        Print results. Deprecated

        .. deprecated: 0.14

           verbose is deprecated and will be removed after 0.15 is released



    Returns
    -------
    dict
        All test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        test statistic, pvalues, degrees of freedom, the second element are
        the OLS estimation results for the restricted model, the unrestricted
        model and the restriction (contrast) matrix for the parameter f_test.

    Notes
    -----
    TODO: convert to class and attach results properly

    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the pvalues are below a desired size
    of the test.

    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.

    `params_ftest`, `ssr_ftest` are based on F distribution

    `ssr_chi2test`, `lrtest` are based on chi-square distribution

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Granger_causality

    .. [2] Greene: Econometric Analysis

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.tsa.stattools import grangercausalitytests
    >>> import numpy as np
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> data = data.data[["realgdp", "realcons"]].pct_change().dropna()

    All lags up to 4

    >>> gc_res = grangercausalitytests(data, 4)

    Only lag 4

    >>> gc_res = grangercausalitytests(data, [4])
    """
    pass

def coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag: str | None='aic', return_results=None):
    """
    Test for no-cointegration of a univariate equation.

    The null hypothesis is no cointegration. Variables in y0 and y1 are
    assumed to be integrated of order 1, I(1).

    This uses the augmented Engle-Granger two-step cointegration test.
    Constant or trend is included in 1st stage regression, i.e. in
    cointegrating equation.

    **Warning:** The autolag default has changed compared to statsmodels 0.8.
    In 0.8 autolag was always None, no the keyword is used and defaults to
    "aic". Use `autolag=None` to avoid the lag search.

    Parameters
    ----------
    y0 : array_like
        The first element in cointegrated system. Must be 1-d.
    y1 : array_like
        The remaining elements in cointegrated system.
    trend : str {"c", "ct"}
        The trend term included in regression for cointegrating equation.

        * "c" : constant.
        * "ct" : constant and linear trend.
        * also available quadratic trend "ctt", and no constant "n".

    method : {"aeg"}
        Only "aeg" (augmented Engle-Granger) is available.
    maxlag : None or int
        Argument for `adfuller`, largest or given number of lags.
    autolag : str
        Argument for `adfuller`, lag selection criterion.

        * If None, then maxlag lags are used without lag search.
        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
    return_results : bool
        For future compatibility, currently only tuple available.
        If True, then a results instance is returned. Otherwise, a tuple
        with the test outcome is returned. Set `return_results=False` to
        avoid future changes in return.

    Returns
    -------
    coint_t : float
        The t-statistic of unit-root test on residuals.
    pvalue : float
        MacKinnon"s approximate, asymptotic p-value based on MacKinnon (1994).
    crit_value : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels based on regression curve. This depends on the number of
        observations.

    Notes
    -----
    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue is
    small, below a critical size, then we can reject the hypothesis that there
    is no cointegrating relationship.

    P-values and critical values are obtained through regression surface
    approximation from MacKinnon 1994 and 2010.

    If the two series are almost perfectly collinear, then computing the
    test is numerically unstable. However, the two series will be cointegrated
    under the maintained assumption that they are integrated. In this case
    the t-statistic will be set to -inf and the pvalue to zero.

    TODO: We could handle gaps in data by dropping rows with nans in the
    Auxiliary regressions. Not implemented yet, currently assumes no nans
    and no gaps in time series.

    References
    ----------
    .. [1] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions
       for Unit-Root and Cointegration Tests." Journal of Business & Economics
       Statistics, 12.2, 167-76.
    .. [2] MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
       Queen"s University, Dept of Economics Working Papers 1227.
       http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    pass

def arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c', model_kw=None, fit_kw=None):
    """
    Compute information criteria for many ARMA models.

    Parameters
    ----------
    y : array_like
        Array of time-series data.
    max_ar : int
        Maximum number of AR lags to use. Default 4.
    max_ma : int
        Maximum number of MA lags to use. Default 2.
    ic : str, list
        Information criteria to report. Either a single string or a list
        of different criteria is possible.
    trend : str
        The trend to use when fitting the ARMA models.
    model_kw : dict
        Keyword arguments to be passed to the ``ARMA`` model.
    fit_kw : dict
        Keyword arguments to be passed to ``ARMA.fit``.

    Returns
    -------
    Bunch
        Dict-like object with attribute access. Each ic is an attribute with a
        DataFrame for the results. The AR order used is the row index. The ma
        order used is the column index. The minimum orders are available as
        ``ic_min_order``.

    Notes
    -----
    This method can be used to tentatively identify the order of an ARMA
    process, provided that the time series is stationary and invertible. This
    function computes the full exact MLE estimate of each model and can be,
    therefore a little slow. An implementation using approximate estimates
    will be provided in the future. In the meantime, consider passing
    {method : "css"} to fit_kw.

    Examples
    --------

    >>> from statsmodels.tsa.arima_process import arma_generate_sample
    >>> import statsmodels.api as sm
    >>> import numpy as np

    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> arparams = np.r_[1, -arparams]
    >>> maparam = np.r_[1, maparams]
    >>> nobs = 250
    >>> np.random.seed(2014)
    >>> y = arma_generate_sample(arparams, maparams, nobs)
    >>> res = sm.tsa.arma_order_select_ic(y, ic=["aic", "bic"], trend="n")
    >>> res.aic_min_order
    >>> res.bic_min_order
    """
    pass

def has_missing(data):
    """
    Returns True if "data" contains missing entries, otherwise False
    """
    pass

def kpss(x, regression: Literal['c', 'ct']='c', nlags: Literal['auto', 'legacy'] | int='auto', store: bool=False) -> tuple[float, float, int, dict[str, float]]:
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

    Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null
    hypothesis that x is level or trend stationary.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    regression : str{"c", "ct"}
        The null hypothesis for the KPSS test.

        * "c" : The data is stationary around a constant (default).
        * "ct" : The data is stationary around a trend.
    nlags : {str, int}, optional
        Indicates the number of lags to be used. If "auto" (default), lags
        is calculated using the data-dependent method of Hobijn et al. (1998).
        See also Andrews (1991), Newey & West (1994), and Schwert (1989). If
        set to "legacy",  uses int(12 * (n / 100)**(1 / 4)) , as outlined in
        Schwert (1989).
    store : bool
        If True, then a result instance is returned additionally to
        the KPSS statistic (default is False).

    Returns
    -------
    kpss_stat : float
        The KPSS test statistic.
    p_value : float
        The p-value of the test. The p-value is interpolated from
        Table 1 in Kwiatkowski et al. (1992), and a boundary point
        is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the
        interval (0.01, 0.1).
    lags : int
        The truncation lag parameter.
    crit : dict
        The critical values at 10%, 5%, 2.5% and 1%. Based on
        Kwiatkowski et al. (1992).
    resstore : (optional) instance of ResultStore
        An instance of a dummy class with results attached as attributes.

    Notes
    -----
    To estimate sigma^2 the Newey-West estimator is used. If lags is "legacy",
    the truncation lag parameter is set to int(12 * (n / 100) ** (1 / 4)),
    as outlined in Schwert (1989). The p-values are interpolated from
    Table 1 of Kwiatkowski et al. (1992). If the computed statistic is
    outside the table of critical values, then a warning message is
    generated.

    Missing values are not handled.

    See the notebook `Stationarity and detrending (ADF/KPSS)
    <../examples/notebooks/generated/stationarity_detrending_adf_kpss.html>`__
    for an overview.

    References
    ----------
    .. [1] Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation
       consistent covariance matrix estimation. Econometrica, 59: 817-858.

    .. [2] Hobijn, B., Frances, B.H., & Ooms, M. (2004). Generalizations of the
       KPSS-test for stationarity. Statistica Neerlandica, 52: 483-502.

    .. [3] Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992).
       Testing the null hypothesis of stationarity against the alternative of a
       unit root. Journal of Econometrics, 54: 159-178.

    .. [4] Newey, W.K., & West, K.D. (1994). Automatic lag selection in
       covariance matrix estimation. Review of Economic Studies, 61: 631-653.

    .. [5] Schwert, G. W. (1989). Tests for unit roots: A Monte Carlo
       investigation. Journal of Business and Economic Statistics, 7 (2):
       147-159.
    """
    pass

def _sigma_est_kpss(resids, nobs, lags):
    """
    Computes equation 10, p. 164 of Kwiatkowski et al. (1992). This is the
    consistent estimator for the variance.
    """
    pass

def _kpss_autolag(resids, nobs):
    """
    Computes the number of lags for covariance matrix estimation in KPSS test
    using method of Hobijn et al (1998). See also Andrews (1991), Newey & West
    (1994), and Schwert (1989). Assumes Bartlett / Newey-West kernel.
    """
    pass

def range_unit_root_test(x, store=False):
    """
    Range unit-root test for stationarity.

    Computes the Range Unit-Root (RUR) test for the null
    hypothesis that x is stationary.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    store : bool
        If True, then a result instance is returned additionally to
        the RUR statistic (default is False).

    Returns
    -------
    rur_stat : float
        The RUR test statistic.
    p_value : float
        The p-value of the test. The p-value is interpolated from
        Table 1 in Aparicio et al. (2006), and a boundary point
        is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the
        interval (0.01, 0.1).
    crit : dict
        The critical values at 10%, 5%, 2.5% and 1%. Based on
        Aparicio et al. (2006).
    resstore : (optional) instance of ResultStore
        An instance of a dummy class with results attached as attributes.

    Notes
    -----
    The p-values are interpolated from
    Table 1 of Aparicio et al. (2006). If the computed statistic is
    outside the table of critical values, then a warning message is
    generated.

    Missing values are not handled.

    References
    ----------
    .. [1] Aparicio, F., Escribano A., Sipols, A.E. (2006). Range Unit-Root (RUR)
        tests: robust against nonlinearities, error distributions, structural breaks
        and outliers. Journal of Time Series Analysis, 27 (4): 545-576.
    """
    pass

class ZivotAndrewsUnitRoot:
    """
    Class wrapper for Zivot-Andrews structural-break unit-root test
    """

    def __init__(self):
        """
        Critical values for the three different models specified for the
        Zivot-Andrews unit-root test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using
        100,000 replications and 2000 data points.
        """
        self._za_critical_values = {}
        self._c = ((0.001, -6.78442), (0.1, -5.83192), (0.2, -5.68139), (0.3, -5.58461), (0.4, -5.51308), (0.5, -5.45043), (0.6, -5.39924), (0.7, -5.36023), (0.8, -5.33219), (0.9, -5.30294), (1.0, -5.27644), (2.5, -5.0334), (5.0, -4.81067), (7.5, -4.67636), (10.0, -4.56618), (12.5, -4.4813), (15.0, -4.40507), (17.5, -4.33947), (20.0, -4.28155), (22.5, -4.22683), (25.0, -4.1783), (27.5, -4.13101), (30.0, -4.08586), (32.5, -4.04455), (35.0, -4.0038), (37.5, -3.96144), (40.0, -3.92078), (42.5, -3.88178), (45.0, -3.84503), (47.5, -3.80549), (50.0, -3.77031), (52.5, -3.73209), (55.0, -3.696), (57.5, -3.65985), (60.0, -3.62126), (65.0, -3.5458), (70.0, -3.46848), (75.0, -3.38533), (80.0, -3.29112), (85.0, -3.17832), (90.0, -3.04165), (92.5, -2.95146), (95.0, -2.83179), (96.0, -2.76465), (97.0, -2.68624), (98.0, -2.57884), (99.0, -2.40044), (99.9, -1.88932))
        self._za_critical_values['c'] = np.asarray(self._c)
        self._t = ((0.001, -83.9094), (0.1, -13.8837), (0.2, -9.13205), (0.3, -6.32564), (0.4, -5.60803), (0.5, -5.38794), (0.6, -5.26585), (0.7, -5.18734), (0.8, -5.12756), (0.9, -5.07984), (1.0, -5.03421), (2.5, -4.65634), (5.0, -4.4058), (7.5, -4.25214), (10.0, -4.13678), (12.5, -4.03765), (15.0, -3.95185), (17.5, -3.87945), (20.0, -3.81295), (22.5, -3.75273), (25.0, -3.69836), (27.5, -3.64785), (30.0, -3.59819), (32.5, -3.55146), (35.0, -3.50522), (37.5, -3.45987), (40.0, -3.41672), (42.5, -3.37465), (45.0, -3.33394), (47.5, -3.29393), (50.0, -3.25316), (52.5, -3.21244), (55.0, -3.17124), (57.5, -3.13211), (60.0, -3.09204), (65.0, -3.01135), (70.0, -2.92897), (75.0, -2.83614), (80.0, -2.73893), (85.0, -2.6284), (90.0, -2.49611), (92.5, -2.41337), (95.0, -2.3082), (96.0, -2.25797), (97.0, -2.19648), (98.0, -2.1132), (99.0, -1.99138), (99.9, -1.67466))
        self._za_critical_values['t'] = np.asarray(self._t)
        self._ct = ((0.001, -38.178), (0.1, -6.43107), (0.2, -6.07279), (0.3, -5.95496), (0.4, -5.86254), (0.5, -5.77081), (0.6, -5.72541), (0.7, -5.68406), (0.8, -5.65163), (0.9, -5.60419), (1.0, -5.57556), (2.5, -5.29704), (5.0, -5.07332), (7.5, -4.93003), (10.0, -4.82668), (12.5, -4.73711), (15.0, -4.6602), (17.5, -4.5897), (20.0, -4.52855), (22.5, -4.471), (25.0, -4.42011), (27.5, -4.37387), (30.0, -4.32705), (32.5, -4.28126), (35.0, -4.23793), (37.5, -4.19822), (40.0, -4.158), (42.5, -4.11946), (45.0, -4.08064), (47.5, -4.04286), (50.0, -4.00489), (52.5, -3.96837), (55.0, -3.932), (57.5, -3.89496), (60.0, -3.85577), (65.0, -3.77795), (70.0, -3.69794), (75.0, -3.61852), (80.0, -3.52485), (85.0, -3.41665), (90.0, -3.28527), (92.5, -3.19724), (95.0, -3.08769), (96.0, -3.03088), (97.0, -2.96091), (98.0, -2.85581), (99.0, -2.71015), (99.9, -2.28767))
        self._za_critical_values['ct'] = np.asarray(self._ct)

    def _za_crit(self, stat, model='c'):
        """
        Linear interpolation for Zivot-Andrews p-values and critical values

        Parameters
        ----------
        stat : float
            The ZA test statistic
        model : {"c","t","ct"}
            The model used when computing the ZA statistic. "c" is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated ZA test statistic distribution
        """
        pass

    def _quick_ols(self, endog, exog):
        """
        Minimal implementation of LS estimator for internal use
        """
        pass

    def _format_regression_data(self, series, nobs, const, trend, cols, lags):
        """
        Create the endog/exog data for the auxiliary regressions
        from the original (standardized) series under test.
        """
        pass

    def _update_regression_exog(self, exog, regression, period, nobs, const, trend, cols, lags):
        """
        Update the exog array for the next regression.
        """
        pass

    def run(self, x, trim=0.15, maxlag=None, regression='c', autolag='AIC'):
        """
        Zivot-Andrews structural-break unit-root test.

        The Zivot-Andrews test tests for a unit root in a univariate process
        in the presence of serial correlation and a single structural break.

        Parameters
        ----------
        x : array_like
            The data series to test.
        trim : float
            The percentage of series at begin/end to exclude from break-period
            calculation in range [0, 0.333] (default=0.15).
        maxlag : int
            The maximum lag which is included in test, default is
            12*(nobs/100)^{1/4} (Schwert, 1989).
        regression : {"c","t","ct"}
            Constant and trend order to include in regression.

            * "c" : constant only (default).
            * "t" : trend only.
            * "ct" : constant and trend.
        autolag : {"AIC", "BIC", "t-stat", None}
            The method to select the lag length when using automatic selection.

            * if None, then maxlag lags are used,
            * if "AIC" (default) or "BIC", then the number of lags is chosen
              to minimize the corresponding information criterion,
            * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
              lag until the t-statistic on the last lag length is significant
              using a 5%-sized test.

        Returns
        -------
        zastat : float
            The test statistic.
        pvalue : float
            The pvalue based on MC-derived critical values.
        cvdict : dict
            The critical values for the test statistic at the 1%, 5%, and 10%
            levels.
        baselag : int
            The number of lags used for period regressions.
        bpidx : int
            The index of x corresponding to endogenously calculated break period
            with values in the range [0..nobs-1].

        Notes
        -----
        H0 = unit root with a single structural break

        Algorithm follows Baum (2004/2015) approximation to original
        Zivot-Andrews method. Rather than performing an autolag regression at
        each candidate break period (as per the original paper), a single
        autolag regression is run up-front on the base model (constant + trend
        with no dummies) to determine the best lag length. This lag length is
        then used for all subsequent break-period regressions. This results in
        significant run time reduction but also slightly more pessimistic test
        statistics than the original Zivot-Andrews method, although no attempt
        has been made to characterize the size/power trade-off.

        References
        ----------
        .. [1] Baum, C.F. (2004). ZANDREWS: Stata module to calculate
           Zivot-Andrews unit root test in presence of structural break,"
           Statistical Software Components S437301, Boston College Department
           of Economics, revised 2015.

        .. [2] Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo
           investigation. Journal of Business & Economic Statistics, 7:
           147-159.

        .. [3] Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the
           great crash, the oil-price shock, and the unit-root hypothesis.
           Journal of Business & Economic Studies, 10: 251-270.
        """
        pass

    def __call__(self, x, trim=0.15, maxlag=None, regression='c', autolag='AIC'):
        return self.run(x, trim=trim, maxlag=maxlag, regression=regression, autolag=autolag)
zivot_andrews = ZivotAndrewsUnitRoot()
zivot_andrews.__doc__ = zivot_andrews.run.__doc__
