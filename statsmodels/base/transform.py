import numpy as np
from statsmodels.robust import mad
from scipy.optimize import minimize_scalar

class BoxCox:
    """
    Mixin class to allow for a Box-Cox transformation.
    """

    def transform_boxcox(self, x, lmbda=None, method='guerrero', **kwargs):
        """
        Performs a Box-Cox transformation on the data array x. If lmbda is None,
        the indicated method is used to estimate a suitable lambda parameter.

        Parameters
        ----------
        x : array_like
        lmbda : float
            The lambda parameter for the Box-Cox transform. If None, a value
            will be estimated by means of the specified method.
        method : {'guerrero', 'loglik'}
            The method to estimate the lambda parameter. Will only be used if
            lmbda is None, and defaults to 'guerrero', detailed in Guerrero
            (1993). 'loglik' maximizes the profile likelihood.
        **kwargs
            Options for the specified method.
            * For 'guerrero', this entails window_length, the grouping
              parameter, scale, the dispersion measure, and options, to be
              passed to the optimizer.
            * For 'loglik': options, to be passed to the optimizer.

        Returns
        -------
        y : array_like
            The transformed series.
        lmbda : float
            The lmbda parameter used to transform the series.

        References
        ----------
        Guerrero, Victor M. 1993. "Time-series analysis supported by power
        transformations". `Journal of Forecasting`. 12 (1): 37-48.

        Guerrero, Victor M. and Perera, Rafael. 2004. "Variance Stabilizing
        Power Transformation for Time Series," `Journal of Modern Applied
        Statistical Methods`. 3 (2): 357-369.

        Box, G. E. P., and D. R. Cox. 1964. "An Analysis of Transformations".
        `Journal of the Royal Statistical Society`. 26 (2): 211-252.
        """
        x = np.asarray(x)
        if lmbda is None:
            lmbda = self._est_lambda(x, method=method, **kwargs)
        
        if lmbda == 0:
            y = np.log(x)
        else:
            y = (x**lmbda - 1) / lmbda
        
        return y, lmbda

    def untransform_boxcox(self, x, lmbda, method='naive'):
        """
        Back-transforms the Box-Cox transformed data array, by means of the
        indicated method. The provided argument lmbda should be the lambda
        parameter that was used to initially transform the data.

        Parameters
        ----------
        x : array_like
            The transformed series.
        lmbda : float
            The lambda parameter that was used to transform the series.
        method : {'naive'}
            Indicates the method to be used in the untransformation. Defaults
            to 'naive', which reverses the transformation.

            NOTE: 'naive' is implemented natively, while other methods may be
            available in subclasses!

        Returns
        -------
        y : array_like
            The untransformed series.
        """
        x = np.asarray(x)
        if method != 'naive':
            raise ValueError("Only 'naive' method is currently implemented.")
        
        if lmbda == 0:
            y = np.exp(x)
        else:
            y = (x * lmbda + 1) ** (1 / lmbda)
        
        return y

    def _est_lambda(self, x, bounds=(-1, 2), method='guerrero', **kwargs):
        """
        Computes an estimate for the lambda parameter in the Box-Cox
        transformation using method.

        Parameters
        ----------
        x : array_like
            The untransformed data.
        bounds : tuple
            Numeric 2-tuple, that indicate the solution space for the lambda
            parameter. Default (-1, 2).
        method : {'guerrero', 'loglik'}
            The method by which to estimate lambda. Defaults to 'guerrero', but
            the profile likelihood ('loglik') is also available.
        **kwargs
            Options for the specified method.
            * For 'guerrero': window_length (int), the seasonality/grouping
              parameter. Scale ({'mad', 'sd'}), the dispersion measure. Options
              (dict), to be passed to the optimizer.
            * For 'loglik': Options (dict), to be passed to the optimizer.

        Returns
        -------
        lmbda : float
            The lambda parameter.
        """
        if method == 'guerrero':
            return self._guerrero_cv(x, bounds, **kwargs)
        elif method == 'loglik':
            return self._loglik_boxcox(x, bounds, **kwargs)
        else:
            raise ValueError("Method must be either 'guerrero' or 'loglik'")

    def _guerrero_cv(self, x, bounds, window_length=4, scale='sd', options={'maxiter': 25}):
        """
        Computes lambda using guerrero's coefficient of variation. If no
        seasonality is present in the data, window_length is set to 4 (as
        per Guerrero and Perera, (2004)).

        NOTE: Seasonality-specific auxiliaries *should* provide their own
        seasonality parameter.

        Parameters
        ----------
        x : array_like
        bounds : tuple
            Numeric 2-tuple, that indicate the solution space for the lambda
            parameter.
        window_length : int
            Seasonality/grouping parameter. Default 4, as per Guerrero and
            Perera (2004). NOTE: this indicates the length of the individual
            groups, not the total number of groups!
        scale : {'sd', 'mad'}
            The dispersion measure to be used. 'sd' indicates the sample
            standard deviation, but the more robust 'mad' is also available.
        options : dict
            The options (as a dict) to be passed to the optimizer.
        """
        x = np.asarray(x)
        n = len(x)
        k = n // window_length

        def cv(lmbda):
            y, _ = self.transform_boxcox(x, lmbda)
            y_grouped = y[:k*window_length].reshape(k, window_length)
            if scale == 'sd':
                dispersion = np.std(y_grouped, axis=1, ddof=1)
            elif scale == 'mad':
                dispersion = mad(y_grouped, axis=1)
            else:
                raise ValueError("Scale must be either 'sd' or 'mad'")
            return np.std(dispersion) / np.mean(dispersion)

        result = minimize_scalar(cv, bounds=bounds, method='bounded', options=options)
        return result.x

    def _loglik_boxcox(self, x, bounds, options={'maxiter': 25}):
        """
        Taken from the Stata manual on Box-Cox regressions, where this is the
        special case of 'lhs only'. As an estimator for the variance, the
        sample variance is used, by means of the well-known formula.

        Parameters
        ----------
        x : array_like
        options : dict
            The options (as a dict) to be passed to the optimizer.
        """
        x = np.asarray(x)
        n = len(x)

        def loglik(lmbda):
            y, _ = self.transform_boxcox(x, lmbda)
            s2 = np.var(y, ddof=1)
            loglik = -n/2 * np.log(s2) + (lmbda - 1) * np.sum(np.log(x))
            return -loglik  # Minimize negative log-likelihood

        result = minimize_scalar(loglik, bounds=bounds, method='bounded', options=options)
        return result.x
