"""

Created on Tue Dec 20 20:24:20 2011

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from statsmodels.regression.linear_model import OLS, GLS, WLS

class GLSHet2(GLS):
    """WLS with heteroscedasticity that depends on explanatory variables

    note: mixing GLS sigma and weights for heteroscedasticity might not make
    sense

    I think rewriting following the pattern of GLSAR is better
    stopping criteria: improve in GLSAR also, e.g. change in rho

    """

    def __init__(self, endog, exog, exog_var, sigma=None):
        self.exog_var = atleast_2dcols(exog_var)
        super(self.__class__, self).__init__(endog, exog, sigma=sigma)

class GLSHet(WLS):
    """
    A regression model with an estimated heteroscedasticity.

    A subclass of WLS, that additionally estimates the weight matrix as a
    function of additional explanatory variables.

    Parameters
    ----------
    endog : array_like
    exog : array_like
    exog_var : array_like, 1d or 2d
        regressors, explanatory variables for the variance
    weights : array_like or None
        If weights are given, then they are used in the first step estimation.
    link : link function or None
        If None, then the variance is assumed to be a linear combination of
        the exog_var. If given, then ...  not tested yet

    *extra attributes*

    history : dict
       contains the parameter estimates in both regression for each iteration

    result instance has

    results_residual_regression : OLS result instance
        result of heteroscedasticity estimation

    except for fit_iterative all methods are inherited from WLS.

    Notes
    -----
    GLSHet is considered to be experimental.

    `fit` is just standard WLS fit for fixed weights
    `fit_iterative` updates the estimate for weights, see its docstring

    The two alternative for handling heteroscedasticity in the data are to
    use heteroscedasticity robust standard errors or estimating the
    heteroscedasticity
    Estimating heteroscedasticity and using weighted least squares produces
    smaller confidence intervals for the estimated parameters then the
    heteroscedasticity robust standard errors if the heteroscedasticity is
    correctly specified. If the heteroscedasticity is incorrectly specified
    then the estimated covariance is inconsistent.

    Stock and Watson for example argue in favor of using OLS with
    heteroscedasticity robust standard errors instead of GLSHet sind we are
    seldom sure enough about the correct specification (in economics).

    GLSHet has asymptotically the same distribution as WLS if the true
    weights are know. In both cases the asymptotic distribution of the
    parameter estimates is the normal distribution.

    The assumption of the model:

    y = X*beta + u,
    with E(u) = 0, E(X*u)=0, var(u_i) = z_i*gamma
    or for vector of all observations Sigma = diag(Z*gamma)

    where
    y : endog (nobs)
    X : exog  (nobs, k_vars)
    Z : exog_var (nobs, k_vars2)
    beta, gamma estimated parameters

    If a link is specified, then the heteroscedasticity is

    var(u_i) = link.inverse(z_i*gamma), or
    link(var(u_i)) = z_i*gamma

    for example for log-linkg
    var(u_i) = exp(z_i*gamma)


    Usage : see example ....

    TODO: test link option
    """

    def __init__(self, endog, exog, exog_var=None, weights=None, link=None):
        self.exog_var = atleast_2dcols(exog_var)
        if weights is None:
            weights = np.ones(endog.shape)
        if link is not None:
            self.link = link
            self.linkinv = link.inverse
        else:
            self.link = lambda x: x
            self.linkinv = lambda x: x
        super(self.__class__, self).__init__(endog, exog, weights=weights)

    def iterative_fit(self, maxiter=3):
        """
        Perform an iterative two-step procedure to estimate a WLS model.

        The model is assumed to have heteroskedastic errors.
        The variance is estimated by OLS regression of the link transformed
        squared residuals on Z, i.e.::

           link(sigma_i) = x_i*gamma.

        Parameters
        ----------
        maxiter : int, optional
            the number of iterations

        Notes
        -----
        maxiter=1: returns the estimated based on given weights
        maxiter=2: performs a second estimation with the updated weights,
                   this is 2-step estimation
        maxiter>2: iteratively estimate and update the weights

        TODO: possible extension stop iteration if change in parameter
            estimates is smaller than x_tol

        Repeated calls to fit_iterative, will do one redundant pinv_wexog
        calculation. Calling fit_iterative(maxiter) ones does not do any
        redundant recalculations (whitening or calculating pinv_wexog).
        """
        self.history = {'params': [], 'weights': []}
        
        for iteration in range(maxiter):
            # Fit the model using current weights
            model = self.fit()
            self.history['params'].append(model.params)
            self.history['weights'].append(self.weights)
            
            if iteration == maxiter - 1:
                break
            
            # Estimate new weights based on residuals
            resid = model.resid
            exog_var = self.exog_var
            
            # Apply link function to squared residuals
            y = self.link(resid**2)
            
            # Fit OLS model to estimate variance
            variance_model = OLS(y, exog_var).fit()
            
            # Update weights using inverse link function
            new_weights = 1 / self.linkinv(exog_var.dot(variance_model.params))
            self.weights = new_weights / new_weights.mean()  # Normalize weights
        
        self.results_residual_regression = variance_model
        return model
