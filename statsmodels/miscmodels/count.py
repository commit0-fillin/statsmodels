"""
Created on Mon Jul 26 08:34:59 2010

Author: josef-pktd

changes:
added offset and zero-inflated version of Poisson
 - kind of ok, need better test cases,
 - a nan in ZIP bse, need to check hessian calculations
 - found error in ZIP loglike
 - all tests pass with

Issues
------
* If true model is not zero-inflated then numerical Hessian for ZIP has zeros
  for the inflation probability and is not invertible.
  -> hessian inverts and bse look ok if row and column are dropped, pinv also works
* GenericMLE: still get somewhere (where?)
   "CacheWriteWarning: The attribute 'bse' cannot be overwritten"
* bfgs is too fragile, does not come back
* `nm` is slow but seems to work
* need good start_params and their use in genericmle needs to be checked for
  consistency, set as attribute or method (called as attribute)
* numerical hessian needs better scaling

* check taking parts out of the loop, e.g. factorial(endog) could be precalculated


"""
import numpy as np
from scipy import stats
from scipy.special import factorial
from statsmodels.base.model import GenericLikelihoodModel

class PoissonGMLE(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    """

    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        The log likelihood of the model evaluated at `params`

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        XB = np.dot(self.exog, params)
        endog = self.endog
        L = np.exp(XB) - endog * XB + np.log(factorial(endog))
        return L

    def predict_distribution(self, exog):
        """return frozen scipy.stats distribution with mu at estimated prediction
        """
        mu = np.exp(np.dot(exog, self.params))
        return stats.poisson(mu)

class PoissonOffsetGMLE(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same
    statistical model as discretemod.Poisson but adds offset

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    """

    def __init__(self, endog, exog=None, offset=None, missing='none', **kwds):
        if offset is not None:
            if offset.ndim == 1:
                offset = offset[:, None]
            self.offset = offset.ravel()
        else:
            self.offset = 0.0
        super(PoissonOffsetGMLE, self).__init__(endog, exog, missing=missing, **kwds)

    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        The log likelihood of the model evaluated at `params`

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        XB = np.dot(self.exog, params) + self.offset
        endog = self.endog
        L = np.exp(XB) - endog * XB + np.log(factorial(endog))
        return L

class PoissonZiGMLE(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Poisson Model

    This is an example for generic MLE which has the same statistical model
    as discretemod.Poisson but adds offset and zero-inflation.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    There are numerical problems if there is no zero-inflation.

    """

    def __init__(self, endog, exog=None, offset=None, missing='none', **kwds):
        self.k_extra = 1
        super(PoissonZiGMLE, self).__init__(endog, exog, missing=missing, extra_params_names=['zi'], **kwds)
        if offset is not None:
            if offset.ndim == 1:
                offset = offset[:, None]
            self.offset = offset.ravel()
        else:
            self.offset = 0.0
        if exog is None:
            self.exog = np.ones((self.nobs, 1))
        self.nparams = self.exog.shape[1]
        self.start_params = np.hstack((np.ones(self.nparams), 0))
        self.nparams += 1
        self.cloneattr = ['start_params']

    def nloglikeobs(self, params):
        """
        Loglikelihood of Poisson model

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        The log likelihood of the model evaluated at `params`

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]
        """
        beta = params[:-1]
        zi = params[-1]
        XB = np.dot(self.exog, beta) + self.offset
        endog = self.endog
        mu = np.exp(XB)
        
        L = np.where(endog == 0,
                     np.log(zi + (1 - zi) * np.exp(-mu)),
                     np.log(1 - zi) - mu + endog * np.log(mu) - np.log(factorial(endog)))
        return -L
