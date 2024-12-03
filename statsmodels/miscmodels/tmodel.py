"""Linear Model with Student-t distributed errors

Because the t distribution has fatter tails than the normal distribution, it
can be used to model observations with heavier tails and observations that have
some outliers. For the latter case, the t-distribution provides more robust
estimators for mean or mean parameters (what about var?).



References
----------
Kenneth L. Lange, Roderick J. A. Little, Jeremy M. G. Taylor (1989)
Robust Statistical Modeling Using the t Distribution
Journal of the American Statistical Association
Vol. 84, No. 408 (Dec., 1989), pp. 881-896
Published by: American Statistical Association
Stable URL: http://www.jstor.org/stable/2290063

not read yet


Created on 2010-09-24
Author: josef-pktd
License: BSD

TODO
----
* add starting values based on OLS
* bugs: store_params does not seem to be defined, I think this was a module
        global for debugging - commented out
* parameter restriction: check whether version with some fixed parameters works


"""
import numpy as np
from scipy import special, stats
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.arma_mle import Arma
np_log = np.log
np_pi = np.pi
sps_gamln = special.gammaln

class TLinearModel(GenericLikelihoodModel):
    """Maximum Likelihood Estimation of Linear Model with t-distributed errors

    This is an example for generic MLE.

    Except for defining the negative log-likelihood method, all
    methods and results are generic. Gradients and Hessian
    and all resulting statistics are based on numerical
    differentiation.

    """

    def nloglikeobs(self, params):
        """
        Loglikelihood of linear model with t distributed errors.

        Parameters
        ----------
        params : ndarray
            The parameters of the model. The last 2 parameters are degrees of
            freedom and scale.

        Returns
        -------
        loglike : ndarray
            The log likelihood of the model evaluated at `params` for each
            observation defined by self.endog and self.exog.

        Notes
        -----
        .. math:: \\ln L=\\sum_{i=1}^{n}\\left[-\\lambda_{i}+y_{i}x_{i}^{\\prime}\\beta-\\ln y_{i}!\\right]

        The t distribution is the standard t distribution and not a standardized
        t distribution, which means that the scale parameter is not equal to the
        standard deviation.

        self.fixed_params and self.expandparams can be used to fix some
        parameters. (I doubt this has been tested in this model.)
        """
        y = self.endog
        X = self.exog
        df, scale = params[-2:]
        beta = params[:-2]
        
        nobs = len(y)
        k = len(beta)
        
        resid = y - np.dot(X, beta)
        
        nloglike = (nobs * (np_log(scale) + 0.5 * np_log(np_pi) + sps_gamln(0.5 * df) -
                            sps_gamln(0.5 * (df + 1))))
        nloglike += ((df + 1) / 2) * np.sum(np_log(1 + (resid / scale)**2 / df))
        
        return nloglike / nobs

class TArma(Arma):
    """Univariate Arma Model with t-distributed errors

    This inherit all methods except loglike from tsa.arma_mle.Arma

    This uses the standard t-distribution, the implied variance of
    the error is not equal to scale, but ::

        error_variance = df/(df-2)*scale**2

    Notes
    -----
    This might be replaced by a standardized t-distribution with scale**2
    equal to variance

    """

    def nloglikeobs(self, params):
        """
        Loglikelihood for arma model for each observation, t-distribute

        Notes
        -----
        The ancillary parameter is assumed to be the last element of
        the params vector
        """
        df = params[-1]
        scale = params[-2]
        arma_params = params[:-2]
        
        # Get ARMA residuals
        arma_resid = super().geterrors(arma_params)
        
        nloglike = (np_log(scale) + 0.5 * np_log(np_pi) + sps_gamln(0.5 * df) -
                    sps_gamln(0.5 * (df + 1)))
        nloglike += ((df + 1) / 2) * np_log(1 + (arma_resid / scale)**2 / df)
        
        return nloglike
