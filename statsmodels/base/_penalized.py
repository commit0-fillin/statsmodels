"""
Created on Sun May 10 08:23:48 2015

Author: Josef Perktold
License: BSD-3
"""
import numpy as np
from ._penalties import NonePenalty
from statsmodels.tools.numdiff import approx_fprime_cs, approx_fprime

class PenalizedMixin:
    """Mixin class for Maximum Penalized Likelihood

    Parameters
    ----------
    args and kwds for the model super class
    penal : None or instance of Penalized function class
        If penal is None, then NonePenalty is used.
    pen_weight : float or None
        factor for weighting the penalization term.
        If None, then pen_weight is set to nobs.


    TODO: missing **kwds or explicit keywords

    TODO: do we adjust the inherited docstrings?
    We would need templating to add the penalization parameters
    """

    def __init__(self, *args, **kwds):
        self.penal = kwds.pop('penal', None)
        self.pen_weight = kwds.pop('pen_weight', None)
        super(PenalizedMixin, self).__init__(*args, **kwds)
        if self.pen_weight is None:
            self.pen_weight = len(self.endog)
        if self.penal is None:
            self.penal = NonePenalty()
            self.pen_weight = 0
        self._init_keys.extend(['penal', 'pen_weight'])
        self._null_drop_keys = getattr(self, '_null_drop_keys', [])
        self._null_drop_keys.extend(['penal', 'pen_weight'])

    def loglike(self, params, pen_weight=None, **kwds):
        """
        Log-likelihood of model at params
        """
        if pen_weight is None:
            pen_weight = self.pen_weight
        
        llf = self.loglikeobs(params, **kwds).sum()
        penalization = self.penal.func(params)
        return llf - pen_weight * penalization

    def loglikeobs(self, params, pen_weight=None, **kwds):
        """
        Log-likelihood of model observations at params
        """
        llf_obs = super().loglikeobs(params, **kwds)
        if pen_weight is None:
            pen_weight = self.pen_weight
        
        penalization = self.penal.func(params) / len(llf_obs)
        return llf_obs - pen_weight * penalization

    def score_numdiff(self, params, pen_weight=None, method='fd', **kwds):
        """score based on finite difference derivative
        """
        if pen_weight is None:
            pen_weight = self.pen_weight

        def f(params):
            return self.loglike(params, pen_weight=pen_weight, **kwds)

        if method == 'fd':
            return approx_fprime(params, f)
        elif method == 'cs':
            return approx_fprime_cs(params, f)
        else:
            raise ValueError("method must be 'fd' or 'cs'")

    def score(self, params, pen_weight=None, **kwds):
        """
        Gradient of model at params
        """
        if pen_weight is None:
            pen_weight = self.pen_weight
        
        score_obs = self.score_obs(params, **kwds)
        penalization_deriv = self.penal.deriv(params)
        return score_obs.sum(axis=0) - pen_weight * penalization_deriv

    def score_obs(self, params, pen_weight=None, **kwds):
        """
        Gradient of model observations at params
        """
        score_obs = super().score_obs(params, **kwds)
        if pen_weight is None:
            pen_weight = self.pen_weight
        
        penalization_deriv = self.penal.deriv(params) / len(score_obs)
        return score_obs - pen_weight * penalization_deriv

    def hessian_numdiff(self, params, pen_weight=None, **kwds):
        """hessian based on finite difference derivative
        """
        if pen_weight is None:
            pen_weight = self.pen_weight

        def f(params):
            return self.score(params, pen_weight=pen_weight, **kwds)

        return approx_fprime(params, f)

    def hessian(self, params, pen_weight=None, **kwds):
        """
        Hessian of model at params
        """
        if pen_weight is None:
            pen_weight = self.pen_weight
        
        hessian = super().hessian(params, **kwds)
        penalization_hessian = self.penal.hessian(params)
        return hessian - pen_weight * penalization_hessian

    def fit(self, method=None, trim=None, **kwds):
        """minimize negative penalized log-likelihood

        Parameters
        ----------
        method : None or str
            Method specifies the scipy optimizer as in nonlinear MLE models.
        trim : {bool, float}
            Default is False or None, which uses no trimming.
            If trim is True or a float, then small parameters are set to zero.
            If True, then a default threshold is used. If trim is a float, then
            it will be used as threshold.
            The default threshold is currently 1e-4, but it will change in
            future and become penalty function dependent.
        kwds : extra keyword arguments
            This keyword arguments are treated in the same way as in the
            fit method of the underlying model class.
            Specifically, additional optimizer keywords and cov_type related
            keywords can be added.
        """
        if method is None:
            method = 'bfgs'
        
        fit_kwds = kwds.copy()
        fit_kwds['method'] = method
        
        results = super().fit(**fit_kwds)
        
        if trim is not None:
            threshold = 1e-4 if trim is True else trim
            results.params[np.abs(results.params) < threshold] = 0
        
        return results
