"""
Generalized linear models currently supports estimation using the one-parameter
exponential families

References
----------
Gill, Jeff. 2000. Generalized Linear Models: A Unified Approach.
    SAGE QASS Series.

Green, PJ. 1984.  "Iteratively reweighted least squares for maximum
    likelihood estimation, and some robust and resistant alternatives."
    Journal of the Royal Statistical Society, Series B, 46, 149-192.

Hardin, J.W. and Hilbe, J.M. 2007.  "Generalized Linear Models and
    Extensions."  2nd ed.  Stata Press, College Station, TX.

McCullagh, P. and Nelder, J.A.  1989.  "Generalized Linear Models." 2nd ed.
    Chapman & Hall, Boca Rotan.
"""
from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from numpy.linalg.linalg import LinAlgError
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base import _prediction_inference as pred
from statsmodels.base._prediction_inference import PredictionResultsMean
import statsmodels.base._parameter_inference as pinfer
from statsmodels.graphics._regressionplots_doc import _plot_added_variable_doc, _plot_ceres_residuals_doc, _plot_partial_residuals_doc
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly, cached_data, cached_value
from statsmodels.tools.docstring import Docstring
from statsmodels.tools.sm_exceptions import DomainWarning, HessianInversionWarning, PerfectSeparationWarning
from statsmodels.tools.validation import float_like
from . import families
__all__ = ['GLM', 'PredictionResultsMean']

class _ModuleVariable:
    _value = None
_use_bic_helper = _ModuleVariable()
SET_USE_BIC_LLF = _use_bic_helper.set_use_bic_llf

class GLM(base.LikelihoodModel):
    __doc__ = "\n    Generalized Linear Models\n\n    GLM inherits from statsmodels.base.model.LikelihoodModel\n\n    Parameters\n    ----------\n    endog : array_like\n        1d array of endogenous response variable.  This array can be 1d or 2d.\n        Binomial family models accept a 2d array with two columns. If\n        supplied, each observation is expected to be [success, failure].\n    exog : array_like\n        A nobs x k array where `nobs` is the number of observations and `k`\n        is the number of regressors. An intercept is not included by default\n        and should be added by the user (models specified using a formula\n        include an intercept by default). See `statsmodels.tools.add_constant`.\n    family : family class instance\n        The default is Gaussian.  To specify the binomial distribution\n        family = sm.family.Binomial()\n        Each family can take a link instance as an argument.  See\n        statsmodels.family.family for more information.\n    offset : array_like or None\n        An offset to be included in the model.  If provided, must be\n        an array whose length is the number of rows in exog.\n    exposure : array_like or None\n        Log(exposure) will be added to the linear prediction in the model.\n        Exposure is only valid if the log link is used. If provided, it must be\n        an array with the same length as endog.\n    freq_weights : array_like\n        1d array of frequency weights. The default is None. If None is selected\n        or a blank value, then the algorithm will replace with an array of 1's\n        with length equal to the endog.\n        WARNING: Using weights is not verified yet for all possible options\n        and results, see Notes.\n    var_weights : array_like\n        1d array of variance (analytic) weights. The default is None. If None\n        is selected or a blank value, then the algorithm will replace with an\n        array of 1's with length equal to the endog.\n        WARNING: Using weights is not verified yet for all possible options\n        and results, see Notes.\n    %(extra_params)s\n\n    Attributes\n    ----------\n    df_model : float\n        Model degrees of freedom is equal to p - 1, where p is the number\n        of regressors.  Note that the intercept is not reported as a\n        degree of freedom.\n    df_resid : float\n        Residual degrees of freedom is equal to the number of observation n\n        minus the number of regressors p.\n    endog : ndarray\n        See Notes.  Note that `endog` is a reference to the data so that if\n        data is already an array and it is changed, then `endog` changes\n        as well.\n    exposure : array_like\n        Include ln(exposure) in model with coefficient constrained to 1. Can\n        only be used if the link is the logarithm function.\n    exog : ndarray\n        See Notes.  Note that `exog` is a reference to the data so that if\n        data is already an array and it is changed, then `exog` changes\n        as well.\n    freq_weights : ndarray\n        See Notes. Note that `freq_weights` is a reference to the data so that\n        if data is already an array and it is changed, then `freq_weights`\n        changes as well.\n    var_weights : ndarray\n        See Notes. Note that `var_weights` is a reference to the data so that\n        if data is already an array and it is changed, then `var_weights`\n        changes as well.\n    iteration : int\n        The number of iterations that fit has run.  Initialized at 0.\n    family : family class instance\n        The distribution family of the model. Can be any family in\n        statsmodels.families.  Default is Gaussian.\n    mu : ndarray\n        The mean response of the transformed variable.  `mu` is the value of\n        the inverse of the link function at lin_pred, where lin_pred is the\n        linear predicted value of the WLS fit of the transformed variable.\n        `mu` is only available after fit is called.  See\n        statsmodels.families.family.fitted of the distribution family for more\n        information.\n    n_trials : ndarray\n        See Notes. Note that `n_trials` is a reference to the data so that if\n        data is already an array and it is changed, then `n_trials` changes\n        as well. `n_trials` is the number of binomial trials and only available\n        with that distribution. See statsmodels.families.Binomial for more\n        information.\n    normalized_cov_params : ndarray\n        The p x p normalized covariance of the design / exogenous data.\n        This is approximately equal to (X.T X)^(-1)\n    offset : array_like\n        Include offset in model with coefficient constrained to 1.\n    scale : float\n        The estimate of the scale / dispersion of the model fit.  Only\n        available after fit is called.  See GLM.fit and GLM.estimate_scale\n        for more information.\n    scaletype : str\n        The scaling used for fitting the model.  This is only available after\n        fit is called.  The default is None.  See GLM.fit for more information.\n    weights : ndarray\n        The value of the weights after the last iteration of fit.  Only\n        available after fit is called.  See statsmodels.families.family for\n        the specific distribution weighting functions.\n\n    Examples\n    --------\n    >>> import statsmodels.api as sm\n    >>> data = sm.datasets.scotland.load()\n    >>> data.exog = sm.add_constant(data.exog)\n\n    Instantiate a gamma family model with the default link function.\n\n    >>> gamma_model = sm.GLM(data.endog, data.exog,\n    ...                      family=sm.families.Gamma())\n\n    >>> gamma_results = gamma_model.fit()\n    >>> gamma_results.params\n    array([-0.01776527,  0.00004962,  0.00203442, -0.00007181,  0.00011185,\n           -0.00000015, -0.00051868, -0.00000243])\n    >>> gamma_results.scale\n    0.0035842831734919055\n    >>> gamma_results.deviance\n    0.087388516416999198\n    >>> gamma_results.pearson_chi2\n    0.086022796163805704\n    >>> gamma_results.llf\n    -83.017202161073527\n\n    See Also\n    --------\n    statsmodels.genmod.families.family.Family\n    :ref:`families`\n    :ref:`links`\n\n    Notes\n    -----\n    Note: PerfectSeparationError exception has been converted to a\n    PerfectSeparationWarning and perfect separation or perfect prediction will\n    not raise an exception by default. (changed in version 0.14)\n\n    Only the following combinations make sense for family and link:\n\n     ============= ===== === ===== ====== ======= === ==== ====== ====== ====\n     Family        ident log logit probit cloglog pow opow nbinom loglog logc\n     ============= ===== === ===== ====== ======= === ==== ====== ====== ====\n     Gaussian      x     x   x     x      x       x   x     x      x\n     inv Gaussian  x     x                        x\n     binomial      x     x   x     x      x       x   x           x      x\n     Poisson       x     x                        x\n     neg binomial  x     x                        x        x\n     gamma         x     x                        x\n     Tweedie       x     x                        x\n     ============= ===== === ===== ====== ======= === ==== ====== ====== ====\n\n    Not all of these link functions are currently available.\n\n    Endog and exog are references so that if the data they refer to are already\n    arrays and these arrays are changed, endog and exog will change.\n\n    statsmodels supports two separate definitions of weights: frequency weights\n    and variance weights.\n\n    Frequency weights produce the same results as repeating observations by the\n    frequencies (if those are integers). Frequency weights will keep the number\n    of observations consistent, but the degrees of freedom will change to\n    reflect the new weights.\n\n    Variance weights (referred to in other packages as analytic weights) are\n    used when ``endog`` represents an an average or mean. This relies on the\n    assumption that that the inverse variance scales proportionally to the\n    weight--an observation that is deemed more credible should have less\n    variance and therefore have more weight. For the ``Poisson`` family--which\n    assumes that occurrences scale proportionally with time--a natural practice\n    would be to use the amount of time as the variance weight and set ``endog``\n    to be a rate (occurrences per period of time). Similarly, using a\n    compound Poisson family, namely ``Tweedie``, makes a similar assumption\n    about the rate (or frequency) of occurrences having variance proportional to\n    time.\n\n    Both frequency and variance weights are verified for all basic results with\n    nonrobust or heteroscedasticity robust ``cov_type``. Other robust\n    covariance types have not yet been verified, and at least the small sample\n    correction is currently not based on the correct total frequency count.\n\n    Currently, all residuals are not weighted by frequency, although they may\n    incorporate ``n_trials`` for ``Binomial`` and ``var_weights``\n\n    +---------------+----------------------------------+\n    | Residual Type | Applicable weights               |\n    +===============+==================================+\n    | Anscombe      | ``var_weights``                  |\n    +---------------+----------------------------------+\n    | Deviance      | ``var_weights``                  |\n    +---------------+----------------------------------+\n    | Pearson       | ``var_weights`` and ``n_trials`` |\n    +---------------+----------------------------------+\n    | Reponse       | ``n_trials``                     |\n    +---------------+----------------------------------+\n    | Working       | ``n_trials``                     |\n    +---------------+----------------------------------+\n\n    WARNING: Loglikelihood and deviance are not valid in models where\n    scale is equal to 1 (i.e., ``Binomial``, ``NegativeBinomial``, and\n    ``Poisson``). If variance weights are specified, then results such as\n    ``loglike`` and ``deviance`` are based on a quasi-likelihood\n    interpretation. The loglikelihood is not correctly specified in this case,\n    and statistics based on it, such AIC or likelihood ratio tests, are not\n    appropriate.\n    " % {'extra_params': base._missing_param_doc}
    _formula_max_endog = 2

    def __init__(self, endog, exog, family=None, offset=None, exposure=None, freq_weights=None, var_weights=None, missing='none', **kwargs):
        if type(self) is GLM:
            self._check_kwargs(kwargs, ['n_trials'])
        if family is not None and (not isinstance(family.link, tuple(family.safe_links))):
            warnings.warn(f'The {type(family.link).__name__} link function does not respect the domain of the {type(family).__name__} family.', DomainWarning)
        if exposure is not None:
            exposure = np.log(exposure)
        if offset is not None:
            offset = np.asarray(offset)
        if freq_weights is not None:
            freq_weights = np.asarray(freq_weights)
        if var_weights is not None:
            var_weights = np.asarray(var_weights)
        self.freq_weights = freq_weights
        self.var_weights = var_weights
        super(GLM, self).__init__(endog, exog, missing=missing, offset=offset, exposure=exposure, freq_weights=freq_weights, var_weights=var_weights, **kwargs)
        self._check_inputs(family, self.offset, self.exposure, self.endog, self.freq_weights, self.var_weights)
        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')
        self.nobs = self.endog.shape[0]
        self._data_attr.extend(['weights', 'mu', 'freq_weights', 'var_weights', 'iweights', '_offset_exposure', 'n_trials'])
        self._init_keys.append('family')
        self._setup_binomial()
        if 'n_trials' in kwargs:
            self.n_trials = kwargs['n_trials']
        offset_exposure = 0.0
        if hasattr(self, 'offset'):
            offset_exposure = self.offset
        if hasattr(self, 'exposure'):
            offset_exposure = offset_exposure + self.exposure
        self._offset_exposure = offset_exposure
        self.scaletype = None

    def initialize(self):
        """
        Initialize a generalized linear model.
        """
        self.exog = self.exog
        self.endog = self.endog
        self.nobs = self.endog.shape[0]
        self.nparams = self.exog.shape[1]
        self.df_model = self.nparams - 1
        self.df_resid = self.nobs - self.nparams
        self.offset = getattr(self, 'offset', 0)
        self.exposure = getattr(self, 'exposure', 0)
        self._offset_exposure = self.offset + self.exposure
        self.mu = self.family.starting_mu(self.endog)
        self.scale = 1.
        self.scaletype = 'deviance'
        self.iteration = 0
        self.history = {'deviance': [np.inf], 'params': [np.inf], 'scale': []}

    def loglike_mu(self, mu, scale=1.0):
        """
        Evaluate the log-likelihood for a generalized linear model.
        """
        return self.family.loglike(self.endog, mu, scale=scale,
                                   freq_weights=self.freq_weights,
                                   var_weights=self.var_weights)

    def loglike(self, params, scale=None):
        """
        Evaluate the log-likelihood for a generalized linear model.
        """
        lin_pred = np.dot(self.exog, params) + self._offset_exposure
        mu = self.family.link.inverse(lin_pred)
        if scale is None:
            scale = self.estimate_scale(mu)
        return self.loglike_mu(mu, scale)

    def score_obs(self, params, scale=None):
        """score first derivative of the loglikelihood for each observation.

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.

        Returns
        -------
        score_obs : ndarray, 2d
            The first derivative of the loglikelihood function evaluated at
            params for each observation.
        """
        mu = self.predict(params)
        if scale is None:
            scale = self.estimate_scale(mu)
        score_factor = self.family.score_factor(self.endog, mu, scale, self.var_weights)
        return score_factor[:, None] * self.exog

    def score(self, params, scale=None):
        """score, first derivative of the loglikelihood function

        Parameters
        ----------
        params : ndarray
            Parameter at which score is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.

        Returns
        -------
        score : ndarray_1d
            The first derivative of the loglikelihood function calculated as
            the sum of `score_obs`
        """
        score_obs = self.score_obs(params, scale=scale)
        return np.sum(score_obs, axis=0)

    def score_factor(self, params, scale=None):
        """weights for score for each observation

        This can be considered as score residuals.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.

        Returns
        -------
        score_factor : ndarray_1d
            A 1d weight vector used in the calculation of the score_obs.
            The score_obs are obtained by `score_factor[:, None] * exog`
        """
        mu = self.predict(params)
        if scale is None:
            scale = self.estimate_scale(mu)
        return self.family.score_factor(self.endog, mu, scale, self.var_weights)

    def hessian_factor(self, params, scale=None, observed=True):
        """Weights for calculating Hessian

        Parameters
        ----------
        params : ndarray
            parameter at which Hessian is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        hessian_factor : ndarray, 1d
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`
        """
        pass

    def hessian(self, params, scale=None, observed=None):
        """Hessian, second derivative of loglikelihood function

        Parameters
        ----------
        params : ndarray
            parameter at which Hessian is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned (default).
            If False, then the expected information matrix is returned.

        Returns
        -------
        hessian : ndarray
            Hessian, i.e. observed information, or expected information matrix.
        """
        pass

    def information(self, params, scale=None):
        """
        Fisher information matrix.
        """
        pass

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None, offset=None, exposure=None):
        """
        Derivative of mean, expected endog with respect to the parameters
        """
        pass

    def _derivative_exog_helper(self, margeff, params, exog, dummy_idx, count_idx, transform):
        """
        Helper for _derivative_exog to wrap results appropriately
        """
        pass

    def _derivative_predict(self, params, exog=None, transform='dydx', offset=None, exposure=None):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated
        exog : ndarray or None
            Explanatory variables at which derivative are computed.
            If None, then the estimation exog is used.
        offset, exposure : None
            Not yet implemented.

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
        pass

    def _deriv_mean_dparams(self, params):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
        pass

    def _deriv_score_obs_dendog(self, params, scale=None):
        """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog. This
            can is given by `score_factor0[:, None] * exog` where
            `score_factor0` is the score_factor without the residual.
        """
        pass

    def score_test(self, params_constrained, k_constraints=None, exog_extra=None, observed=True):
        """score test for restrictions or for omitted variables

        The covariance matrix for the score is based on the Hessian, i.e.
        observed information matrix or optionally on the expected information
        matrix..

        Parameters
        ----------
        params_constrained : array_like
            estimated parameter of the restricted model. This can be the
            parameter estimate for the current when testing for omitted
            variables.
        k_constraints : int or None
            Number of constraints that were used in the estimation of params
            restricted relative to the number of exog in the model.
            This must be provided if no exog_extra are given. If exog_extra is
            not None, then k_constraints is assumed to be zero if it is None.
        exog_extra : None or array_like
            Explanatory variables that are jointly tested for inclusion in the
            model, i.e. omitted variables.
        observed : bool
            If True, then the observed Hessian is used in calculating the
            covariance matrix of the score. If false then the expected
            information matrix is used.

        Returns
        -------
        chi2_stat : float
            chisquare statistic for the score test
        p-value : float
            P-value of the score test based on the chisquare distribution.
        df : int
            Degrees of freedom used in the p-value calculation. This is equal
            to the number of constraints.

        Notes
        -----
        not yet verified for case with scale not equal to 1.
        """
        pass

    def _update_history(self, tmp_result, mu, history):
        """
        Helper method to update history during iterative fit.
        """
        pass

    def estimate_scale(self, mu):
        """
        Estimate the dispersion/scale.

        Type of scale can be chose in the fit method.

        Parameters
        ----------
        mu : ndarray
            mu is the mean response estimate

        Returns
        -------
        Estimate of scale

        Notes
        -----
        The default scale for Binomial, Poisson and Negative Binomial
        families is 1.  The default for the other families is Pearson's
        Chi-Square estimate.

        See Also
        --------
        statsmodels.genmod.generalized_linear_model.GLM.fit
        """
        pass

    def estimate_tweedie_power(self, mu, method='brentq', low=1.01, high=5.0):
        """
        Tweedie specific function to estimate scale and the variance parameter.
        The variance parameter is also referred to as p, xi, or shape.

        Parameters
        ----------
        mu : array_like
            Fitted mean response variable
        method : str, defaults to 'brentq'
            Scipy optimizer used to solve the Pearson equation. Only brentq
            currently supported.
        low : float, optional
            Low end of the bracketing interval [a,b] to be used in the search
            for the power. Defaults to 1.01.
        high : float, optional
            High end of the bracketing interval [a,b] to be used in the search
            for the power. Defaults to 5.

        Returns
        -------
        power : float
            The estimated shape or power.
        """
        pass

    def predict(self, params, exog=None, exposure=None, offset=None, which='mean', linear=None):
        """
        Return predicted values for a design matrix

        Parameters
        ----------
        params : array_like
            Parameters / coefficients of a GLM.
        exog : array_like, optional
            Design / exogenous data. Is exog is None, model exog is used.
        exposure : array_like, optional
            Exposure time values, only can be used with the log link
            function.  See notes for details.
        offset : array_like, optional
            Offset values.  See notes for details.
        which : 'mean', 'linear', 'var'(optional)
            Statitistic to predict. Default is 'mean'.

            - 'mean' returns the conditional expectation of endog E(y | x),
              i.e. inverse of the model's link function of linear predictor.
            - 'linear' returns the linear predictor of the mean function.
            - 'var_unscaled' variance of endog implied by the likelihood model.
              This does not include scale or var_weights.

        linear : bool
            The ``linear` keyword is deprecated and will be removed,
            use ``which`` keyword instead.
            If True, returns the linear predicted values.  If False or None,
            then the statistic specified by ``which`` will be returned.


        Returns
        -------
        An array of fitted values

        Notes
        -----
        Any `exposure` and `offset` provided here take precedence over
        the `exposure` and `offset` used in the model fit.  If `exog`
        is passed as an argument here, then any `exposure` and
        `offset` values in the fit will be ignored.

        Exposure values must be strictly positive.
        """
        pass

    def get_distribution(self, params, scale=None, exog=None, exposure=None, offset=None, var_weights=1.0, n_trials=1.0):
        """
        Return a instance of the predictive distribution.

        Parameters
        ----------
        params : array_like
            The model parameters.
        scale : scalar
            The scale parameter.
        exog : array_like
            The predictor variable matrix.
        offset : array_like or None
            Offset variable for predicted mean.
        exposure : array_like or None
            Log(exposure) will be added to the linear prediction.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is None.
        n_trials : int
            Number of trials for the binomial distribution. The default is 1
            which corresponds to a Bernoulli random variable.

        Returns
        -------
        gen
            Instance of a scipy frozen distribution based on estimated
            parameters.
            Use the ``rvs`` method to generate random values.

        Notes
        -----
        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """
        pass

    def fit(self, start_params=None, maxiter=100, method='IRLS', tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, full_output=True, disp=False, max_start_irls=3, **kwargs):
        """
        Fits a generalized linear model for a given family.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is family-specific and is given by the
            ``family.starting_mu(endog)``. If start_params is given then the
            initial mean will be calculated as ``np.dot(exog, start_params)``.
        maxiter : int, optional
            Default is 100.
        method : str
            Default is 'IRLS' for iteratively reweighted least squares.
            Otherwise gradient optimization is used.
        tol : float
            Convergence tolerance.  Default is 1e-8.
        scale : str or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default value is None, which uses `X2` for Gamma, Gaussian,
            and Inverse Gaussian.
            `X2` is Pearson's chi-square divided by `df_resid`.
            The default is 1 for the Binomial and Poisson families.
            `dev` is the deviance divided by df_resid
        cov_type : str
            The type of parameter estimate covariance matrix to compute.
        cov_kwds : dict-like
            Extra arguments for calculating the covariance of the parameter
            estimates.
        use_t : bool
            If True, the Student t-distribution is used for inference.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
            Not used if methhod is IRLS.
        disp : bool, optional
            Set to True to print convergence messages.  Not used if method is
            IRLS.
        max_start_irls : int
            The number of IRLS iterations used to obtain starting
            values for gradient optimization.  Only relevant if
            `method` is set to something other than 'IRLS'.
        atol : float, optional
            (available with IRLS fits) The absolute tolerance criterion that
            must be satisfied. Defaults to ``tol``. Convergence is attained
            when: :math:`rtol * prior + atol > abs(current - prior)`
        rtol : float, optional
            (available with IRLS fits) The relative tolerance criterion that
            must be satisfied. Defaults to 0 which means ``rtol`` is not used.
            Convergence is attained when:
            :math:`rtol * prior + atol > abs(current - prior)`
        tol_criterion : str, optional
            (available with IRLS fits) Defaults to ``'deviance'``. Can
            optionally be ``'params'``.
        wls_method : str, optional
            (available with IRLS fits) options are 'lstsq', 'pinv' and 'qr'
            specifies which linear algebra function to use for the irls
            optimization. Default is `lstsq` which uses the same underlying
            svd based approach as 'pinv', but is faster during iterations.
            'lstsq' and 'pinv' regularize the estimate in singular and
            near-singular cases by truncating small singular values based
            on `rcond` of the respective numpy.linalg function. 'qr' is
            only valid for cases that are not singular nor near-singular.
        optim_hessian : {'eim', 'oim'}, optional
            (available with scipy optimizer fits) When 'oim'--the default--the
            observed Hessian is used in fitting. 'eim' is the expected Hessian.
            This may provide more stable fits, but adds assumption that the
            Hessian is correctly specified.

        Notes
        -----
        If method is 'IRLS', then an additional keyword 'attach_wls' is
        available. This is currently for internal use only and might change
        in future versions. If attach_wls' is true, then the final WLS
        instance of the IRLS iteration is attached to the results instance
        as `results_wls` attribute.
        """
        pass

    def _fit_gradient(self, start_params=None, method='newton', maxiter=100, tol=1e-08, full_output=True, disp=True, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, max_start_irls=3, **kwargs):
        """
        Fits a generalized linear model for a given family iteratively
        using the scipy gradient optimizers.
        """
        pass

    def _fit_irls(self, start_params=None, maxiter=100, tol=1e-08, scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        """
        Fits a generalized linear model for a given family using
        iteratively reweighted least squares (IRLS).
        """
        pass

    def fit_regularized(self, method='elastic_net', alpha=0.0, start_params=None, refit=False, opt_method='bfgs', **kwargs):
        """
        Return a regularized fit to a linear regression model.

        Parameters
        ----------
        method : {'elastic_net'}
            Only the `elastic_net` approach is currently implemented.
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.
        start_params : array_like
            Starting values for `params`.
        refit : bool
            If True, the model is refit using only the variables that
            have non-zero coefficients in the regularized fit.  The
            refitted model is not regularized.
        opt_method : string
            The method used for numerical optimization.
        **kwargs
            Additional keyword arguments used when fitting the model.

        Returns
        -------
        GLMResults
            An array or a GLMResults object, same type returned by `fit`.

        Notes
        -----
        The penalty is the ``elastic net`` penalty, which is a
        combination of L1 and L2 penalties.

        The function that is minimized is:

        .. math::

            -loglike/n + alpha*((1-L1\\_wt)*|params|_2^2/2 + L1\\_wt*|params|_1)

        where :math:`|*|_1` and :math:`|*|_2` are the L1 and L2 norms.

        Post-estimation results are based on the same data used to
        select variables, hence may be subject to overfitting biases.

        The elastic_net method uses the following keyword arguments:

        maxiter : int
            Maximum number of iterations
        L1_wt  : float
            Must be in [0, 1].  The L1 penalty has weight L1_wt and the
            L2 penalty has weight 1 - L1_wt.
        cnvrg_tol : float
            Convergence threshold for maximum parameter change after
            one sweep through all coefficients.
        zero_tol : float
            Coefficients below this threshold are treated as zero.
        """
        pass

    def fit_constrained(self, constraints, start_params=None, **fit_kwds):
        """fit the model subject to linear equality constraints

        The constraints are of the form   `R params = q`
        where R is the constraint_matrix and q is the vector of
        constraint_values.

        The estimation creates a new model with transformed design matrix,
        exog, and converts the results back to the original parameterization.


        Parameters
        ----------
        constraints : formula expression or tuple
            If it is a tuple, then the constraint needs to be given by two
            arrays (constraint_matrix, constraint_value), i.e. (R, q).
            Otherwise, the constraints can be given as strings or list of
            strings.
            see t_test for details
        start_params : None or array_like
            starting values for the optimization. `start_params` needs to be
            given in the original parameter space and are internally
            transformed.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the transformed model.

        Returns
        -------
        results : Results instance
        """
        pass
get_prediction_doc = Docstring(pred.get_prediction_glm.__doc__)
get_prediction_doc.remove_parameters('pred_kwds')

class GLMResults(base.LikelihoodModelResults):
    """
    Class to contain GLM results.

    GLMResults inherits from statsmodels.LikelihoodModelResults

    Attributes
    ----------
    df_model : float
        See GLM.df_model
    df_resid : float
        See GLM.df_resid
    fit_history : dict
        Contains information about the iterations. Its keys are `iterations`,
        `deviance` and `params`.
    model : class instance
        Pointer to GLM model instance that called fit.
    nobs : float
        The number of observations n.
    normalized_cov_params : ndarray
        See GLM docstring
    params : ndarray
        The coefficients of the fitted model.  Note that interpretation
        of the coefficients often depends on the distribution family and the
        data.
    pvalues : ndarray
        The two-tailed p-values for the parameters.
    scale : float
        The estimate of the scale / dispersion for the model fit.
        See GLM.fit and GLM.estimate_scale for more information.
    stand_errors : ndarray
        The standard errors of the fitted GLM.   #TODO still named bse

    See Also
    --------
    statsmodels.base.model.LikelihoodModelResults
    """

    def __init__(self, model, params, normalized_cov_params, scale, cov_type='nonrobust', cov_kwds=None, use_t=None):
        super(GLMResults, self).__init__(model, params, normalized_cov_params=normalized_cov_params, scale=scale)
        self.family = model.family
        self._endog = model.endog
        self.nobs = model.endog.shape[0]
        self._freq_weights = model.freq_weights
        self._var_weights = model.var_weights
        self._iweights = model.iweights
        if isinstance(self.family, families.Binomial):
            self._n_trials = self.model.n_trials
        else:
            self._n_trials = 1
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self._cache = {}
        self._data_attr.extend(['results_constrained', '_freq_weights', '_var_weights', '_iweights'])
        self._data_in_cache.extend(['null', 'mu'])
        self._data_attr_model = getattr(self, '_data_attr_model', [])
        self._data_attr_model.append('mu')
        from statsmodels.base.covtype import get_robustcov_results
        if use_t is None:
            self.use_t = False
        else:
            self.use_t = use_t
        ct = cov_type == 'nonrobust' or cov_type.upper().startswith('HC')
        if self.model._has_freq_weights and (not ct):
            from statsmodels.tools.sm_exceptions import SpecificationWarning
            warnings.warn('cov_type not fully supported with freq_weights', SpecificationWarning)
        if self.model._has_var_weights and (not ct):
            from statsmodels.tools.sm_exceptions import SpecificationWarning
            warnings.warn('cov_type not fully supported with var_weights', SpecificationWarning)
        if cov_type == 'nonrobust':
            self.cov_type = 'nonrobust'
            self.cov_kwds = {'description': 'Standard Errors assume that the' + ' covariance matrix of the errors is correctly ' + 'specified.'}
        else:
            if cov_kwds is None:
                cov_kwds = {}
            get_robustcov_results(self, cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)

    @cached_data
    def resid_response(self):
        """
        Response residuals.  The response residuals are defined as
        `endog` - `fittedvalues`
        """
        pass

    @cached_data
    def resid_pearson(self):
        """
        Pearson residuals.  The Pearson residuals are defined as
        (`endog` - `mu`)/sqrt(VAR(`mu`)) where VAR is the distribution
        specific variance function.  See statsmodels.families.family and
        statsmodels.families.varfuncs for more information.
        """
        pass

    @cached_data
    def resid_working(self):
        """
        Working residuals.  The working residuals are defined as
        `resid_response`/link'(`mu`).  See statsmodels.family.links for the
        derivatives of the link functions.  They are defined analytically.
        """
        pass

    @cached_data
    def resid_anscombe(self):
        """
        Anscombe residuals.  See statsmodels.families.family for distribution-
        specific Anscombe residuals. Currently, the unscaled residuals are
        provided. In a future version, the scaled residuals will be provided.
        """
        pass

    @cached_data
    def resid_anscombe_scaled(self):
        """
        Scaled Anscombe residuals.  See statsmodels.families.family for
        distribution-specific Anscombe residuals.
        """
        pass

    @cached_data
    def resid_anscombe_unscaled(self):
        """
        Unscaled Anscombe residuals.  See statsmodels.families.family for
        distribution-specific Anscombe residuals.
        """
        pass

    @cached_data
    def resid_deviance(self):
        """
        Deviance residuals.  See statsmodels.families.family for distribution-
        specific deviance residuals.
        """
        pass

    @cached_value
    def pearson_chi2(self):
        """
        Pearson's Chi-Squared statistic is defined as the sum of the squares
        of the Pearson residuals.
        """
        pass

    @cached_data
    def fittedvalues(self):
        """
        The estimated mean response.

        This is the value of the inverse of the link function at
        lin_pred, where lin_pred is the linear predicted value
        obtained by multiplying the design matrix by the coefficient
        vector.
        """
        pass

    @cached_data
    def mu(self):
        """
        See GLM docstring.
        """
        pass

    @cache_readonly
    def null(self):
        """
        Fitted values of the null model
        """
        pass

    @cache_readonly
    def deviance(self):
        """
        See statsmodels.families.family for the distribution-specific deviance
        functions.
        """
        pass

    @cache_readonly
    def null_deviance(self):
        """The value of the deviance function for the model fit with a constant
        as the only regressor."""
        pass

    @cache_readonly
    def llnull(self):
        """
        Log-likelihood of the model fit with a constant as the only regressor
        """
        pass

    def llf_scaled(self, scale=None):
        """
        Return the log-likelihood at the given scale, using the
        estimated scale if the provided scale is None.  In the Gaussian
        case with linear link, the concentrated log-likelihood is
        returned.
        """
        pass

    @cached_value
    def llf(self):
        """
        Value of the loglikelihood function evalued at params.
        See statsmodels.families.family for distribution-specific
        loglikelihoods.  The result uses the concentrated
        log-likelihood if the family is Gaussian and the link is linear,
        otherwise it uses the non-concentrated log-likelihood evaluated
        at the estimated scale.
        """
        pass

    def pseudo_rsquared(self, kind='cs'):
        """
        Pseudo R-squared

        Cox-Snell likelihood ratio pseudo R-squared is valid for both discrete
        and continuous data. McFadden's pseudo R-squared is only valid for
        discrete data.

        Cox & Snell's pseudo-R-squared:  1 - exp((llnull - llf)*(2/nobs))

        McFadden's pseudo-R-squared: 1 - (llf / llnull)

        Parameters
        ----------
        kind : P"cs", "mcf"}
            Type of pseudo R-square to return

        Returns
        -------
        float
            Pseudo R-squared
        """
        pass

    @cached_value
    def aic(self):
        """
        Akaike Information Criterion
        -2 * `llf` + 2 * (`df_model` + 1)
        """
        pass

    @property
    def bic(self):
        """
        Bayes Information Criterion

        `deviance` - `df_resid` * log(`nobs`)

        .. warning::

            The current definition is based on the deviance rather than the
            log-likelihood. This is not consistent with the AIC definition,
            and after 0.13 both will make use of the log-likelihood definition.

        Notes
        -----
        The log-likelihood version is defined
        -2 * `llf` + (`df_model` + 1)*log(n)
        """
        pass

    @cached_value
    def bic_deviance(self):
        """
        Bayes Information Criterion

        Based on the deviance,
        `deviance` - `df_resid` * log(`nobs`)
        """
        pass

    @cached_value
    def bic_llf(self):
        """
        Bayes Information Criterion

        Based on the log-likelihood,
        -2 * `llf` + log(n) * (`df_model` + 1)
        """
        pass

    def info_criteria(self, crit, scale=None, dk_params=0):
        """Return an information criterion for the model.

        Parameters
        ----------
        crit : string
            One of 'aic', 'bic', or 'qaic'.
        scale : float
            The scale parameter estimated using the parent model,
            used only for qaic.
        dk_params : int or float
            Correction to the number of parameters used in the information
            criterion. By default, only mean parameters are included, the
            scale parameter is not included in the parameter count.
            Use ``dk_params=1`` to include scale in the parameter count.

        Returns
        -------
        Value of information criterion.

        Notes
        -----
        The quasi-Akaike Information criterion (qaic) is -2 *
        `llf`/`scale` + 2 * (`df_model` + 1).  It may not give
        meaningful results except for Poisson and related models.

        The QAIC (ic_type='qaic') must be evaluated with a provided
        scale parameter.  Two QAIC values are only comparable if they
        are calculated using the same scale parameter.  The scale
        parameter should be estimated using the largest model among
        all models being compared.

        References
        ----------
        Burnham KP, Anderson KR (2002). Model Selection and Multimodel
        Inference; Springer New York.
        """
        pass

    def get_prediction(self, exog=None, exposure=None, offset=None, transform=True, which=None, linear=None, average=False, agg_weights=None, row_labels=None):
        """
    Compute prediction results for GLM compatible models.

    Options and return class depend on whether "which" is None or not.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    exposure : array_like, optional
        Exposure time values, only can be used with the log link
        function.
    offset : array_like, optional
        Offset values.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    which : 'mean', 'linear', 'var'(optional)
        Statitistic to predict. Default is 'mean'.
        If which is None, then the deprecated keyword "linear" applies.
        If which is not None, then a generic Prediction results class will
        be returned. Some options are only available if which is not None.
        See notes.

        - 'mean' returns the conditional expectation of endog E(y | x),
          i.e. inverse of the model's link function of linear predictor.
        - 'linear' returns the linear predictor of the mean function.
        - 'var_unscaled' variance of endog implied by the likelihood model.
          This does not include scale or var_weights.

    linear : bool
        The ``linear` keyword is deprecated and will be removed,
        use ``which`` keyword instead.
        If which is None, then the linear keyword is used, otherwise it will
        be ignored.
        If True and which is None, the linear predicted values are returned.
        If False or None, then the statistic specified by ``which`` will be
        returned.
    average : bool
        Keyword is only used if ``which`` is not None.
        If average is True, then the mean prediction is computed, that is,
        predictions are computed for individual exog and then the average
        over observation is used.
        If average is False, then the results are the predictions for all
        observations, i.e. same length as ``exog``.
    agg_weights : ndarray, optional
        Keyword is only used if ``which`` is not None.
        Aggregation weights, only used if average is True.
    row_labels : list of str or None
        If row_lables are provided, then they will replace the generated
        labels.

    Returns
    -------
    prediction_results : instance of a PredictionResults class.
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
        The Results class of the return depends on the value of ``which``.

    See Also
    --------
    GLM.predict
    GLMResults.predict

    Notes
    -----
    Changes in statsmodels 0.14: The ``which`` keyword has been added.
    If ``which`` is None, then the behavior is the same as in previous
    versions, and returns the mean and linear prediction results.
    If the ``which`` keyword is not None, then a generic prediction results
    class is returned and is not backwards compatible with the old prediction
    results class, e.g. column names of summary_frame differs.
    There are more choices for the returned predicted statistic using
    ``which``. More choices will be added in the next release.
    Two additional keyword, average and agg_weights options are now also
    available if ``which`` is not None.
    In a future version ``which`` will become not None and the backwards
    compatible prediction results class will be removed.

    """
        pass

    def get_hat_matrix_diag(self, observed=True):
        """
        Compute the diagonal of the hat matrix

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.

        Returns
        -------
        hat_matrix_diag : ndarray
            The diagonal of the hat matrix computed from the observed
            or expected hessian.
        """
        pass

    def get_influence(self, observed=True):
        """
        Get an instance of GLMInfluence with influence and outlier measures

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.

        Returns
        -------
        infl : GLMInfluence instance
            The instance has methods to calculate the main influence and
            outlier measures as attributes.

        See Also
        --------
        statsmodels.stats.outliers_influence.GLMInfluence
        """
        pass

    def get_distribution(self, exog=None, exposure=None, offset=None, var_weights=1.0, n_trials=1.0):
        """
        Return a instance of the predictive distribution.

        Parameters
        ----------
        scale : scalar
            The scale parameter.
        exog : array_like
            The predictor variable matrix.
        offset : array_like or None
            Offset variable for predicted mean.
        exposure : array_like or None
            Log(exposure) will be added to the linear prediction.
        var_weights : array_like
            1d array of variance (analytic) weights. The default is None.
        n_trials : int
            Number of trials for the binomial distribution. The default is 1
            which corresponds to a Bernoulli random variable.

        Returns
        -------
        gen
            Instance of a scipy frozen distribution based on estimated
            parameters.
            Use the ``rvs`` method to generate random values.

        Notes
        -----
        Due to the behavior of ``scipy.stats.distributions objects``, the
        returned random number generator must be called with ``gen.rvs(n)``
        where ``n`` is the number of observations in the data set used
        to fit the model.  If any other value is used for ``n``, misleading
        results will be produced.
        """
        pass

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Warning: offset, exposure and weights (var_weights and freq_weights)
        are not supported by margeff.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is all
              only margeff will be available from the returned object.

            Note that if `exog` is specified, then marginal effects for all
            variables not specified by `exog` are calculated using the `at`
            option.
        method : str, optional
            Options are:

            - 'dydx' - dy/dx - No transformation is made and marginal effects
              are returned.  This is the default.
            - 'eyex' - estimate elasticities of variables in `exog` --
              d(lny)/d(lnx)
            - 'dyex' - estimate semi-elasticity -- dy/d(lnx)
            - 'eydx' - estimate semi-elasticity -- d(lny)/dx

            Note that tranformations are done after each observation is
            calculated.  Semi-elasticities for binary variables are computed
            using the midpoint method. 'dyex' and 'eyex' do not make sense
            for discrete variables. For interpretations of these methods
            see notes below.
        atexog : array_like, optional
            Optionally, you can provide the exogenous variables over which to
            get the marginal effects.  This should be a dictionary with the key
            as the zero-indexed column number and the value of the dictionary.
            Default is None for all independent variables less the constant.
        dummy : bool, optional
            If False, treats binary variables (if present) as continuous.  This
            is the default.  Else if True, treats binary variables as
            changing from 0 to 1.  Note that any variable that is either 0 or 1
            is treated as binary.  Each binary variable is treated separately
            for now.
        count : bool, optional
            If False, treats count variables (if present) as continuous.  This
            is the default.  Else if True, the marginal effect is the
            change in probabilities when each observation is increased by one.

        Returns
        -------
        DiscreteMargins : marginal effects instance
            Returns an object that holds the marginal effects, standard
            errors, confidence intervals, etc. See
            `statsmodels.discrete.discrete_margins.DiscreteMargins` for more
            information.

        Notes
        -----
        Interpretations of methods:

        - 'dydx' - change in `endog` for a change in `exog`.
        - 'eyex' - proportional change in `endog` for a proportional change
          in `exog`.
        - 'dyex' - change in `endog` for a proportional change in `exog`.
        - 'eydx' - proportional change in `endog` for a change in `exog`.

        When using after Poisson, returns the expected number of events per
        period, assuming that the model is loglinear.

        Status : unsupported features offset, exposure and weights. Default
        handling of freq_weights for average effect "overall" might change.

        """
        pass

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        pass

    def summary2(self, yname=None, xname=None, title=None, alpha=0.05, float_format='%.4f'):
        """Experimental summary for regression Results

        Parameters
        ----------
        yname : str
            Name of the dependent variable (optional)
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces the
            default title
        alpha : float
            significance level for the confidence intervals
        float_format : str
            print format for floats in parameters summary

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary2.Summary : class to hold summary results
        """
        pass

class GLMResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {'resid_anscombe': 'rows', 'resid_deviance': 'rows', 'resid_pearson': 'rows', 'resid_response': 'rows', 'resid_working': 'rows'}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs, _attrs)
wrap.populate_wrapper(GLMResultsWrapper, GLMResults)
if __name__ == '__main__':
    from statsmodels.datasets import longley
    data = longley.load()
    GLMmod = GLM(data.endog, data.exog).fit()
    GLMT = GLMmod.summary(returns='tables')
    GLMTp = GLMmod.summary(title='Test GLM')
    '\nFrom Stata\n. webuse beetle\n. glm r i.beetle ldose, family(binomial n) link(cloglog)\n\nIteration 0:   log likelihood = -79.012269\nIteration 1:   log likelihood =  -76.94951\nIteration 2:   log likelihood = -76.945645\nIteration 3:   log likelihood = -76.945645\n\nGeneralized linear models                          No. of obs      =        24\nOptimization     : ML                              Residual df     =        20\n                                                   Scale parameter =         1\nDeviance         =  73.76505595                    (1/df) Deviance =  3.688253\nPearson          =   71.8901173                    (1/df) Pearson  =  3.594506\n\nVariance function: V(u) = u*(1-u/n)                [Binomial]\nLink function    : g(u) = ln(-ln(1-u/n))           [Complementary log-log]\n\n                                                   AIC             =   6.74547\nLog likelihood   = -76.94564525                    BIC             =  10.20398\n\n------------------------------------------------------------------------------\n             |                 OIM\n           r |      Coef.   Std. Err.      z    P>|z|     [95% Conf. Interval]\n-------------+----------------------------------------------------------------\n      beetle |\n          2  |  -.0910396   .1076132    -0.85   0.398    -.3019576    .1198783\n          3  |  -1.836058   .1307125   -14.05   0.000     -2.09225   -1.579867\n             |\n       ldose |   19.41558   .9954265    19.50   0.000     17.46458    21.36658\n       _cons |  -34.84602    1.79333   -19.43   0.000    -38.36089   -31.33116\n------------------------------------------------------------------------------\n'
