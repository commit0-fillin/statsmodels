from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import ContrastResults, WaldTestResults, t_test_pairwise
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import cache_readonly, cached_data, cached_value
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import HessianInversionWarning, ValueWarning
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
ERROR_INIT_KWARGS = False
_model_params_doc = 'Parameters\n    ----------\n    endog : array_like\n        A 1-d endogenous response variable. The dependent variable.\n    exog : array_like\n        A nobs x k array where `nobs` is the number of observations and `k`\n        is the number of regressors. An intercept is not included by default\n        and should be added by the user. See\n        :func:`statsmodels.tools.add_constant`.'
_missing_param_doc = "missing : str\n        Available options are 'none', 'drop', and 'raise'. If 'none', no nan\n        checking is done. If 'drop', any observations with nans are dropped.\n        If 'raise', an error is raised. Default is 'none'."
_extra_param_doc = '\n    hasconst : None or bool\n        Indicates whether the RHS includes a user-supplied constant. If True,\n        a constant is not checked for and k_constant is set to 1 and all\n        result statistics are calculated as if a constant is present. If\n        False, a constant is not checked for and k_constant is set to 0.\n    **kwargs\n        Extra arguments that are used to set model properties when using the\n        formula interface.'

class Model:
    __doc__ = '\n    A (predictive) statistical model. Intended to be subclassed not used.\n\n    %(params_doc)s\n    %(extra_params_doc)s\n\n    Attributes\n    ----------\n    exog_names\n    endog_names\n\n    Notes\n    -----\n    `endog` and `exog` are references to any data provided.  So if the data is\n    already stored in numpy arrays and it is changed then `endog` and `exog`\n    will change as well.\n    ' % {'params_doc': _model_params_doc, 'extra_params_doc': _missing_param_doc + _extra_param_doc}
    _formula_max_endog = 1
    _kwargs_allowed = ['missing', 'missing_idx', 'formula', 'design_info', 'hasconst']

    def __init__(self, endog, exog=None, **kwargs):
        missing = kwargs.pop('missing', 'none')
        hasconst = kwargs.pop('hasconst', None)
        self.data = self._handle_data(endog, exog, missing, hasconst, **kwargs)
        self.k_constant = self.data.k_constant
        self.exog = self.data.exog
        self.endog = self.data.endog
        self._data_attr = []
        self._data_attr.extend(['exog', 'endog', 'data.exog', 'data.endog'])
        if 'formula' not in kwargs:
            self._data_attr.extend(['data.orig_endog', 'data.orig_exog'])
        self._init_keys = list(kwargs.keys())
        if hasconst is not None:
            self._init_keys.append('hasconst')

    def _get_init_kwds(self):
        """return dictionary with extra keys used in model.__init__
        """
        pass

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None, *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model.
        data : array_like
            The data for the model. See Notes.
        subset : array_like
            An array-like object of booleans, integers, or index values that
            indicate the subset of df to use in the model. Assumes df is a
            `pandas.DataFrame`.
        drop_cols : array_like
            Columns to drop from the design matrix.  Cannot be used to
            drop terms involving categoricals.
        *args
            Additional positional argument that are passed to the model.
        **kwargs
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model
            The model instance.

        Notes
        -----
        data must define __getitem__ with the keys in the formula terms
        args and kwargs are passed on to the model instantiation. E.g.,
        a numpy structured or rec array, a dictionary, or a pandas DataFrame.
        """
        pass

    @property
    def endog_names(self):
        """
        Names of endogenous variables.
        """
        pass

    @property
    def exog_names(self) -> list[str] | None:
        """
        Names of exogenous variables.
        """
        pass

    def fit(self):
        """
        Fit a model to data.
        """
        pass

    def predict(self, params, exog=None, *args, **kwargs):
        """
        After a model has been fit predict returns the fitted values.

        This is a placeholder intended to be overwritten by individual models.
        """
        pass

class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """

    def __init__(self, endog, exog=None, **kwargs):
        super().__init__(endog, exog, **kwargs)
        self.initialize()

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance.

        For example, if the the design matrix of a linear model changes then
        initialized can be used to recompute values using the modified design
        matrix.
        """
        pass

    def loglike(self, params):
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : ndarray
            The model parameters used to compute the log-likelihood.

        Notes
        -----
        Must be overridden by subclasses.
        """
        pass

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The score vector evaluated at the parameters.
        """
        pass

    def information(self, params):
        """
        Fisher information matrix of model.

        Returns -1 * Hessian of the log-likelihood evaluated at params.

        Parameters
        ----------
        params : ndarray
            The model parameters.
        """
        pass

    def hessian(self, params):
        """
        The Hessian matrix of the model.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The hessian evaluated at the parameters.
        """
        pass

    def fit(self, start_params=None, method='newton', maxiter=100, full_output=True, disp=True, fargs=(), callback=None, retall=False, skip_hessian=False, **kwargs):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver
            - 'minimize' for generic wrapper of scipy minimize (BFGS by default)

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        fargs : tuple, optional
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x,*args)
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool, optional
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.
        skip_hessian : bool, optional
            If False (default), then the negative inverse hessian is calculated
            after the optimization. If True, then the hessian will not be
            calculated. However, it will be available in methods that use the
            hessian in the optimization (currently only with `"newton"`).
        kwargs : keywords
            All kwargs are passed to the chosen solver with one exception. The
            following keyword controls what happens after the fit::

                warn_convergence : bool, optional
                    If True, checks the model for the converged flag. If the
                    converged flag is False, a ConvergenceWarning is issued.

        Notes
        -----
        The 'basinhopping' solver ignores `maxiter`, `retall`, `full_output`
        explicit arguments.

        Optional arguments for solvers (see returned Results.mle_settings)::

            'newton'
                tol : float
                    Relative error in params acceptable for convergence.
            'nm' -- Nelder Mead
                xtol : float
                    Relative error in params acceptable for convergence
                ftol : float
                    Relative error in loglike(params) acceptable for
                    convergence
                maxfun : int
                    Maximum number of function evaluations to make.
            'bfgs'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
            'lbfgs'
                m : int
                    This many terms are used for the Hessian approximation.
                factr : float
                    A stop condition that is a variant of relative error.
                pgtol : float
                    A stop condition that uses the projected gradient.
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
                maxfun : int
                    Maximum number of function evaluations to make.
                bounds : sequence
                    (min, max) pairs for each element in x,
                    defining the bounds on that parameter.
                    Use None for one of min or max when there is no bound
                    in that direction.
            'cg'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon : float
                    If fprime is approximated, use this value for the step
                    size. Can be scalar or vector.  Only relevant if
                    Likelihoodmodel.score is None.
            'ncg'
                fhess_p : callable f'(x,*args)
                    Function which computes the Hessian of f times an arbitrary
                    vector, p.  Should only be supplied if
                    LikelihoodModel.hessian is None.
                avextol : float
                    Stop when the average relative error in the minimizer
                    falls below this amount.
                epsilon : float or ndarray
                    If fhess is approximated, use this value for the step size.
                    Only relevant if Likelihoodmodel.hessian is None.
            'powell'
                xtol : float
                    Line-search error tolerance
                ftol : float
                    Relative error in loglike(params) for acceptable for
                    convergence.
                maxfun : int
                    Maximum number of function evaluations to make.
                start_direc : ndarray
                    Initial direction set.
            'basinhopping'
                niter : int
                    The number of basin hopping iterations.
                niter_success : int
                    Stop the run if the global minimum candidate remains the
                    same for this number of iterations.
                T : float
                    The "temperature" parameter for the accept or reject
                    criterion. Higher "temperatures" mean that larger jumps
                    in function value will be accepted. For best results
                    `T` should be comparable to the separation (in function
                    value) between local minima.
                stepsize : float
                    Initial step size for use in the random displacement.
                interval : int
                    The interval for how often to update the `stepsize`.
                minimizer : dict
                    Extra keyword arguments to be passed to the minimizer
                    `scipy.optimize.minimize()`, for example 'method' - the
                    minimization method (e.g. 'L-BFGS-B'), or 'tol' - the
                    tolerance for termination. Other arguments are mapped from
                    explicit argument of `fit`:
                      - `args` <- `fargs`
                      - `jac` <- `score`
                      - `hess` <- `hess`
            'minimize'
                min_method : str, optional
                    Name of minimization method to use.
                    Any method specific arguments can be passed directly.
                    For a list of methods and their arguments, see
                    documentation of `scipy.optimize.minimize`.
                    If no method is specified, then BFGS is used.
        """
        pass

    def _fit_zeros(self, keep_index=None, start_params=None, return_auxiliary=False, k_params=None, **fit_kwds):
        """experimental, fit the model subject to zero constraints

        Intended for internal use cases until we know what we need.
        API will need to change to handle models with two exog.
        This is not yet supported by all model subclasses.

        This is essentially a simplified version of `fit_constrained`, and
        does not need to use `offset`.

        The estimation creates a new model with transformed design matrix,
        exog, and converts the results back to the original parameterization.

        Some subclasses could use a more efficient calculation than using a
        new model.

        Parameters
        ----------
        keep_index : array_like (int or bool) or slice
            variables that should be dropped.
        start_params : None or array_like
            starting values for the optimization. `start_params` needs to be
            given in the original parameter space and are internally
            transformed.
        k_params : int or None
            If None, then we try to infer from start_params or model.
        **fit_kwds : keyword arguments
            fit_kwds are used in the optimization of the transformed model.

        Returns
        -------
        results : Results instance
        """
        pass

    def _fit_collinear(self, atol=1e-14, rtol=1e-13, **kwds):
        """experimental, fit of the model without collinear variables

        This currently uses QR to drop variables based on the given
        sequence.
        Options will be added in future, when the supporting functions
        to identify collinear variables become available.
        """
        pass

class GenericLikelihoodModel(LikelihoodModel):
    """
    Allows the fitting of any likelihood function via maximum likelihood.

    A subclass needs to specify at least the log-likelihood
    If the log-likelihood is specified for each observation, then results that
    require the Jacobian will be available. (The other case is not tested yet.)

    Notes
    -----
    Optimization methods that require only a likelihood function are 'nm' and
    'powell'

    Optimization methods that require a likelihood function and a
    score/gradient are 'bfgs', 'cg', and 'ncg'. A function to compute the
    Hessian is optional for 'ncg'.

    Optimization method that require a likelihood function, a score/gradient,
    and a Hessian is 'newton'

    If they are not overwritten by a subclass, then numerical gradient,
    Jacobian and Hessian of the log-likelihood are calculated by numerical
    forward differentiation. This might results in some cases in precision
    problems, and the Hessian might not be positive definite. Even if the
    Hessian is not positive definite the covariance matrix of the parameter
    estimates based on the outer product of the Jacobian might still be valid.


    Examples
    --------
    see also subclasses in directory miscmodels

    import statsmodels.api as sm
    data = sm.datasets.spector.load()
    data.exog = sm.add_constant(data.exog)
    # in this dir
    from model import GenericLikelihoodModel
    probit_mod = sm.Probit(data.endog, data.exog)
    probit_res = probit_mod.fit()
    loglike = probit_mod.loglike
    score = probit_mod.score
    mod = GenericLikelihoodModel(data.endog, data.exog, loglike, score)
    res = mod.fit(method="nm", maxiter = 500)
    import numpy as np
    np.allclose(res.params, probit_res.params)
    """

    def __init__(self, endog, exog=None, loglike=None, score=None, hessian=None, missing='none', extra_params_names=None, **kwds):
        if loglike is not None:
            self.loglike = loglike
        if score is not None:
            self.score = score
        if hessian is not None:
            self.hessian = hessian
        hasconst = kwds.pop('hasconst', None)
        self.__dict__.update(kwds)
        super(GenericLikelihoodModel, self).__init__(endog, exog, missing=missing, hasconst=hasconst, **kwds)
        if exog is not None:
            self.nparams = exog.shape[1] if np.ndim(exog) == 2 else 1
        if extra_params_names is not None:
            self._set_extra_params_names(extra_params_names)

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance. For
        instance, the design matrix of a linear model may change
        and some things must be recomputed.
        """
        pass

    def expandparams(self, params):
        """
        expand to full parameter array when some parameters are fixed

        Parameters
        ----------
        params : ndarray
            reduced parameter array

        Returns
        -------
        paramsfull : ndarray
            expanded parameter array where fixed parameters are included

        Notes
        -----
        Calling this requires that self.fixed_params and self.fixed_paramsmask
        are defined.

        *developer notes:*

        This can be used in the log-likelihood to ...

        this could also be replaced by a more general parameter
        transformation.
        """
        pass

    def reduceparams(self, params):
        """Reduce parameters"""
        pass

    def loglike(self, params):
        """Log-likelihood of model at params"""
        pass

    def nloglike(self, params):
        """Negative log-likelihood of model at params"""
        pass

    def loglikeobs(self, params):
        """
        Log-likelihood of the model for all observations at params.

        Parameters
        ----------
        params : array_like
            The parameters of the model.

        Returns
        -------
        loglike : array_like
            The log likelihood of the model evaluated at `params`.
        """
        pass

    def score(self, params):
        """
        Gradient of log-likelihood evaluated at params
        """
        pass

    def score_obs(self, params, **kwds):
        """
        Jacobian/Gradient of log-likelihood evaluated at params for each
        observation.
        """
        pass

    def hessian(self, params):
        """
        Hessian of log-likelihood evaluated at params
        """
        pass

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

class Results:
    """
    Class to contain model results

    Parameters
    ----------
    model : class instance
        the previously specified model instance
    params : ndarray
        parameter estimates from the fit model
    """

    def __init__(self, model, params, **kwd):
        self.__dict__.update(kwd)
        self.initialize(model, params, **kwd)
        self._data_attr = []
        self._data_in_cache = ['fittedvalues', 'resid', 'wresid']

    def initialize(self, model, params, **kwargs):
        """
        Initialize (possibly re-initialize) a Results instance.

        Parameters
        ----------
        model : Model
            The model instance.
        params : ndarray
            The model parameters.
        **kwargs
            Any additional keyword arguments required to initialize the model.
        """
        pass

    def predict(self, exog=None, transform=True, *args, **kwargs):
        """
        Call self.model.predict with self.params as the first argument.

        Parameters
        ----------
        exog : array_like, optional
            The values for which you want to predict. see Notes below.
        transform : bool, optional
            If the model was fit via a formula, do you want to pass
            exog through the formula. Default is True. E.g., if you fit
            a model y ~ log(x1) + log(x2), and transform is True, then
            you can pass a data structure that contains x1 and x2 in
            their original form. Otherwise, you'd need to log the data
            first.
        *args
            Additional arguments to pass to the model, see the
            predict method of the model for the details.
        **kwargs
            Additional keywords arguments to pass to the model, see the
            predict method of the model for the details.

        Returns
        -------
        array_like
            See self.model.predict.

        Notes
        -----
        The types of exog that are supported depends on whether a formula
        was used in the specification of the model.

        If a formula was used, then exog is processed in the same way as
        the original data. This transformation needs to have key access to the
        same variable names, and can be a pandas DataFrame or a dict like
        object that contains numpy arrays.

        If no formula was used, then the provided exog needs to have the
        same number of columns as the original exog in the model. No
        transformation of the data is performed except converting it to
        a numpy array.

        Row indices as in pandas data frames are supported, and added to the
        returned prediction.
        """
        pass

    def summary(self):
        """
        Summary

        Not implemented
        """
        pass

class LikelihoodModelResults(Results):
    """
    Class to contain results from likelihood models

    Parameters
    ----------
    model : LikelihoodModel instance or subclass instance
        LikelihoodModelResults holds a reference to the model that is fit.
    params : 1d array_like
        parameter estimates from estimated model
    normalized_cov_params : 2d array
       Normalized (before scaling) covariance of params. (dot(X.T,X))**-1
    scale : float
        For (some subset of models) scale will typically be the
        mean square error from the estimated model (sigma^2)

    Attributes
    ----------
    mle_retvals : dict
        Contains the values returned from the chosen optimization method if
        full_output is True during the fit.  Available only if the model
        is fit by maximum likelihood.  See notes below for the output from
        the different methods.
    mle_settings : dict
        Contains the arguments passed to the chosen optimization method.
        Available if the model is fit by maximum likelihood.  See
        LikelihoodModel.fit for more information.
    model : model instance
        LikelihoodResults contains a reference to the model that is fit.
    params : ndarray
        The parameters estimated for the model.
    scale : float
        The scaling factor of the model given during instantiation.
    tvalues : ndarray
        The t-values of the standard errors.


    Notes
    -----
    The covariance of params is given by scale times normalized_cov_params.

    Return values by solver if full_output is True during fit:

        'newton'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            score : ndarray
                The score vector at the optimum.
            Hessian : ndarray
                The Hessian at the optimum.
            warnflag : int
                1 if maxiter is exceeded. 0 if successful convergence.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'nm'
            fopt : float
                The value of the (negative) loglikelihood at its
                minimum.
            iterations : int
                Number of iterations performed.
            warnflag : int
                1: Maximum number of function evaluations made.
                2: Maximum number of iterations reached.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                List of solutions at each iteration.
        'bfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            Hinv : ndarray
                value of the inverse Hessian matrix at minimum.  Note
                that this is just an approximation and will often be
                different from the value of the analytic Hessian.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient
                and/or function calls are not changing.
            converged : bool
                True: converged.  False: did not converge.
            allvecs : list
                Results at each iteration.
        'lbfgs'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            gopt : float
                Value of gradient at minimum, which should be near 0.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                Warning flag:

                - 0 if converged
                - 1 if too many function evaluations or too many iterations
                - 2 if stopped for another reason

            converged : bool
                True: converged.  False: did not converge.
        'powell'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            direc : ndarray
                Current direction set.
            iterations : int
                Number of iterations performed.
            fcalls : int
                Number of calls to loglike.
            warnflag : int
                1: Maximum number of function evaluations. 2: Maximum number
                of iterations.
            converged : bool
                True : converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'cg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            warnflag : int
                1: Maximum number of iterations exceeded. 2: Gradient and/
                or function calls not changing.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        'ncg'
            fopt : float
                Value of the (negative) loglikelihood at its minimum.
            fcalls : int
                Number of calls to loglike.
            gcalls : int
                Number of calls to gradient/score.
            hcalls : int
                Number of calls to hessian.
            warnflag : int
                1: Maximum number of iterations exceeded.
            converged : bool
                True: converged. False: did not converge.
            allvecs : list
                Results at each iteration.
        """

    def __init__(self, model, params, normalized_cov_params=None, scale=1.0, **kwargs):
        super(LikelihoodModelResults, self).__init__(model, params)
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale
        self._use_t = False
        if 'use_t' in kwargs:
            use_t = kwargs['use_t']
            self.use_t = use_t if use_t is not None else False
        if 'cov_type' in kwargs:
            cov_type = kwargs.get('cov_type', 'nonrobust')
            cov_kwds = kwargs.get('cov_kwds', {})
            if cov_type == 'nonrobust':
                self.cov_type = 'nonrobust'
                self.cov_kwds = {'description': 'Standard Errors assume that the ' + 'covariance matrix of the errors is correctly ' + 'specified.'}
            else:
                from statsmodels.base.covtype import get_robustcov_results
                if cov_kwds is None:
                    cov_kwds = {}
                use_t = self.use_t
                get_robustcov_results(self, cov_type=cov_type, use_self=True, use_t=use_t, **cov_kwds)

    def normalized_cov_params(self):
        """See specific model class docstring"""
        pass

    @property
    def use_t(self):
        """Flag indicating to use the Student's distribution in inference."""
        pass

    @cached_value
    def llf(self):
        """Log-likelihood of model"""
        pass

    @cached_value
    def bse(self):
        """The standard errors of the parameter estimates."""
        pass

    @cached_value
    def tvalues(self):
        """
        Return the t-statistic for a given parameter estimate.
        """
        pass

    @cached_value
    def pvalues(self):
        """The two-tailed p values for the t-stats of the params."""
        pass

    def cov_params(self, r_matrix=None, column=None, scale=None, cov_p=None, other=None):
        """
        Compute the variance/covariance matrix.

        The variance/covariance matrix can be of a linear contrast of the
        estimated parameters or all params multiplied by scale which will
        usually be an estimate of sigma^2.  Scale is assumed to be a scalar.

        Parameters
        ----------
        r_matrix : array_like
            Can be 1d, or 2d.  Can be used alone or with other.
        column : array_like, optional
            Must be used on its own.  Can be 0d or 1d see below.
        scale : float, optional
            Can be specified or not.  Default is None, which means that
            the scale argument is taken from the model.
        cov_p : ndarray, optional
            The covariance of the parameters. If not provided, this value is
            read from `self.normalized_cov_params` or
            `self.cov_params_default`.
        other : array_like, optional
            Can be used when r_matrix is specified.

        Returns
        -------
        ndarray
            The covariance matrix of the parameter estimates or of linear
            combination of parameter estimates. See Notes.

        Notes
        -----
        (The below are assumed to be in matrix notation.)

        If no argument is specified returns the covariance matrix of a model
        ``(scale)*(X.T X)^(-1)``

        If contrast is specified it pre and post-multiplies as follows
        ``(scale) * r_matrix (X.T X)^(-1) r_matrix.T``

        If contrast and other are specified returns
        ``(scale) * r_matrix (X.T X)^(-1) other.T``

        If column is specified returns
        ``(scale) * (X.T X)^(-1)[column,column]`` if column is 0d

        OR

        ``(scale) * (X.T X)^(-1)[column][:,column]`` if column is 1d
        """
        pass

    def t_test(self, r_matrix, cov_p=None, use_t=None):
        """
        Compute a t-test for a each linear hypothesis of the form Rb = q.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : If an array is given, a p x k 2d array or length k 1d
              array specifying the linear restrictions. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q). If q is given,
              can be either a scalar or a length p row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        use_t : bool, optional
            If use_t is None, then the default of the model is used. If use_t
            is True, then the p-values are based on the t distribution. If
            use_t is False, then the p-values are based on the normal
            distribution.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.
            The available results have the same elements as the parameter table
            in `summary()`.

        See Also
        --------
        tvalues : Individual t statistics for the estimated parameters.
        f_test : Perform an F tests on model parameters.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> r = np.zeros_like(results.params)
        >>> r[5:] = [1,-1]
        >>> print(r)
        [ 0.  0.  0.  0.  0.  1. -1.]

        r tests that the coefficients on the 5th and 6th independent
        variable are the same.

        >>> T_test = results.t_test(r)
        >>> print(T_test)
                                     Test for Constraints
        ==============================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        c0         -1829.2026    455.391     -4.017      0.003   -2859.368    -799.037
        ==============================================================================
        >>> T_test.effect
        -1829.2025687192481
        >>> T_test.sd
        455.39079425193762
        >>> T_test.tvalue
        -4.0167754636411717
        >>> T_test.pvalue
        0.0015163772380899498

        Alternatively, you can specify the hypothesis tests using a string

        >>> from statsmodels.formula.api import ols
        >>> dta = sm.datasets.longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = 'GNPDEFL = GNP, UNEMP = 2, YEAR/1829 = 1'
        >>> t_test = results.t_test(hypotheses)
        >>> print(t_test)
                                     Test for Constraints
        ==============================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
        ------------------------------------------------------------------------------
        c0            15.0977     84.937      0.178      0.863    -177.042     207.238
        c1            -2.0202      0.488     -8.231      0.000      -3.125      -0.915
        c2             1.0001      0.249      0.000      1.000       0.437       1.563
        ==============================================================================
        """
        pass

    def f_test(self, r_matrix, cov_p=None, invcov=None):
        """
        Compute the F-test for a joint linear hypothesis.

        This is a special case of `wald_test` that always uses the F
        distribution.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed
              that the linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length k row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.

        See Also
        --------
        t_test : Perform a single hypothesis test.
        wald_test : Perform a Wald-test using a quadratic form.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.

        Examples
        --------
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> A = np.identity(len(results.params))
        >>> A = A[1:,:]

        This tests that each coefficient is jointly statistically
        significantly different from zero.

        >>> print(results.f_test(A))
        <F test: F=array([[ 330.28533923]]), p=4.984030528700946e-10, df_denom=9, df_num=6>

        Compare this to

        >>> results.fvalue
        330.2853392346658
        >>> results.f_pvalue
        4.98403096572e-10

        >>> B = np.array(([0,0,1,-1,0,0,0],[0,0,0,0,0,1,-1]))

        This tests that the coefficient on the 2nd and 3rd regressors are
        equal and jointly that the coefficient on the 5th and 6th regressors
        are equal.

        >>> print(results.f_test(B))
        <F test: F=array([[ 9.74046187]]), p=0.005605288531708235, df_denom=9, df_num=2>

        Alternatively, you can specify the hypothesis tests using a string

        >>> from statsmodels.datasets import longley
        >>> from statsmodels.formula.api import ols
        >>> dta = longley.load_pandas().data
        >>> formula = 'TOTEMP ~ GNPDEFL + GNP + UNEMP + ARMED + POP + YEAR'
        >>> results = ols(formula, dta).fit()
        >>> hypotheses = '(GNPDEFL = GNP), (UNEMP = 2), (YEAR/1829 = 1)'
        >>> f_test = results.f_test(hypotheses)
        >>> print(f_test)
        <F test: F=array([[ 144.17976065]]), p=6.322026217355609e-08, df_denom=9, df_num=3>
        """
        pass

    def wald_test(self, r_matrix, cov_p=None, invcov=None, use_f=None, df_constraints=None, scalar=None):
        """
        Compute a Wald-test for a joint linear hypothesis.

        Parameters
        ----------
        r_matrix : {array_like, str, tuple}
            One of:

            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.

        cov_p : array_like, optional
            An alternative estimate for the parameter covariance matrix.
            If None is given, self.normalized_cov_params is used.
        invcov : array_like, optional
            A q x q array to specify an inverse covariance matrix based on a
            restrictions matrix.
        use_f : bool
            If True, then the F-distribution is used. If False, then the
            asymptotic distribution, chisquare is used. If use_f is None, then
            the F distribution is used if the model specifies that use_t is True.
            The test statistic is proportionally adjusted for the distribution
            by the number of constraints in the hypothesis.
        df_constraints : int, optional
            The number of constraints. If not provided the number of
            constraints is determined from r_matrix.
        scalar : bool, optional
            Flag indicating whether the Wald test statistic should be returned
            as a sclar float. The current behavior is to return an array.
            This will switch to a scalar float after 0.14 is released. To
            get the future behavior now, set scalar to True. To silence
            the warning and retain the legacy behavior, set scalar to
            False.

        Returns
        -------
        ContrastResults
            The results for the test are attributes of this results instance.

        See Also
        --------
        f_test : Perform an F tests on model parameters.
        t_test : Perform a single hypothesis test.
        statsmodels.stats.contrast.ContrastResults : Test results.
        patsy.DesignInfo.linear_constraint : Specify a linear constraint.

        Notes
        -----
        The matrix `r_matrix` is assumed to be non-singular. More precisely,

        r_matrix (pX pX.T) r_matrix.T

        is assumed invertible. Here, pX is the generalized inverse of the
        design matrix of the model. There can be problems in non-OLS models
        where the rank of the covariance of the noise is not full.
        """
        pass

    def wald_test_terms(self, skip_single=False, extra_constraints=None, combine_terms=None, scalar=None):
        """
        Compute a sequence of Wald tests for terms over multiple columns.

        This computes joined Wald tests for the hypothesis that all
        coefficients corresponding to a `term` are zero.
        `Terms` are defined by the underlying formula or by string matching.

        Parameters
        ----------
        skip_single : bool
            If true, then terms that consist only of a single column and,
            therefore, refers only to a single parameter is skipped.
            If false, then all terms are included.
        extra_constraints : ndarray
            Additional constraints to test. Note that this input has not been
            tested.
        combine_terms : {list[str], None}
            Each string in this list is matched to the name of the terms or
            the name of the exogenous variables. All columns whose name
            includes that string are combined in one joint test.
        scalar : bool, optional
            Flag indicating whether the Wald test statistic should be returned
            as a sclar float. The current behavior is to return an array.
            This will switch to a scalar float after 0.14 is released. To
            get the future behavior now, set scalar to True. To silence
            the warning and retain the legacy behavior, set scalar to
            False.

        Returns
        -------
        WaldTestResults
            The result instance contains `table` which is a pandas DataFrame
            with the test results: test statistic, degrees of freedom and
            pvalues.

        Examples
        --------
        >>> res_ols = ols("np.log(Days+1) ~ C(Duration, Sum)*C(Weight, Sum)", data).fit()
        >>> res_ols.wald_test_terms()
        <class 'statsmodels.stats.contrast.WaldTestResults'>
                                                  F                P>F  df constraint  df denom
        Intercept                        279.754525  2.37985521351e-22              1        51
        C(Duration, Sum)                   5.367071    0.0245738436636              1        51
        C(Weight, Sum)                    12.432445  3.99943118767e-05              2        51
        C(Duration, Sum):C(Weight, Sum)    0.176002      0.83912310946              2        51

        >>> res_poi = Poisson.from_formula("Days ~ C(Weight) * C(Duration)",                                            data).fit(cov_type='HC0')
        >>> wt = res_poi.wald_test_terms(skip_single=False,                                          combine_terms=['Duration', 'Weight'])
        >>> print(wt)
                                    chi2             P>chi2  df constraint
        Intercept              15.695625  7.43960374424e-05              1
        C(Weight)              16.132616  0.000313940174705              2
        C(Duration)             1.009147     0.315107378931              1
        C(Weight):C(Duration)   0.216694     0.897315972824              2
        Duration               11.187849     0.010752286833              3
        Weight                 30.263368  4.32586407145e-06              4
        """
        pass

    def t_test_pairwise(self, term_name, method='hs', alpha=0.05, factor_labels=None):
        """
        Perform pairwise t_test with multiple testing corrected p-values.

        This uses the formula design_info encoding contrast matrix and should
        work for all encodings of a main effect.

        Parameters
        ----------
        term_name : str
            The name of the term for which pairwise comparisons are computed.
            Term names for categorical effects are created by patsy and
            correspond to the main part of the exog names.
        method : {str, list[str]}
            The multiple testing p-value correction to apply. The default is
            'hs'. See stats.multipletesting.
        alpha : float
            The significance level for multiple testing reject decision.
        factor_labels : {list[str], None}
            Labels for the factor levels used for pairwise labels. If not
            provided, then the labels from the formula design_info are used.

        Returns
        -------
        MultiCompResult
            The results are stored as attributes, the main attributes are the
            following two. Other attributes are added for debugging purposes
            or as background information.

            - result_frame : pandas DataFrame with t_test results and multiple
              testing corrected p-values.
            - contrasts : matrix of constraints of the null hypothesis in the
              t_test.

        Notes
        -----
        Status: experimental. Currently only checked for treatment coding with
        and without specified reference level.

        Currently there are no multiple testing corrected confidence intervals
        available.

        Examples
        --------
        >>> res = ols("np.log(Days+1) ~ C(Weight) + C(Duration)", data).fit()
        >>> pw = res.t_test_pairwise("C(Weight)")
        >>> pw.result_frame
                 coef   std err         t         P>|t|  Conf. Int. Low
        2-1  0.632315  0.230003  2.749157  8.028083e-03        0.171563
        3-1  1.302555  0.230003  5.663201  5.331513e-07        0.841803
        3-2  0.670240  0.230003  2.914044  5.119126e-03        0.209488
             Conf. Int. Upp.  pvalue-hs reject-hs
        2-1         1.093067   0.010212      True
        3-1         1.763307   0.000002      True
        3-2         1.130992   0.010212      True
        """
        pass

    def _get_wald_nonlinear(self, func, deriv=None):
        """Experimental method for nonlinear prediction and tests

        Parameters
        ----------
        func : callable, f(params)
            nonlinear function of the estimation parameters. The return of
            the function can be vector valued, i.e. a 1-D array
        deriv : function or None
            first derivative or Jacobian of func. If deriv is None, then a
            numerical derivative will be used. If func returns a 1-D array,
            then the `deriv` should have rows corresponding to the elements
            of the return of func.

        Returns
        -------
        nl : instance of `NonlinearDeltaCov` with attributes and methods to
            calculate the results for the prediction or tests

        """
        pass

    def conf_int(self, alpha=0.05, cols=None):
        """
        Construct confidence interval for the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the confidence interval. The default
            `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
            Specifies which confidence intervals to return.

        .. deprecated: 0.13

           cols is deprecated and will be removed after 0.14 is released.
           cols only works when inputs are NumPy arrays and will fail
           when using pandas Series or DataFrames as input. You can
           subset the confidence intervals using slices.

        Returns
        -------
        array_like
            Each row contains [lower, upper] limits of the confidence interval
            for the corresponding parameter. The first column contains all
            lower, the second column contains all upper limits.

        Notes
        -----
        The confidence interval is based on the standard normal distribution
        if self.use_t is False. If self.use_t is True, then uses a Student's t
        with self.df_resid_inference (or self.df_resid if df_resid_inference is
        not defined) degrees of freedom.

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> data = sm.datasets.longley.load()
        >>> data.exog = sm.add_constant(data.exog)
        >>> results = sm.OLS(data.endog, data.exog).fit()
        >>> results.conf_int()
        array([[-5496529.48322745, -1467987.78596704],
               [    -177.02903529,      207.15277984],
               [      -0.1115811 ,        0.03994274],
               [      -3.12506664,       -0.91539297],
               [      -1.5179487 ,       -0.54850503],
               [      -0.56251721,        0.460309  ],
               [     798.7875153 ,     2859.51541392]])

        >>> results.conf_int(cols=(2,3))
        array([[-0.1115811 ,  0.03994274],
               [-3.12506664, -0.91539297]])
        """
        pass

    def save(self, fname, remove_data=False):
        """
        Save a pickle of this instance.

        Parameters
        ----------
        fname : {str, handle}
            A string filename or a file handle.
        remove_data : bool
            If False (default), then the instance is pickled without changes.
            If True, then all arrays with length nobs are set to None before
            pickling. See the remove_data method.
            In some cases not all arrays will be set to None.

        Notes
        -----
        If remove_data is true and the model result does not implement a
        remove_data method then this will raise an exception.
        """
        pass

    @classmethod
    def load(cls, fname):
        """
        Load a pickled results instance

        .. warning::

           Loading pickled models is not secure against erroneous or
           maliciously constructed data. Never unpickle data received from
           an untrusted or unauthenticated source.

        Parameters
        ----------
        fname : {str, handle, pathlib.Path}
            A string filename or a file handle.

        Returns
        -------
        Results
            The unpickled results instance.
        """
        pass

    def remove_data(self):
        """
        Remove data arrays, all nobs arrays from result and model.

        This reduces the size of the instance, so it can be pickled with less
        memory. Currently tested for use with predict from an unpickled
        results and model instance.

        .. warning::

           Since data and some intermediate results have been removed
           calculating new statistics that require them will raise exceptions.
           The exception will occur the first time an attribute is accessed
           that has been set to None.

        Not fully tested for time series models, tsa, and might delete too much
        for prediction or not all that would be possible.

        The lists of arrays to delete are maintained as attributes of
        the result and model instance, except for cached values. These
        lists could be changed before calling remove_data.

        The attributes to remove are named in:

        model._data_attr : arrays attached to both the model instance
            and the results instance with the same attribute name.

        result._data_in_cache : arrays that may exist as values in
            result._cache

        result._data_attr_model : arrays attached to the model
            instance but not to the results instance
        """
        pass

class LikelihoodResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'params': 'columns', 'bse': 'columns', 'pvalues': 'columns', 'tvalues': 'columns', 'resid': 'rows', 'fittedvalues': 'rows', 'normalized_cov_params': 'cov'}
    _wrap_attrs = _attrs
    _wrap_methods = {'cov_params': 'cov', 'conf_int': 'columns'}
wrap.populate_wrapper(LikelihoodResultsWrapper, LikelihoodModelResults)

class ResultMixin:

    @cache_readonly
    def df_modelwc(self):
        """Model WC"""
        pass

    @cache_readonly
    def aic(self):
        """Akaike information criterion"""
        pass

    @cache_readonly
    def bic(self):
        """Bayesian information criterion"""
        pass

    @cache_readonly
    def score_obsv(self):
        """cached Jacobian of log-likelihood
        """
        pass

    @cache_readonly
    def hessv(self):
        """cached Hessian of log-likelihood
        """
        pass

    @cache_readonly
    def covjac(self):
        """
        covariance of parameters based on outer product of jacobian of
        log-likelihood
        """
        pass

    @cache_readonly
    def covjhj(self):
        """covariance of parameters based on HJJH

        dot product of Hessian, Jacobian, Jacobian, Hessian of likelihood

        name should be covhjh
        """
        pass

    @cache_readonly
    def bsejhj(self):
        """standard deviation of parameter estimates based on covHJH
        """
        pass

    @cache_readonly
    def bsejac(self):
        """standard deviation of parameter estimates based on covjac
        """
        pass

    def bootstrap(self, nrep=100, method='nm', disp=0, store=1):
        """simple bootstrap to get mean and variance of estimator

        see notes

        Parameters
        ----------
        nrep : int
            number of bootstrap replications
        method : str
            optimization method to use
        disp : bool
            If true, then optimization prints results
        store : bool
            If true, then parameter estimates for all bootstrap iterations
            are attached in self.bootstrap_results

        Returns
        -------
        mean : ndarray
            mean of parameter estimates over bootstrap replications
        std : ndarray
            standard deviation of parameter estimates over bootstrap
            replications

        Notes
        -----
        This was mainly written to compare estimators of the standard errors of
        the parameter estimates.  It uses independent random sampling from the
        original endog and exog, and therefore is only correct if observations
        are independently distributed.

        This will be moved to apply only to models with independently
        distributed observations.
        """
        pass

    def get_nlfun(self, fun):
        """
        get_nlfun

        This is not Implemented
        """
        pass

class _LLRMixin:
    """Mixin class for Null model and likelihood ratio
    """

    def pseudo_rsquared(self, kind='mcf'):
        """
        McFadden's pseudo-R-squared. `1 - (llf / llnull)`
        """
        pass

    @cache_readonly
    def llr(self):
        """
        Likelihood ratio chi-squared statistic; `-2*(llnull - llf)`
        """
        pass

    @cache_readonly
    def llr_pvalue(self):
        """
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
        """
        pass

    def set_null_options(self, llnull=None, attach_results=True, **kwargs):
        """
        Set the fit options for the Null (constant-only) model.

        This resets the cache for related attributes which is potentially
        fragile. This only sets the option, the null model is estimated
        when llnull is accessed, if llnull is not yet in cache.

        Parameters
        ----------
        llnull : {None, float}
            If llnull is not None, then the value will be directly assigned to
            the cached attribute "llnull".
        attach_results : bool
            Sets an internal flag whether the results instance of the null
            model should be attached. By default without calling this method,
            thenull model results are not attached and only the loglikelihood
            value llnull is stored.
        **kwargs
            Additional keyword arguments used as fit keyword arguments for the
            null model. The override and model default values.

        Notes
        -----
        Modifies attributes of this instance, and so has no return.
        """
        pass

    @cache_readonly
    def llnull(self):
        """
        Value of the constant-only loglikelihood
        """
        pass

class GenericLikelihoodModelResults(LikelihoodModelResults, ResultMixin):
    """
    A results class for the discrete dependent variable models.

    ..Warning :

    The following description has not been updated to this version/class.
    Where are AIC, BIC, ....? docstring looks like copy from discretemod

    Parameters
    ----------
    model : A DiscreteModel instance
    mlefit : instance of LikelihoodResults
        This contains the numerical optimization results as returned by
        LikelihoodModel.fit(), in a superclass of GnericLikelihoodModels


    Attributes
    ----------
    aic : float
        Akaike information criterion.  -2*(`llf` - p) where p is the number
        of regressors including the intercept.
    bic : float
        Bayesian information criterion. -2*`llf` + ln(`nobs`)*p where p is the
        number of regressors including the intercept.
    bse : ndarray
        The standard errors of the coefficients.
    df_resid : float
        See model definition.
    df_model : float
        See model definition.
    fitted_values : ndarray
        Linear predictor XB.
    llf : float
        Value of the loglikelihood
    llnull : float
        Value of the constant-only loglikelihood
    llr : float
        Likelihood ratio chi-squared statistic; -2*(`llnull` - `llf`)
    llr_pvalue : float
        The chi-squared probability of getting a log-likelihood ratio
        statistic greater than llr.  llr has a chi-squared distribution
        with degrees of freedom `df_model`.
    prsquared : float
        McFadden's pseudo-R-squared. 1 - (`llf`/`llnull`)
    """

    def __init__(self, model, mlefit):
        self.model = model
        self.endog = model.endog
        self.exog = model.exog
        self.nobs = model.endog.shape[0]
        k_extra = getattr(self.model, 'k_extra', 0)
        if hasattr(model, 'df_model') and (not np.isnan(model.df_model)):
            self.df_model = model.df_model
        else:
            df_model = len(mlefit.params) - self.model.k_constant - k_extra
            self.df_model = df_model
            self.model.df_model = df_model
        if hasattr(model, 'df_resid') and (not np.isnan(model.df_resid)):
            self.df_resid = model.df_resid
        else:
            self.df_resid = self.endog.shape[0] - self.df_model - k_extra
            self.model.df_resid = self.df_resid
        self._cache = {}
        self.__dict__.update(mlefit.__dict__)
        k_params = len(mlefit.params)
        if self.df_model + self.model.k_constant + k_extra != k_params:
            warnings.warn('df_model + k_constant + k_extra differs from k_params')
        if self.df_resid != self.nobs - k_params:
            warnings.warn('df_resid differs from nobs - k_params')

    def get_prediction(self, exog=None, which='mean', transform=True, row_labels=None, average=False, agg_weights=None, **kwargs):
        """
        Compute prediction results when endpoint transformation is valid.

        Parameters
        ----------
        exog : array_like, optional
            The values for which you want to predict.
        transform : bool, optional
            If the model was fit via a formula, do you want to pass
            exog through the formula. Default is True. E.g., if you fit
            a model y ~ log(x1) + log(x2), and transform is True, then
            you can pass a data structure that contains x1 and x2 in
            their original form. Otherwise, you'd need to log the data
            first.
        which : str
            Which statistic is to be predicted. Default is "mean".
            The available statistics and options depend on the model.
            see the model.predict docstring
        row_labels : list of str or None
            If row_lables are provided, then they will replace the generated
            labels.
        average : bool
            If average is True, then the mean prediction is computed, that is,
            predictions are computed for individual exog and then the average
            over observation is used.
            If average is False, then the results are the predictions for all
            observations, i.e. same length as ``exog``.
        agg_weights : ndarray, optional
            Aggregation weights, only used if average is True.
            The weights are not normalized.
        **kwargs :
            Some models can take additional keyword arguments, such as offset,
            exposure or additional exog in multi-part models like zero inflated
            models.
            See the predict method of the model for the details.

        Returns
        -------
        prediction_results : PredictionResults
            The prediction results instance contains prediction and prediction
            variance and can on demand calculate confidence intervals and
            summary dataframe for the prediction.

        Notes
        -----
        Status: new in 0.14, experimental
        """
        pass

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """Summarize the Regression Results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model
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