"""
Procedures for fitting marginal regression models to dependent data
using Generalized Estimating Equations.

References
----------
KY Liang and S Zeger. "Longitudinal data analysis using
generalized linear models". Biometrika (1986) 73 (1): 13-22.

S Zeger and KY Liang. "Longitudinal Data Analysis for Discrete and
Continuous Outcomes". Biometrics Vol. 42, No. 1 (Mar., 1986),
pp. 121-130

A Rotnitzky and NP Jewell (1990). "Hypothesis testing of regression
parameters in semiparametric generalized linear models for cluster
correlated data", Biometrika, 77, 485-497.

Xu Guo and Wei Pan (2002). "Small sample performance of the score
test in GEE".
http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf

LA Mancl LA, TA DeRouen (2001). A covariance estimator for GEE with
improved small-sample properties.  Biometrics. 2001 Mar;57(1):126-34.
"""
from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import ConvergenceWarning, DomainWarning, IterationLimitWarning, ValueWarning
import warnings
from statsmodels.graphics._regressionplots_doc import _plot_added_variable_doc, _plot_partial_residuals_doc, _plot_ceres_residuals_doc
from statsmodels.discrete.discrete_margins import _get_margeff_exog, _check_margeff_args, _effects_at, margeff_cov_with_se, _check_at_is_all, _transform_names, _check_discrete_args, _get_dummy_index, _get_count_index

class ParameterConstraint:
    """
    A class for managing linear equality constraints for a parameter
    vector.
    """

    def __init__(self, lhs, rhs, exog):
        """
        Parameters
        ----------
        lhs : ndarray
           A q x p matrix which is the left hand side of the
           constraint lhs * param = rhs.  The number of constraints is
           q >= 1 and p is the dimension of the parameter vector.
        rhs : ndarray
          A 1-dimensional vector of length q which is the right hand
          side of the constraint equation.
        exog : ndarray
          The n x p exognenous data for the full model.
        """
        rhs = np.atleast_1d(rhs.squeeze())
        if rhs.ndim > 1:
            raise ValueError('The right hand side of the constraint must be a vector.')
        if len(rhs) != lhs.shape[0]:
            raise ValueError('The number of rows of the left hand side constraint matrix L must equal the length of the right hand side constraint vector R.')
        self.lhs = lhs
        self.rhs = rhs
        lhs_u, lhs_s, lhs_vt = np.linalg.svd(lhs.T, full_matrices=1)
        self.lhs0 = lhs_u[:, len(lhs_s):]
        self.lhs1 = lhs_u[:, 0:len(lhs_s)]
        self.lhsf = np.hstack((self.lhs0, self.lhs1))
        self.param0 = np.dot(self.lhs1, np.dot(lhs_vt, self.rhs) / lhs_s)
        self._offset_increment = np.dot(exog, self.param0)
        self.orig_exog = exog
        self.exog_fulltrans = np.dot(exog, self.lhsf)

    def offset_increment(self):
        """
        Returns a vector that should be added to the offset vector to
        accommodate the constraint.

        Parameters
        ----------
        exog : array_like
           The exogeneous data for the model.
        """
        return self._offset_increment

    def reduced_exog(self):
        """
        Returns a linearly transformed exog matrix whose columns span
        the constrained model space.

        Parameters
        ----------
        exog : array_like
           The exogeneous data for the model.
        """
        return self.exog_fulltrans[:, :self.lhs0.shape[1]]

    def restore_exog(self):
        """
        Returns the full exog matrix before it was reduced to
        satisfy the constraint.
        """
        return self.orig_exog

    def unpack_param(self, params):
        """
        Converts the parameter vector `params` from reduced to full
        coordinates.
        """
        return np.dot(self.lhsf, np.concatenate((params, self.param0)))

    def unpack_cov(self, bcov):
        """
        Converts the covariance matrix `bcov` from reduced to full
        coordinates.
        """
        m = self.lhs0.shape[1]
        lhs0 = self.lhsf[:, :m]
        return np.dot(lhs0, np.dot(bcov, lhs0.T))
_gee_init_doc = '\n    Marginal regression model fit using Generalized Estimating Equations.\n\n    GEE can be used to fit Generalized Linear Models (GLMs) when the\n    data have a grouped structure, and the observations are possibly\n    correlated within groups but not between groups.\n\n    Parameters\n    ----------\n    endog : array_like\n        1d array of endogenous values (i.e. responses, outcomes,\n        dependent variables, or \'Y\' values).\n    exog : array_like\n        2d array of exogeneous values (i.e. covariates, predictors,\n        independent variables, regressors, or \'X\' values). A `nobs x\n        k` array where `nobs` is the number of observations and `k` is\n        the number of regressors. An intercept is not included by\n        default and should be added by the user. See\n        `statsmodels.tools.add_constant`.\n    groups : array_like\n        A 1d array of length `nobs` containing the group labels.\n    time : array_like\n        A 2d array of time (or other index) values, used by some\n        dependence structures to define similarity relationships among\n        observations within a cluster.\n    family : family class instance\n%(family_doc)s\n    cov_struct : CovStruct class instance\n        The default is Independence.  To specify an exchangeable\n        structure use cov_struct = Exchangeable().  See\n        statsmodels.genmod.cov_struct.CovStruct for more\n        information.\n    offset : array_like\n        An offset to be included in the fit.  If provided, must be\n        an array whose length is the number of rows in exog.\n    dep_data : array_like\n        Additional data passed to the dependence structure.\n    constraint : (ndarray, ndarray)\n        If provided, the constraint is a tuple (L, R) such that the\n        model parameters are estimated under the constraint L *\n        param = R, where L is a q x p matrix and R is a\n        q-dimensional vector.  If constraint is provided, a score\n        test is performed to compare the constrained model to the\n        unconstrained model.\n    update_dep : bool\n        If true, the dependence parameters are optimized, otherwise\n        they are held fixed at their starting values.\n    weights : array_like\n        An array of case weights to use in the analysis.\n    %(extra_params)s\n\n    See Also\n    --------\n    statsmodels.genmod.families.family\n    :ref:`families`\n    :ref:`links`\n\n    Notes\n    -----\n    Only the following combinations make sense for family and link ::\n\n                   + ident log logit probit cloglog pow opow nbinom loglog logc\n      Gaussian     |   x    x                        x\n      inv Gaussian |   x    x                        x\n      binomial     |   x    x    x     x       x     x    x           x      x\n      Poisson      |   x    x                        x\n      neg binomial |   x    x                        x          x\n      gamma        |   x    x                        x\n\n    Not all of these link functions are currently available.\n\n    Endog and exog are references so that if the data they refer\n    to are already arrays and these arrays are changed, endog and\n    exog will change.\n\n    The "robust" covariance type is the standard "sandwich estimator"\n    (e.g. Liang and Zeger (1986)).  It is the default here and in most\n    other packages.  The "naive" estimator gives smaller standard\n    errors, but is only correct if the working correlation structure\n    is correctly specified.  The "bias reduced" estimator of Mancl and\n    DeRouen (Biometrics, 2001) reduces the downward bias of the robust\n    estimator.\n\n    The robust covariance provided here follows Liang and Zeger (1986)\n    and agrees with R\'s gee implementation.  To obtain the robust\n    standard errors reported in Stata, multiply by sqrt(N / (N - g)),\n    where N is the total sample size, and g is the average group size.\n    %(notes)s\n    Examples\n    --------\n    %(example)s\n'
_gee_nointercept = '\n    The nominal and ordinal GEE models should not have an intercept\n    (either implicit or explicit).  Use "0 + " in a formula to\n    suppress the intercept.\n'
_gee_family_doc = '        The default is Gaussian.  To specify the binomial\n        distribution use `family=sm.families.Binomial()`. Each family\n        can take a link instance as an argument.  See\n        statsmodels.genmod.families.family for more information.'
_gee_ordinal_family_doc = '        The only family supported is `Binomial`.  The default `Logit`\n        link may be replaced with `probit` if desired.'
_gee_nominal_family_doc = '        The default value `None` uses a multinomial logit family\n        specifically designed for use with GEE.  Setting this\n        argument to a non-default value is not currently supported.'
_gee_fit_doc = '\n    Fits a marginal regression model using generalized estimating\n    equations (GEE).\n\n    Parameters\n    ----------\n    maxiter : int\n        The maximum number of iterations\n    ctol : float\n        The convergence criterion for stopping the Gauss-Seidel\n        iterations\n    start_params : array_like\n        A vector of starting values for the regression\n        coefficients.  If None, a default is chosen.\n    params_niter : int\n        The number of Gauss-Seidel updates of the mean structure\n        parameters that take place prior to each update of the\n        dependence structure.\n    first_dep_update : int\n        No dependence structure updates occur before this\n        iteration number.\n    cov_type : str\n        One of "robust", "naive", or "bias_reduced".\n    ddof_scale : scalar or None\n        The scale parameter is estimated as the sum of squared\n        Pearson residuals divided by `N - ddof_scale`, where N\n        is the total sample size.  If `ddof_scale` is None, the\n        number of covariates (including an intercept if present)\n        is used.\n    scaling_factor : scalar\n        The estimated covariance of the parameter estimates is\n        scaled by this value.  Default is 1, Stata uses N / (N - g),\n        where N is the total sample size and g is the average group\n        size.\n    scale : str or float, optional\n        `scale` can be None, \'X2\', or a float\n        If a float, its value is used as the scale parameter.\n        The default value is None, which uses `X2` (Pearson\'s\n        chi-square) for Gamma, Gaussian, and Inverse Gaussian.\n        The default is 1 for the Binomial and Poisson families.\n\n    Returns\n    -------\n    An instance of the GEEResults class or subclass\n\n    Notes\n    -----\n    If convergence difficulties occur, increase the values of\n    `first_dep_update` and/or `params_niter`.  Setting\n    `first_dep_update` to a greater value (e.g. ~10-20) causes the\n    algorithm to move close to the GLM solution before attempting\n    to identify the dependence structure.\n\n    For the Gaussian family, there is no benefit to setting\n    `params_niter` to a value greater than 1, since the mean\n    structure parameters converge in one step.\n'
_gee_results_doc = '\n    Attributes\n    ----------\n\n    cov_params_default : ndarray\n        default covariance of the parameter estimates. Is chosen among one\n        of the following three based on `cov_type`\n    cov_robust : ndarray\n        covariance of the parameter estimates that is robust\n    cov_naive : ndarray\n        covariance of the parameter estimates that is not robust to\n        correlation or variance misspecification\n    cov_robust_bc : ndarray\n        covariance of the parameter estimates that is robust and bias\n        reduced\n    converged : bool\n        indicator for convergence of the optimization.\n        True if the norm of the score is smaller than a threshold\n    cov_type : str\n        string indicating whether a "robust", "naive" or "bias_reduced"\n        covariance is used as default\n    fit_history : dict\n        Contains information about the iterations.\n    fittedvalues : ndarray\n        Linear predicted values for the fitted model.\n        dot(exog, params)\n    model : class instance\n        Pointer to GEE model instance that called `fit`.\n    normalized_cov_params : ndarray\n        See GEE docstring\n    params : ndarray\n        The coefficients of the fitted model.  Note that\n        interpretation of the coefficients often depends on the\n        distribution family and the data.\n    scale : float\n        The estimate of the scale / dispersion for the model fit.\n        See GEE.fit for more information.\n    score_norm : float\n        norm of the score at the end of the iterative estimation.\n    bse : ndarray\n        The standard errors of the fitted GEE parameters.\n'
_gee_example = '\n    Logistic regression with autoregressive working dependence:\n\n    >>> import statsmodels.api as sm\n    >>> family = sm.families.Binomial()\n    >>> va = sm.cov_struct.Autoregressive()\n    >>> model = sm.GEE(endog, exog, group, family=family, cov_struct=va)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Use formulas to fit a Poisson GLM with independent working\n    dependence:\n\n    >>> import statsmodels.api as sm\n    >>> fam = sm.families.Poisson()\n    >>> ind = sm.cov_struct.Independence()\n    >>> model = sm.GEE.from_formula("y ~ age + trt + base", "subject",\n                                 data, cov_struct=ind, family=fam)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Equivalent, using the formula API:\n\n    >>> import statsmodels.api as sm\n    >>> import statsmodels.formula.api as smf\n    >>> fam = sm.families.Poisson()\n    >>> ind = sm.cov_struct.Independence()\n    >>> model = smf.gee("y ~ age + trt + base", "subject",\n                    data, cov_struct=ind, family=fam)\n    >>> result = model.fit()\n    >>> print(result.summary())\n'
_gee_ordinal_example = '\n    Fit an ordinal regression model using GEE, with "global\n    odds ratio" dependence:\n\n    >>> import statsmodels.api as sm\n    >>> gor = sm.cov_struct.GlobalOddsRatio("ordinal")\n    >>> model = sm.OrdinalGEE(endog, exog, groups, cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Using formulas:\n\n    >>> import statsmodels.formula.api as smf\n    >>> model = smf.ordinal_gee("y ~ 0 + x1 + x2", groups, data,\n                                    cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n'
_gee_nominal_example = '\n    Fit a nominal regression model using GEE:\n\n    >>> import statsmodels.api as sm\n    >>> import statsmodels.formula.api as smf\n    >>> gor = sm.cov_struct.GlobalOddsRatio("nominal")\n    >>> model = sm.NominalGEE(endog, exog, groups, cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Using formulas:\n\n    >>> import statsmodels.api as sm\n    >>> model = sm.NominalGEE.from_formula("y ~ 0 + x1 + x2", groups,\n                     data, cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n\n    Using the formula API:\n\n    >>> import statsmodels.formula.api as smf\n    >>> model = smf.nominal_gee("y ~ 0 + x1 + x2", groups, data,\n                                cov_struct=gor)\n    >>> result = model.fit()\n    >>> print(result.summary())\n'

class GEE(GLM):
    __doc__ = '    Marginal Regression Model using Generalized Estimating Equations.\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_family_doc, 'example': _gee_example, 'notes': ''}
    cached_means = None

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, exposure=None, dep_data=None, constraint=None, update_dep=True, weights=None, **kwargs):
        if type(self) is GEE:
            self._check_kwargs(kwargs)
        if family is not None:
            if not isinstance(family.link, tuple(family.safe_links)):
                msg = 'The {0} link function does not respect the domain of the {1} family.'
                warnings.warn(msg.format(family.link.__class__.__name__, family.__class__.__name__), DomainWarning)
        groups = np.asarray(groups)
        if 'missing_idx' in kwargs and kwargs['missing_idx'] is not None:
            ii = ~kwargs['missing_idx']
            groups = groups[ii]
            if time is not None:
                time = time[ii]
            if offset is not None:
                offset = offset[ii]
            if exposure is not None:
                exposure = exposure[ii]
            del kwargs['missing_idx']
        self.missing = missing
        self.dep_data = dep_data
        self.constraint = constraint
        self.update_dep = update_dep
        self._fit_history = defaultdict(list)
        super(GEE, self).__init__(endog, exog, groups=groups, time=time, offset=offset, exposure=exposure, weights=weights, dep_data=dep_data, missing=missing, family=family, **kwargs)
        _check_args(self.endog, self.exog, self.groups, self.time, getattr(self, 'offset', None), getattr(self, 'exposure', None))
        self._init_keys.extend(['update_dep', 'constraint', 'family', 'cov_struct'])
        try:
            self._init_keys.remove('freq_weights')
            self._init_keys.remove('var_weights')
        except ValueError:
            pass
        if family is None:
            family = families.Gaussian()
        elif not issubclass(family.__class__, families.Family):
            raise ValueError('GEE: `family` must be a genmod family instance')
        self.family = family
        if cov_struct is None:
            cov_struct = cov_structs.Independence()
        elif not issubclass(cov_struct.__class__, cov_structs.CovStruct):
            raise ValueError('GEE: `cov_struct` must be a genmod cov_struct instance')
        self.cov_struct = cov_struct
        self.constraint = None
        if constraint is not None:
            if len(constraint) != 2:
                raise ValueError('GEE: `constraint` must be a 2-tuple.')
            if constraint[0].shape[1] != self.exog.shape[1]:
                raise ValueError('GEE: the left hand side of the constraint must have the same number of columns as the exog matrix.')
            self.constraint = ParameterConstraint(constraint[0], constraint[1], self.exog)
            if self._offset_exposure is not None:
                self._offset_exposure += self.constraint.offset_increment()
            else:
                self._offset_exposure = self.constraint.offset_increment().copy()
            self.exog = self.constraint.reduced_exog()
        group_labels, ix = np.unique(self.groups, return_inverse=True)
        se = pd.Series(index=np.arange(len(ix)), dtype='int')
        gb = se.groupby(ix).groups
        dk = [(lb, np.asarray(gb[k])) for k, lb in enumerate(group_labels)]
        self.group_indices = dict(dk)
        self.group_labels = group_labels
        self.endog_li = self.cluster_list(self.endog)
        self.exog_li = self.cluster_list(self.exog)
        if self.weights is not None:
            self.weights_li = self.cluster_list(self.weights)
        self.num_group = len(self.endog_li)
        if self.time is not None:
            if self.time.ndim == 1:
                self.time = self.time[:, None]
            self.time_li = self.cluster_list(self.time)
        else:
            self.time_li = [np.arange(len(y), dtype=np.float64)[:, None] for y in self.endog_li]
            self.time = np.concatenate(self.time_li)
        if self._offset_exposure is None or (np.isscalar(self._offset_exposure) and self._offset_exposure == 0.0):
            self.offset_li = None
        else:
            self.offset_li = self.cluster_list(self._offset_exposure)
        if constraint is not None:
            self.constraint.exog_fulltrans_li = self.cluster_list(self.constraint.exog_fulltrans)
        self.family = family
        self.cov_struct.initialize(self)
        group_ns = [len(y) for y in self.endog_li]
        self.nobs = sum(group_ns)
        self.df_model = self.exog.shape[1] - 1
        self.df_resid = self.nobs - self.exog.shape[1]
        maxgroup = max([len(x) for x in self.endog_li])
        if maxgroup == 1:
            self.update_dep = False

    def cluster_list(self, array):
        """
        Returns `array` split into subarrays corresponding to the
        cluster structure.
        """
        pass

    def compare_score_test(self, submodel):
        """
        Perform a score test for the given submodel against this model.

        Parameters
        ----------
        submodel : GEEResults instance
            A fitted GEE model that is a submodel of this model.

        Returns
        -------
        A dictionary with keys "statistic", "p-value", and "df",
        containing the score test statistic, its chi^2 p-value,
        and the degrees of freedom used to compute the p-value.

        Notes
        -----
        The score test can be performed without calling 'fit' on the
        larger model.  The provided submodel must be obtained from a
        fitted GEE.

        This method performs the same score test as can be obtained by
        fitting the GEE with a linear constraint and calling `score_test`
        on the results.

        References
        ----------
        Xu Guo and Wei Pan (2002). "Small sample performance of the score
        test in GEE".
        http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
        """
        pass

    def estimate_scale(self):
        """
        Estimate the dispersion/scale.
        """
        pass

    def mean_deriv(self, exog, lin_pred):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        exog : array_like
           The exogeneous data at which the derivative is computed.
        lin_pred : array_like
           The values of the linear predictor.

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.

        Notes
        -----
        If there is an offset or exposure, it should be added to
        `lin_pred` prior to calling this function.
        """
        pass

    def mean_deriv_exog(self, exog, params, offset_exposure=None):
        """
        Derivative of the expected endog with respect to exog.

        Parameters
        ----------
        exog : array_like
            Values of the independent variables at which the derivative
            is calculated.
        params : array_like
            Parameter values at which the derivative is calculated.
        offset_exposure : array_like, optional
            Combined offset and exposure.

        Returns
        -------
        The derivative of the expected endog with respect to exog.
        """
        pass

    def _update_mean_params(self):
        """
        Returns
        -------
        update : array_like
            The update vector such that params + update is the next
            iterate when solving the score equations.
        score : array_like
            The current value of the score equations, not
            incorporating the scale parameter.  If desired,
            multiply this vector by the scale parameter to
            incorporate the scale.
        """
        pass

    def update_cached_means(self, mean_params):
        """
        cached_means should always contain the most recent calculation
        of the group-wise mean vectors.  This function should be
        called every time the regression parameters are changed, to
        keep the cached means up to date.
        """
        pass

    def _covmat(self):
        """
        Returns the sampling covariance matrix of the regression
        parameters and related quantities.

        Returns
        -------
        cov_robust : array_like
           The robust, or sandwich estimate of the covariance, which
           is meaningful even if the working covariance structure is
           incorrectly specified.
        cov_naive : array_like
           The model-based estimate of the covariance, which is
           meaningful if the covariance structure is correctly
           specified.
        cmat : array_like
           The center matrix of the sandwich expression, used in
           obtaining score test results.
        """
        pass

    def fit_regularized(self, pen_wt, scad_param=3.7, maxiter=100, ddof_scale=None, update_assoc=5, ctol=1e-05, ztol=0.001, eps=1e-06, scale=None):
        """
        Regularized estimation for GEE.

        Parameters
        ----------
        pen_wt : float
            The penalty weight (a non-negative scalar).
        scad_param : float
            Non-negative scalar determining the shape of the Scad
            penalty.
        maxiter : int
            The maximum number of iterations.
        ddof_scale : int
            Value to subtract from `nobs` when calculating the
            denominator degrees of freedom for t-statistics, defaults
            to the number of columns in `exog`.
        update_assoc : int
            The dependence parameters are updated every `update_assoc`
            iterations of the mean structure parameter updates.
        ctol : float
            Convergence criterion, default is one order of magnitude
            smaller than proposed in section 3.1 of Wang et al.
        ztol : float
            Coefficients smaller than this value are treated as
            being zero, default is based on section 5 of Wang et al.
        eps : non-negative scalar
            Numerical constant, see section 3.2 of Wang et al.
        scale : float or string
            If a float, this value is used as the scale parameter.
            If "X2", the scale parameter is always estimated using
            Pearson's chi-square method (e.g. as in a quasi-Poisson
            analysis).  If None, the default approach for the family
            is used to estimate the scale parameter.

        Returns
        -------
        GEEResults instance.  Note that not all methods of the results
        class make sense when the model has been fit with regularization.

        Notes
        -----
        This implementation assumes that the link is canonical.

        References
        ----------
        Wang L, Zhou J, Qu A. (2012). Penalized generalized estimating
        equations for high-dimensional longitudinal data analysis.
        Biometrics. 2012 Jun;68(2):353-60.
        doi: 10.1111/j.1541-0420.2011.01678.x.
        https://www.ncbi.nlm.nih.gov/pubmed/21955051
        http://users.stat.umn.edu/~wangx346/research/GEE_selection.pdf
        """
        pass

    def _handle_constraint(self, mean_params, bcov):
        """
        Expand the parameter estimate `mean_params` and covariance matrix
        `bcov` to the coordinate system of the unconstrained model.

        Parameters
        ----------
        mean_params : array_like
            A parameter vector estimate for the reduced model.
        bcov : array_like
            The covariance matrix of mean_params.

        Returns
        -------
        mean_params : array_like
            The input parameter vector mean_params, expanded to the
            coordinate system of the full model
        bcov : array_like
            The input covariance matrix bcov, expanded to the
            coordinate system of the full model
        """
        pass

    def _update_assoc(self, params):
        """
        Update the association parameters
        """
        pass

    def _derivative_exog(self, params, exog=None, transform='dydx', dummy_idx=None, count_idx=None):
        """
        For computing marginal effects, returns dF(XB) / dX where F(.)
        is the fitted mean.

        transform can be 'dydx', 'dyex', 'eydx', or 'eyex'.

        Not all of these make sense in the presence of discrete regressors,
        but checks are done in the results in get_margeff.
        """
        pass

    def qic(self, params, scale, cov_params, n_step=1000):
        """
        Returns quasi-information criteria and quasi-likelihood values.

        Parameters
        ----------
        params : array_like
            The GEE estimates of the regression parameters.
        scale : scalar
            Estimated scale parameter
        cov_params : array_like
            An estimate of the covariance matrix for the
            model parameters.  Conventionally this is the robust
            covariance matrix.
        n_step : integer
            The number of points in the trapezoidal approximation
            to the quasi-likelihood function.

        Returns
        -------
        ql : scalar
            The quasi-likelihood value
        qic : scalar
            A QIC that can be used to compare the mean and covariance
            structures of the model.
        qicu : scalar
            A simplified QIC that can be used to compare mean structures
            but not covariance structures

        Notes
        -----
        The quasi-likelihood used here is obtained by numerically evaluating
        Wedderburn's integral representation of the quasi-likelihood function.
        This approach is valid for all families and  links.  Many other
        packages use analytical expressions for quasi-likelihoods that are
        valid in special cases where the link function is canonical.  These
        analytical expressions may omit additive constants that only depend
        on the data.  Therefore, the numerical values of our QL and QIC values
        will differ from the values reported by other packages.  However only
        the differences between two QIC values calculated for different models
        using the same data are meaningful.  Our QIC should produce the same
        QIC differences as other software.

        When using the QIC for models with unknown scale parameter, use a
        common estimate of the scale parameter for all models being compared.

        References
        ----------
        .. [*] W. Pan (2001).  Akaike's information criterion in generalized
               estimating equations.  Biometrics (57) 1.
        """
        pass

class GEEResults(GLMResults):
    __doc__ = 'This class summarizes the fit of a marginal regression model using GEE.\n' + _gee_results_doc

    def __init__(self, model, params, cov_params, scale, cov_type='robust', use_t=False, regularized=False, **kwds):
        super(GEEResults, self).__init__(model, params, normalized_cov_params=cov_params, scale=scale)
        self.df_resid = model.df_resid
        self.df_model = model.df_model
        self.family = model.family
        attr_kwds = kwds.pop('attr_kwds', {})
        self.__dict__.update(attr_kwds)
        if not (hasattr(self, 'cov_type') and hasattr(self, 'cov_params_default')):
            self.cov_type = cov_type
            covariance_type = self.cov_type.lower()
            allowed_covariances = ['robust', 'naive', 'bias_reduced']
            if covariance_type not in allowed_covariances:
                msg = 'GEE: `cov_type` must be one of ' + ', '.join(allowed_covariances)
                raise ValueError(msg)
            if cov_type == 'robust':
                cov = self.cov_robust
            elif cov_type == 'naive':
                cov = self.cov_naive
            elif cov_type == 'bias_reduced':
                cov = self.cov_robust_bc
            self.cov_params_default = cov
        elif self.cov_type != cov_type:
            raise ValueError('cov_type in argument is different from already attached cov_type')

    @cache_readonly
    def resid(self):
        """
        The response residuals.
        """
        pass

    def standard_errors(self, cov_type='robust'):
        """
        This is a convenience function that returns the standard
        errors for any covariance type.  The value of `bse` is the
        standard errors for whichever covariance type is specified as
        an argument to `fit` (defaults to "robust").

        Parameters
        ----------
        cov_type : str
            One of "robust", "naive", or "bias_reduced".  Determines
            the covariance used to compute standard errors.  Defaults
            to "robust".
        """
        pass

    def score_test(self):
        """
        Return the results of a score test for a linear constraint.

        Returns
        -------
        A\x7fdictionary containing the p-value, the test statistic,
        and the degrees of freedom for the score test.

        Notes
        -----
        See also GEE.compare_score_test for an alternative way to perform
        a score test.  GEEResults.score_test is more general, in that it
        supports testing arbitrary linear equality constraints.   However
        GEE.compare_score_test might be easier to use when comparing
        two explicit models.

        References
        ----------
        Xu Guo and Wei Pan (2002). "Small sample performance of the score
        test in GEE".
        http://www.sph.umn.edu/faculty1/wp-content/uploads/2012/11/rr2002-013.pdf
        """
        pass

    @cache_readonly
    def resid_split(self):
        """
        Returns the residuals, the endogeneous data minus the fitted
        values from the model.  The residuals are returned as a list
        of arrays containing the residuals for each cluster.
        """
        pass

    @cache_readonly
    def resid_centered(self):
        """
        Returns the residuals centered within each group.
        """
        pass

    @cache_readonly
    def resid_centered_split(self):
        """
        Returns the residuals centered within each group.  The
        residuals are returned as a list of arrays containing the
        centered residuals for each cluster.
        """
        pass

    def qic(self, scale=None, n_step=1000):
        """
        Returns the QIC and QICu information criteria.

        See GEE.qic for documentation.
        """
        pass
    split_resid = resid_split
    centered_resid = resid_centered
    split_centered_resid = resid_centered_split

    def conf_int(self, alpha=0.05, cols=None, cov_type=None):
        """
        Returns confidence intervals for the fitted parameters.

        Parameters
        ----------
        alpha : float, optional
             The `alpha` level for the confidence interval.  i.e., The
             default `alpha` = .05 returns a 95% confidence interval.
        cols : array_like, optional
             `cols` specifies which confidence intervals to return
        cov_type : str
             The covariance type used for computing standard errors;
             must be one of 'robust', 'naive', and 'bias reduced'.
             See `GEE` for details.

        Notes
        -----
        The confidence interval is based on the Gaussian distribution.
        """
        pass

    def summary(self, yname=None, xname=None, title=None, alpha=0.05):
        """
        Summarize the GEE regression results

        Parameters
        ----------
        yname : str, optional
            Default is `y`
        xname : list[str], optional
            Names for the exogenous variables, default is `var_#` for ## in
            the number of regressors. Must match the number of parameters in
            the model
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            significance level for the confidence intervals
        cov_type : str
            The covariance type used to compute the standard errors;
            one of 'robust' (the usual robust sandwich-type covariance
            estimate), 'naive' (ignores dependence), and 'bias
            reduced' (the Mancl/DeRouen estimate).

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be
            printed or converted to various output formats.

        See Also
        --------
        statsmodels.iolib.summary.Summary : class to hold summary results
        """
        pass

    def get_margeff(self, at='overall', method='dydx', atexog=None, dummy=False, count=False):
        """Get marginal effects of the fitted model.

        Parameters
        ----------
        at : str, optional
            Options are:

            - 'overall', The average of the marginal effects at each
              observation.
            - 'mean', The marginal effects at the mean of each regressor.
            - 'median', The marginal effects at the median of each regressor.
            - 'zero', The marginal effects at zero for each regressor.
            - 'all', The marginal effects at each observation. If `at` is 'all'
              only margeff will be available.

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
            for discrete variables.
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
        effects : ndarray
            the marginal effect corresponding to the input options

        Notes
        -----
        When using after Poisson, returns the expected number of events
        per period, assuming that the model is loglinear.
        """
        pass

    def plot_isotropic_dependence(self, ax=None, xpoints=10, min_n=50):
        """
        Create a plot of the pairwise products of within-group
        residuals against the corresponding time differences.  This
        plot can be used to assess the possible form of an isotropic
        covariance structure.

        Parameters
        ----------
        ax : AxesSubplot
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        xpoints : scalar or array_like
            If scalar, the number of points equally spaced points on
            the time difference axis used to define bins for
            calculating local means.  If an array, the specific points
            that define the bins.
        min_n : int
            The minimum sample size in a bin for the mean residual
            product to be included on the plot.
        """
        pass

    def sensitivity_params(self, dep_params_first, dep_params_last, num_steps):
        """
        Refits the GEE model using a sequence of values for the
        dependence parameters.

        Parameters
        ----------
        dep_params_first : array_like
            The first dep_params in the sequence
        dep_params_last : array_like
            The last dep_params in the sequence
        num_steps : int
            The number of dep_params in the sequence

        Returns
        -------
        results : array_like
            The GEEResults objects resulting from the fits.
        """
        pass
    params_sensitivity = sensitivity_params

class GEEResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {'centered_resid': 'rows'}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs, _attrs)
wrap.populate_wrapper(GEEResultsWrapper, GEEResults)

class OrdinalGEE(GEE):
    __doc__ = '    Ordinal Response Marginal Regression Model using GEE\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_ordinal_family_doc, 'example': _gee_ordinal_example, 'notes': _gee_nointercept}

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, dep_data=None, constraint=None, **kwargs):
        if family is None:
            family = families.Binomial()
        elif not isinstance(family, families.Binomial):
            raise ValueError('ordinal GEE must use a Binomial family')
        if cov_struct is None:
            cov_struct = cov_structs.OrdinalIndependence()
        endog, exog, groups, time, offset = self.setup_ordinal(endog, exog, groups, time, offset)
        super(OrdinalGEE, self).__init__(endog, exog, groups, time, family, cov_struct, missing, offset, dep_data, constraint)

    def setup_ordinal(self, endog, exog, groups, time, offset):
        """
        Restructure ordinal data as binary indicators so that they can
        be analyzed using Generalized Estimating Equations.
        """
        pass

class OrdinalGEEResults(GEEResults):
    __doc__ = 'This class summarizes the fit of a marginal regression modelfor an ordinal response using GEE.\n' + _gee_results_doc

    def plot_distribution(self, ax=None, exog_values=None):
        """
        Plot the fitted probabilities of endog in an ordinal model,
        for specified values of the predictors.

        Parameters
        ----------
        ax : AxesSubplot
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        exog_values : array_like
            A list of dictionaries, with each dictionary mapping
            variable names to values at which the variable is held
            fixed.  The values P(endog=y | exog) are plotted for all
            possible values of y, at the given exog value.  Variables
            not included in a dictionary are held fixed at the mean
            value.

        Example:
        --------
        We have a model with covariates 'age' and 'sex', and wish to
        plot the probabilities P(endog=y | exog) for males (sex=0) and
        for females (sex=1), as separate paths on the plot.  Since
        'age' is not included below in the map, it is held fixed at
        its mean value.

        >>> ev = [{"sex": 1}, {"sex": 0}]
        >>> rslt.distribution_plot(exog_values=ev)
        """
        pass

def _score_test_submodel(par, sub):
    """
    Return transformation matrices for design matrices.

    Parameters
    ----------
    par : instance
        The parent model
    sub : instance
        The sub-model

    Returns
    -------
    qm : array_like
        Matrix mapping the design matrix of the parent to the design matrix
        for the sub-model.
    qc : array_like
        Matrix mapping the design matrix of the parent to the orthogonal
        complement of the columnspace of the submodel in the columnspace
        of the parent.

    Notes
    -----
    Returns None, None if the provided submodel is not actually a submodel.
    """
    pass

class OrdinalGEEResultsWrapper(GEEResultsWrapper):
    pass
wrap.populate_wrapper(OrdinalGEEResultsWrapper, OrdinalGEEResults)

class NominalGEE(GEE):
    __doc__ = '    Nominal Response Marginal Regression Model using GEE.\n' + _gee_init_doc % {'extra_params': base._missing_param_doc, 'family_doc': _gee_nominal_family_doc, 'example': _gee_nominal_example, 'notes': _gee_nointercept}

    def __init__(self, endog, exog, groups, time=None, family=None, cov_struct=None, missing='none', offset=None, dep_data=None, constraint=None, **kwargs):
        endog, exog, groups, time, offset = self.setup_nominal(endog, exog, groups, time, offset)
        if family is None:
            family = _Multinomial(self.ncut + 1)
        if cov_struct is None:
            cov_struct = cov_structs.NominalIndependence()
        super(NominalGEE, self).__init__(endog, exog, groups, time, family, cov_struct, missing, offset, dep_data, constraint)

    def setup_nominal(self, endog, exog, groups, time, offset):
        """
        Restructure nominal data as binary indicators so that they can
        be analyzed using Generalized Estimating Equations.
        """
        pass

    def mean_deriv(self, exog, lin_pred):
        """
        Derivative of the expected endog with respect to the parameters.

        Parameters
        ----------
        exog : array_like
           The exogeneous data at which the derivative is computed,
           number of rows must be a multiple of `ncut`.
        lin_pred : array_like
           The values of the linear predictor, length must be multiple
           of `ncut`.

        Returns
        -------
        The derivative of the expected endog with respect to the
        parameters.
        """
        pass

    def mean_deriv_exog(self, exog, params, offset_exposure=None):
        """
        Derivative of the expected endog with respect to exog for the
        multinomial model, used in analyzing marginal effects.

        Parameters
        ----------
        exog : array_like
           The exogeneous data at which the derivative is computed,
           number of rows must be a multiple of `ncut`.
        lpr : array_like
           The linear predictor values, length must be multiple of
           `ncut`.

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to exog.

        Notes
        -----
        offset_exposure must be set at None for the multinomial family.
        """
        pass

class NominalGEEResults(GEEResults):
    __doc__ = 'This class summarizes the fit of a marginal regression modelfor a nominal response using GEE.\n' + _gee_results_doc

    def plot_distribution(self, ax=None, exog_values=None):
        """
        Plot the fitted probabilities of endog in an nominal model,
        for specified values of the predictors.

        Parameters
        ----------
        ax : AxesSubplot
            An axes on which to draw the graph.  If None, new
            figure and axes objects are created
        exog_values : array_like
            A list of dictionaries, with each dictionary mapping
            variable names to values at which the variable is held
            fixed.  The values P(endog=y | exog) are plotted for all
            possible values of y, at the given exog value.  Variables
            not included in a dictionary are held fixed at the mean
            value.

        Example:
        --------
        We have a model with covariates 'age' and 'sex', and wish to
        plot the probabilities P(endog=y | exog) for males (sex=0) and
        for females (sex=1), as separate paths on the plot.  Since
        'age' is not included below in the map, it is held fixed at
        its mean value.

        >>> ex = [{"sex": 1}, {"sex": 0}]
        >>> rslt.distribution_plot(exog_values=ex)
        """
        pass

class NominalGEEResultsWrapper(GEEResultsWrapper):
    pass
wrap.populate_wrapper(NominalGEEResultsWrapper, NominalGEEResults)

class _MultinomialLogit(Link):
    """
    The multinomial logit transform, only for use with GEE.

    Notes
    -----
    The data are assumed coded as binary indicators, where each
    observed multinomial value y is coded as I(y == S[0]), ..., I(y ==
    S[-1]), where S is the set of possible response labels, excluding
    the largest one.  Thererefore functions in this class should only
    be called using vector argument whose length is a multiple of |S|
    = ncut, which is an argument to be provided when initializing the
    class.

    call and derivative use a private method _clean to trim p by 1e-10
    so that p is in (0, 1)
    """

    def __init__(self, ncut):
        self.ncut = ncut

    def inverse(self, lpr):
        """
        Inverse of the multinomial logit transform, which gives the
        expected values of the data as a function of the linear
        predictors.

        Parameters
        ----------
        lpr : array_like (length must be divisible by `ncut`)
            The linear predictors

        Returns
        -------
        prob : ndarray
            Probabilities, or expected values
        """
        pass

class _Multinomial(families.Family):
    """
    Pseudo-link function for fitting nominal multinomial models with
    GEE.  Not for use outside the GEE class.
    """
    links = [_MultinomialLogit]
    variance = varfuncs.binary
    safe_links = [_MultinomialLogit]

    def __init__(self, nlevels, check_link=True):
        """
        Parameters
        ----------
        nlevels : int
            The number of distinct categories for the multinomial
            distribution.
        """
        self._check_link = check_link
        self.initialize(nlevels)

class GEEMargins:
    """
    Estimated marginal effects for a regression model fit with GEE.

    Parameters
    ----------
    results : GEEResults instance
        The results instance of a fitted discrete choice model
    args : tuple
        Args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    kwargs : dict
        Keyword args are passed to `get_margeff`. This is the same as
        results.get_margeff. See there for more information.
    """

    def __init__(self, results, args, kwargs={}):
        self._cache = {}
        self.results = results
        self.get_margeff(*args, **kwargs)

    def summary_frame(self, alpha=0.05):
        """
        Returns a DataFrame summarizing the marginal effects.

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        frame : DataFrames
            A DataFrame summarizing the marginal effects.
        """
        pass

    def conf_int(self, alpha=0.05):
        """
        Returns the confidence intervals of the marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        conf_int : ndarray
            An array with lower, upper confidence intervals for the marginal
            effects.
        """
        pass

    def summary(self, alpha=0.05):
        """
        Returns a summary table for marginal effects

        Parameters
        ----------
        alpha : float
            Number between 0 and 1. The confidence intervals have the
            probability 1-alpha.

        Returns
        -------
        Summary : SummaryTable
            A SummaryTable instance
        """
        pass
