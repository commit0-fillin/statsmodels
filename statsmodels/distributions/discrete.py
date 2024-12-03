import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel

class genpoisson_p_gen(rv_discrete):
    """Generalized Poisson distribution
    """
genpoisson_p = genpoisson_p_gen(name='genpoisson_p', longname='Generalized Poisson')

class zipoisson_gen(rv_discrete):
    """Zero Inflated Poisson distribution
    """
zipoisson = zipoisson_gen(name='zipoisson', longname='Zero Inflated Poisson')

class zigeneralizedpoisson_gen(rv_discrete):
    """Zero Inflated Generalized Poisson distribution
    """
zigenpoisson = zigeneralizedpoisson_gen(name='zigenpoisson', longname='Zero Inflated Generalized Poisson')

class zinegativebinomial_gen(rv_discrete):
    """Zero Inflated Generalized Negative Binomial distribution
    """
zinegbin = zinegativebinomial_gen(name='zinegbin', longname='Zero Inflated Generalized Negative Binomial')

class truncatedpoisson_gen(rv_discrete):
    """Truncated Poisson discrete random variable
    """
truncatedpoisson = truncatedpoisson_gen(name='truncatedpoisson', longname='Truncated Poisson')

class truncatednegbin_gen(rv_discrete):
    """Truncated Generalized Negative Binomial (NB-P) discrete random variable
    """
truncatednegbin = truncatednegbin_gen(name='truncatednegbin', longname='Truncated Generalized Negative Binomial')

class DiscretizedCount(rv_discrete):
    """Count distribution based on discretized distribution

    Parameters
    ----------
    distr : distribution instance
    d_offset : float
        Offset for integer interval, default is zero.
        The discrete random variable is ``y = floor(x + offset)`` where x is
        the continuous random variable.
        Warning: not verified for all methods.
    add_scale : bool
        If True (default), then the scale of the base distribution is added
        as parameter for the discrete distribution. The scale parameter is in
        the last position.
    kwds : keyword arguments
        The extra keyword arguments are used delegated to the ``__init__`` of
        the super class.
        Their usage has not been checked, e.g. currently the support of the
        distribution is assumed to be all non-negative integers.

    Notes
    -----
    `loc` argument is currently not supported, scale is not available for
    discrete distributions in scipy. The scale parameter of the underlying
    continuous distribution is the last shape parameter in this
    DiscretizedCount distribution if ``add_scale`` is True.

    The implementation was based mainly on [1]_ and [2]_. However, many new
    discrete distributions have been developed based on the approach that we
    use here. Note, that in many cases authors reparameterize the distribution,
    while this class inherits the parameterization from the underlying
    continuous distribution.

    References
    ----------
    .. [1] Chakraborty, Subrata, and Dhrubajyoti Chakravarty. "Discrete gamma
       distributions: Properties and parameter estimations." Communications in
       Statistics-Theory and Methods 41, no. 18 (2012): 3301-3324.

    .. [2] Alzaatreh, Ayman, Carl Lee, and Felix Famoye. 2012. “On the Discrete
       Analogues of Continuous Distributions.” Statistical Methodology 9 (6):
       589–603.


    """

    def __new__(cls, *args, **kwds):
        return super(rv_discrete, cls).__new__(cls)

    def __init__(self, distr, d_offset=0, add_scale=True, **kwds):
        self.distr = distr
        self.d_offset = d_offset
        self._ctor_param = distr._ctor_param
        self.add_scale = add_scale
        if distr.shapes is not None:
            self.k_shapes = len(distr.shapes.split(','))
            if add_scale:
                kwds.update({'shapes': distr.shapes + ', s'})
                self.k_shapes += 1
        elif add_scale:
            kwds.update({'shapes': 's'})
            self.k_shapes = 1
        else:
            self.k_shapes = 0
        super().__init__(**kwds)

class DiscretizedModel(GenericLikelihoodModel):
    """experimental model to fit discretized distribution

    Count models based on discretized distributions can be used to model
    data that is under- or over-dispersed relative to Poisson or that has
    heavier tails.

    Parameters
    ----------
    endog : array_like, 1-D
        Univariate data for fitting the distribution.
    exog : None
        Explanatory variables are not supported. The ``exog`` argument is
        only included for consistency in the signature across models.
    distr : DiscretizedCount instance
        (required) Instance of a DiscretizedCount distribution.

    See Also
    --------
    DiscretizedCount

    Examples
    --------
    >>> from scipy import stats
    >>> from statsmodels.distributions.discrete import (
            DiscretizedCount, DiscretizedModel)

    >>> dd = DiscretizedCount(stats.gamma)
    >>> mod = DiscretizedModel(y, distr=dd)
    >>> res = mod.fit()
    >>> probs = res.predict(which="probs", k_max=5)

    """

    def __init__(self, endog, exog=None, distr=None):
        if exog is not None:
            raise ValueError('exog is not supported')
        super().__init__(endog, exog, distr=distr)
        self._init_keys.append('distr')
        self.df_resid = len(endog) - distr.k_shapes
        self.df_model = 0
        self.k_extra = distr.k_shapes
        self.k_constant = 0
        self.nparams = distr.k_shapes
        self.start_params = 0.5 * np.ones(self.nparams)

    def get_distr(self, params):
        """frozen distribution instance of the discrete distribution.
        """
        if self.distr is None:
            raise ValueError("Distribution not specified")
        
        # Unpack parameters
        if self.distr.add_scale:
            shape_params = params[:-1]
            scale = params[-1]
        else:
            shape_params = params
            scale = 1.0
        
        # Create a frozen instance of the distribution
        frozen_distr = self.distr.distr(*shape_params, scale=scale)
        
        # Return the discretized distribution
        return self.distr(frozen_distr, d_offset=self.distr.d_offset)
