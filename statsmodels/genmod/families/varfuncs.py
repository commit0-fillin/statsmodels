"""
Variance functions for use with the link functions in statsmodels.family.links
"""
import numpy as np
FLOAT_EPS = np.finfo(float).eps

class VarianceFunction:
    """
    Relates the variance of a random variable to its mean. Defaults to 1.

    Methods
    -------
    call
        Returns an array of ones that is the same shape as `mu`

    Notes
    -----
    After a variance function is initialized, its call method can be used.

    Alias for VarianceFunction:
    constant = VarianceFunction()

    See Also
    --------
    statsmodels.genmod.families.family
    """

    def __call__(self, mu):
        """
        Default variance function

        Parameters
        ----------
        mu : array_like
            mean parameters

        Returns
        -------
        v : ndarray
            ones(mu.shape)
        """
        mu = np.asarray(mu)
        return np.ones(mu.shape, np.float64)

    def deriv(self, mu):
        """
        Derivative of the variance function v'(mu)
        """
        return np.zeros_like(mu)
constant = VarianceFunction()
constant.__doc__ = '\nThe call method of constant returns a constant variance, i.e., a vector of\nones.\n\nconstant is an alias of VarianceFunction()\n'

class Power:
    """
    Power variance function

    Parameters
    ----------
    power : float
        exponent used in power variance function

    Methods
    -------
    call
        Returns the power variance

    Notes
    -----
    Formulas
       V(mu) = numpy.fabs(mu)**power

    Aliases for Power:
    mu = Power()
    mu_squared = Power(power=2)
    mu_cubed = Power(power=3)
    """

    def __init__(self, power=1.0):
        self.power = power

    def __call__(self, mu):
        """
        Power variance function

        Parameters
        ----------
        mu : array_like
            mean parameters

        Returns
        -------
        variance : ndarray
            numpy.fabs(mu)**self.power
        """
        return np.power(np.fabs(mu), self.power)

    def deriv(self, mu):
        """
        Derivative of the variance function v'(mu)

        May be undefined at zero.
        """
        return self.power * np.power(np.fabs(mu), self.power - 1) * np.sign(mu)
mu = Power()
mu.__doc__ = '\nReturns np.fabs(mu)\n\nNotes\n-----\nThis is an alias of Power()\n'
mu_squared = Power(power=2)
mu_squared.__doc__ = '\nReturns np.fabs(mu)**2\n\nNotes\n-----\nThis is an alias of statsmodels.family.links.Power(power=2)\n'
mu_cubed = Power(power=3)
mu_cubed.__doc__ = '\nReturns np.fabs(mu)**3\n\nNotes\n-----\nThis is an alias of statsmodels.family.links.Power(power=3)\n'

class Binomial:
    """
    Binomial variance function

    Parameters
    ----------
    n : int, optional
        The number of trials for a binomial variable.  The default is 1 for
        p in (0,1)

    Methods
    -------
    call
        Returns the binomial variance

    Notes
    -----
    Formulas :

       V(mu) = p * (1 - p) * n

    where p = mu / n

    Alias for Binomial:
    binary = Binomial()

    A private method _clean trims the data by machine epsilon so that p is
    in (0,1)
    """

    def __init__(self, n=1):
        self.n = n

    def __call__(self, mu):
        """
        Binomial variance function

        Parameters
        ----------
        mu : array_like
            mean parameters

        Returns
        -------
        variance : ndarray
           variance = mu/n * (1 - mu/n) * self.n
        """
        p = self._clean(mu / self.n)
        return p * (1 - p) * self.n

    def deriv(self, mu):
        """
        Derivative of the variance function v'(mu)
        """
        p = self._clean(mu / self.n)
        return (1 - 2 * p)
binary = Binomial()
binary.__doc__ = '\nThe binomial variance function for n = 1\n\nNotes\n-----\nThis is an alias of Binomial(n=1)\n'

class NegativeBinomial:
    """
    Negative binomial variance function

    Parameters
    ----------
    alpha : float
        The ancillary parameter for the negative binomial variance function.
        `alpha` is assumed to be nonstochastic.  The default is 1.

    Methods
    -------
    call
        Returns the negative binomial variance

    Notes
    -----
    Formulas :

       V(mu) = mu + alpha*mu**2

    Alias for NegativeBinomial:
    nbinom = NegativeBinomial()

    A private method _clean trims the data by machine epsilon so that p is
    in (0,inf)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, mu):
        """
        Negative binomial variance function

        Parameters
        ----------
        mu : array_like
            mean parameters

        Returns
        -------
        variance : ndarray
            variance = mu + alpha*mu**2
        """
        p = self._clean(mu)
        return p + self.alpha * p ** 2

    def deriv(self, mu):
        """
        Derivative of the negative binomial variance function.
        """
        return 1 + 2 * self.alpha * self._clean(mu)
nbinom = NegativeBinomial()
nbinom.__doc__ = '\nNegative Binomial variance function.\n\nNotes\n-----\nThis is an alias of NegativeBinomial(alpha=1.)\n'
