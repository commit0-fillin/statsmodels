import numpy as np

def _cabs(x):
    """absolute value function that changes complex sign based on real sign

    This could be useful for complex step derivatives of functions that
    need abs. Not yet used.
    """
    return np.where(np.real(x) < 0, -x, x)

class RobustNorm:
    """
    The parent class for the norms used for robust regression.

    Lays out the methods expected of the robust norms to be used
    by statsmodels.RLM.

    See Also
    --------
    statsmodels.rlm

    Notes
    -----
    Currently only M-estimators are available.

    References
    ----------
    PJ Huber.  'Robust Statistics' John Wiley and Sons, Inc., New York, 1981.

    DC Montgomery, EA Peck. 'Introduction to Linear Regression Analysis',
        John Wiley and Sons, Inc., New York, 2001.

    R Venables, B Ripley. 'Modern Applied Statistics in S'
        Springer, New York, 2002.
    """

    def rho(self, z):
        """
        The robust criterion estimator function.

        Abstract method:

        -2 loglike used in M-estimator
        """
        raise NotImplementedError("rho method must be implemented in subclass")

    def psi(self, z):
        """
        Derivative of rho.  Sometimes referred to as the influence function.

        Abstract method:

        psi = rho'
        """
        raise NotImplementedError("psi method must be implemented in subclass")

    def weights(self, z):
        """
        Returns the value of psi(z) / z

        Abstract method:

        psi(z) / z
        """
        return self.psi(z) / z

    def psi_deriv(self, z):
        """
        Derivative of psi.  Used to obtain robust covariance matrix.

        See statsmodels.rlm for more information.

        Abstract method:

        psi_derive = psi'
        """
        raise NotImplementedError("psi_deriv method must be implemented in subclass")

    def __call__(self, z):
        """
        Returns the value of estimator rho applied to an input
        """
        return self.rho(z)

class LeastSquares(RobustNorm):
    """
    Least squares rho for M-estimation and its derived functions.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def rho(self, z):
        """
        The least squares estimator rho function

        Parameters
        ----------
        z : ndarray
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = (1/2.)*z**2
        """
        return 0.5 * z**2

    def psi(self, z):
        """
        The psi function for the least squares estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z
        """
        return z

    def weights(self, z):
        """
        The least squares estimator weighting function for the IRLS algorithm.

        The psi function scaled by the input z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = np.ones(z.shape)
        """
        return np.ones_like(z)

    def psi_deriv(self, z):
        """
        The derivative of the least squares psi function.

        Returns
        -------
        psi_deriv : ndarray
            ones(z.shape)

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.ones_like(z)

class HuberT(RobustNorm):
    """
    Huber's T for M estimation.

    Parameters
    ----------
    t : float, optional
        The tuning constant for Huber's t function. The default value is
        1.345.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, t=1.345):
        self.t = t

    def _subset(self, z):
        """
        Huber's T is defined piecewise over the range for z
        """
        return np.less_equal(np.abs(z), self.t)

    def rho(self, z):
        """
        The robust criterion function for Huber's t.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = .5*z**2            for \\|z\\| <= t

            rho(z) = \\|z\\|*t - .5*t**2    for \\|z\\| > t
        """
        z = np.asarray(z)
        subset = self._subset(z)
        return np.where(subset,
                        0.5 * z**2,
                        self.t * np.abs(z) - 0.5 * self.t**2)

    def psi(self, z):
        """
        The psi function for Huber's t estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z      for \\|z\\| <= t

            psi(z) = sign(z)*t for \\|z\\| > t
        """
        z = np.asarray(z)
        subset = self._subset(z)
        return np.where(subset, z, self.t * np.sign(z))

    def weights(self, z):
        """
        Huber's t weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = 1          for \\|z\\| <= t

            weights(z) = t/\\|z\\|      for \\|z\\| > t
        """
        z = np.asarray(z)
        subset = self._subset(z)
        return np.where(subset, 1, self.t / np.abs(z))

    def psi_deriv(self, z):
        """
        The derivative of Huber's t psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        return np.where(self._subset(z), 1, 0)

class RamsayE(RobustNorm):
    """
    Ramsay's Ea for M estimation.

    Parameters
    ----------
    a : float, optional
        The tuning constant for Ramsay's Ea function.  The default value is
        0.3.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, a=0.3):
        self.a = a

    def rho(self, z):
        """
        The robust criterion function for Ramsay's Ea.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = a**-2 * (1 - exp(-a*\\|z\\|)*(1 + a*\\|z\\|))
        """
        pass

    def psi(self, z):
        """
        The psi function for Ramsay's Ea estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z*exp(-a*\\|z\\|)
        """
        pass

    def weights(self, z):
        """
        Ramsay's Ea weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = exp(-a*\\|z\\|)
        """
        pass

    def psi_deriv(self, z):
        """
        The derivative of Ramsay's Ea psi function.

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        pass

class AndrewWave(RobustNorm):
    """
    Andrew's wave for M estimation.

    Parameters
    ----------
    a : float, optional
        The tuning constant for Andrew's Wave function.  The default value is
        1.339.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, a=1.339):
        self.a = a

    def _subset(self, z):
        """
        Andrew's wave is defined piecewise over the range of z.
        """
        pass

    def rho(self, z):
        """
        The robust criterion function for Andrew's wave.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            The elements of rho are defined as:

            .. math::

                rho(z) & = a^2 *(1-cos(z/a)), |z| \\leq a\\pi \\\\
                rho(z) & = 2a, |z|>q\\pi
        """
        pass

    def psi(self, z):
        """
        The psi function for Andrew's wave

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = a * sin(z/a)   for \\|z\\| <= a*pi

            psi(z) = 0              for \\|z\\| > a*pi
        """
        pass

    def weights(self, z):
        """
        Andrew's wave weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = sin(z/a) / (z/a)     for \\|z\\| <= a*pi

            weights(z) = 0                    for \\|z\\| > a*pi
        """
        pass

    def psi_deriv(self, z):
        """
        The derivative of Andrew's wave psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        pass

class TrimmedMean(RobustNorm):
    """
    Trimmed mean function for M-estimation.

    Parameters
    ----------
    c : float, optional
        The tuning constant for Ramsay's Ea function.  The default value is
        2.0.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, c=2.0):
        self.c = c

    def _subset(self, z):
        """
        Least trimmed mean is defined piecewise over the range of z.
        """
        pass

    def rho(self, z):
        """
        The robust criterion function for least trimmed mean.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = (1/2.)*z**2    for \\|z\\| <= c

            rho(z) = (1/2.)*c**2              for \\|z\\| > c
        """
        pass

    def psi(self, z):
        """
        The psi function for least trimmed mean

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z              for \\|z\\| <= c

            psi(z) = 0              for \\|z\\| > c
        """
        pass

    def weights(self, z):
        """
        Least trimmed mean weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = 1             for \\|z\\| <= c

            weights(z) = 0             for \\|z\\| > c
        """
        pass

    def psi_deriv(self, z):
        """
        The derivative of least trimmed mean psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        pass

class Hampel(RobustNorm):
    """

    Hampel function for M-estimation.

    Parameters
    ----------
    a : float, optional
    b : float, optional
    c : float, optional
        The tuning constants for Hampel's function.  The default values are
        a,b,c = 2, 4, 8.

    See Also
    --------
    statsmodels.robust.norms.RobustNorm
    """

    def __init__(self, a=2.0, b=4.0, c=8.0):
        self.a = a
        self.b = b
        self.c = c

    def _subset(self, z):
        """
        Hampel's function is defined piecewise over the range of z
        """
        pass

    def rho(self, z):
        """
        The robust criterion function for Hampel's estimator

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = z**2 / 2                     for \\|z\\| <= a

            rho(z) = a*\\|z\\| - 1/2.*a**2               for a < \\|z\\| <= b

            rho(z) = a*(c - \\|z\\|)**2 / (c - b) / 2    for b < \\|z\\| <= c

            rho(z) = a*(b + c - a) / 2                 for \\|z\\| > c
        """
        pass

    def psi(self, z):
        """
        The psi function for Hampel's estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z                            for \\|z\\| <= a

            psi(z) = a*sign(z)                    for a < \\|z\\| <= b

            psi(z) = a*sign(z)*(c - \\|z\\|)/(c-b)    for b < \\|z\\| <= c

            psi(z) = 0                            for \\|z\\| > c
        """
        pass

    def weights(self, z):
        """
        Hampel weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            weights(z) = 1                                for \\|z\\| <= a

            weights(z) = a/\\|z\\|                          for a < \\|z\\| <= b

            weights(z) = a*(c - \\|z\\|)/(\\|z\\|*(c-b))      for b < \\|z\\| <= c

            weights(z) = 0                                for \\|z\\| > c
        """
        pass

    def psi_deriv(self, z):
        """Derivative of psi function, second derivative of rho function.
        """
        pass

class TukeyBiweight(RobustNorm):
    """

    Tukey's biweight function for M-estimation.

    Parameters
    ----------
    c : float, optional
        The tuning constant for Tukey's Biweight.  The default value is
        c = 4.685.

    Notes
    -----
    Tukey's biweight is sometime's called bisquare.
    """

    def __init__(self, c=4.685):
        self.c = c

    def _subset(self, z):
        """
        Tukey's biweight is defined piecewise over the range of z
        """
        pass

    def rho(self, z):
        """
        The robust criterion function for Tukey's biweight estimator

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
            rho(z) = -(1 - (z/c)**2)**3 * c**2/6.   for \\|z\\| <= R

            rho(z) = 0                              for \\|z\\| > R
        """
        pass

    def psi(self, z):
        """
        The psi function for Tukey's biweight estimator

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
            psi(z) = z*(1 - (z/c)**2)**2        for \\|z\\| <= R

            psi(z) = 0                           for \\|z\\| > R
        """
        pass

    def weights(self, z):
        """
        Tukey's biweight weighting function for the IRLS algorithm

        The psi function scaled by z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
            psi(z) = (1 - (z/c)**2)**2          for \\|z\\| <= R

            psi(z) = 0                          for \\|z\\| > R
        """
        pass

    def psi_deriv(self, z):
        """
        The derivative of Tukey's biweight psi function

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        pass

class MQuantileNorm(RobustNorm):
    """M-quantiles objective function based on a base norm

    This norm has the same asymmetric structure as the objective function
    in QuantileRegression but replaces the L1 absolute value by a chosen
    base norm.

        rho_q(u) = abs(q - I(q < 0)) * rho_base(u)

    or, equivalently,

        rho_q(u) = q * rho_base(u)  if u >= 0
        rho_q(u) = (1 - q) * rho_base(u)  if u < 0


    Parameters
    ----------
    q : float
        M-quantile, must be between 0 and 1
    base_norm : RobustNorm instance
        basic norm that is transformed into an asymmetric M-quantile norm

    Notes
    -----
    This is mainly for base norms that are not redescending, like HuberT or
    LeastSquares. (See Jones for the relationship of M-quantiles to quantiles
    in the case of non-redescending Norms.)

    Expectiles are M-quantiles with the LeastSquares as base norm.

    References
    ----------

    .. [*] Bianchi, Annamaria, and Nicola Salvati. 2015. “Asymptotic Properties
       and Variance Estimators of the M-Quantile Regression Coefficients
       Estimators.” Communications in Statistics - Theory and Methods 44 (11):
       2416–29. doi:10.1080/03610926.2013.791375.

    .. [*] Breckling, Jens, and Ray Chambers. 1988. “M-Quantiles.”
       Biometrika 75 (4): 761–71. doi:10.2307/2336317.

    .. [*] Jones, M. C. 1994. “Expectiles and M-Quantiles Are Quantiles.”
       Statistics & Probability Letters 20 (2): 149–53.
       doi:10.1016/0167-7152(94)90031-0.

    .. [*] Newey, Whitney K., and James L. Powell. 1987. “Asymmetric Least
       Squares Estimation and Testing.” Econometrica 55 (4): 819–47.
       doi:10.2307/1911031.
    """

    def __init__(self, q, base_norm):
        self.q = q
        self.base_norm = base_norm

    def rho(self, z):
        """
        The robust criterion function for MQuantileNorm.

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        rho : ndarray
        """
        pass

    def psi(self, z):
        """
        The psi function for MQuantileNorm estimator.

        The analytic derivative of rho

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi : ndarray
        """
        pass

    def weights(self, z):
        """
        MQuantileNorm weighting function for the IRLS algorithm

        The psi function scaled by z, psi(z) / z

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        weights : ndarray
        """
        pass

    def psi_deriv(self, z):
        """
        The derivative of MQuantileNorm function

        Parameters
        ----------
        z : array_like
            1d array

        Returns
        -------
        psi_deriv : ndarray

        Notes
        -----
        Used to estimate the robust covariance matrix.
        """
        pass

    def __call__(self, z):
        """
        Returns the value of estimator rho applied to an input
        """
        return self.rho(z)

def estimate_location(a, scale, norm=None, axis=0, initial=None, maxiter=30, tol=1e-06):
    """
    M-estimator of location using self.norm and a current
    estimator of scale.

    This iteratively finds a solution to

    norm.psi((a-mu)/scale).sum() == 0

    Parameters
    ----------
    a : ndarray
        Array over which the location parameter is to be estimated
    scale : ndarray
        Scale parameter to be used in M-estimator
    norm : RobustNorm, optional
        Robust norm used in the M-estimator.  The default is HuberT().
    axis : int, optional
        Axis along which to estimate the location parameter.  The default is 0.
    initial : ndarray, optional
        Initial condition for the location parameter.  Default is None, which
        uses the median of a.
    niter : int, optional
        Maximum number of iterations.  The default is 30.
    tol : float, optional
        Toleration for convergence.  The default is 1e-06.

    Returns
    -------
    mu : ndarray
        Estimate of location
    """
    pass
