import warnings
import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import factorial
from scipy.stats import rv_continuous
import scipy.special as special
_faa_di_bruno_cache = {1: [[(1, 1)]], 2: [[(1, 2)], [(2, 1)]], 3: [[(1, 3)], [(2, 1), (1, 1)], [(3, 1)]], 4: [[(1, 4)], [(1, 2), (2, 1)], [(2, 2)], [(3, 1), (1, 1)], [(4, 1)]]}

def _faa_di_bruno_partitions(n):
    """
    Return all non-negative integer solutions of the diophantine equation

            n*k_n + ... + 2*k_2 + 1*k_1 = n   (1)

    Parameters
    ----------
    n : int
        the r.h.s. of Eq. (1)

    Returns
    -------
    partitions : list
        Each solution is itself a list of the form `[(m, k_m), ...]`
        for non-zero `k_m`. Notice that the index `m` is 1-based.

    Examples:
    ---------
    >>> _faa_di_bruno_partitions(2)
    [[(1, 2)], [(2, 1)]]
    >>> for p in _faa_di_bruno_partitions(4):
    ...     assert 4 == sum(m * k for (m, k) in p)
    """
    if n in _faa_di_bruno_cache:
        return _faa_di_bruno_cache[n]

    partitions = []
    
    def generate_partitions(remaining, max_part, current_partition):
        if remaining == 0:
            partitions.append(current_partition)
            return
        
        for i in range(1, min(max_part, remaining) + 1):
            new_partition = current_partition + [(i, 1)]
            generate_partitions(remaining - i, i, new_partition)
    
    generate_partitions(n, n, [])
    
    result = []
    for partition in partitions:
        counts = {}
        for part in partition:
            counts[part] = counts.get(part, 0) + 1
        result.append([(m, k) for (m, k) in counts.items()])
    
    _faa_di_bruno_cache[n] = result
    return result

def cumulant_from_moments(momt, n):
    """Compute n-th cumulant given moments.

    Parameters
    ----------
    momt : array_like
        `momt[j]` contains `(j+1)`-th moment.
        These can be raw moments around zero, or central moments
        (in which case, `momt[0]` == 0).
    n : int
        which cumulant to calculate (must be >1)

    Returns
    -------
    kappa : float
        n-th cumulant.
    """
    if n <= 1:
        raise ValueError("n must be greater than 1")
    
    momt = np.asarray(momt)
    partitions = _faa_di_bruno_partitions(n)
    
    kappa = momt[n-1]
    for partition in partitions[1:]:  # Skip the first partition [(n, 1)]
        term = (-1)**(len(partition) - 1) * factorial(len(partition) - 1)
        for m, k in partition:
            term *= (momt[m-1] / factorial(m)) ** k / factorial(k)
        kappa -= term
    
    return kappa
_norm_pdf_C = np.sqrt(2 * np.pi)

class ExpandedNormal(rv_continuous):
    """Construct the Edgeworth expansion pdf given cumulants.

    Parameters
    ----------
    cum : array_like
        `cum[j]` contains `(j+1)`-th cumulant: cum[0] is the mean,
        cum[1] is the variance and so on.

    Notes
    -----
    This is actually an asymptotic rather than convergent series, hence
    higher orders of the expansion may or may not improve the result.
    In a strongly non-Gaussian case, it is possible that the density
    becomes negative, especially far out in the tails.

    Examples
    --------
    Construct the 4th order expansion for the chi-square distribution using
    the known values of the cumulants:

    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> from scipy.special import factorial
    >>> df = 12
    >>> chi2_c = [2**(j-1) * factorial(j-1) * df for j in range(1, 5)]
    >>> edgw_chi2 = ExpandedNormal(chi2_c, name='edgw_chi2', momtype=0)

    Calculate several moments:
    >>> m, v = edgw_chi2.stats(moments='mv')
    >>> np.allclose([m, v], [df, 2 * df])
    True

    Plot the density function:
    >>> mu, sigma = df, np.sqrt(2*df)
    >>> x = np.linspace(mu - 3*sigma, mu + 3*sigma)
    >>> fig1 = plt.plot(x, stats.chi2.pdf(x, df=df), 'g-', lw=4, alpha=0.5)
    >>> fig2 = plt.plot(x, stats.norm.pdf(x, mu, sigma), 'b--', lw=4, alpha=0.5)
    >>> fig3 = plt.plot(x, edgw_chi2.pdf(x), 'r-', lw=2)
    >>> plt.show()

    References
    ----------
    .. [*] E.A. Cornish and R.A. Fisher, Moments and cumulants in the
         specification of distributions, Revue de l'Institut Internat.
         de Statistique. 5: 307 (1938), reprinted in
         R.A. Fisher, Contributions to Mathematical Statistics. Wiley, 1950.
    .. [*] https://en.wikipedia.org/wiki/Edgeworth_series
    .. [*] S. Blinnikov and R. Moessner, Expansions for nearly Gaussian
        distributions, Astron. Astrophys. Suppl. Ser. 130, 193 (1998)
    """

    def __init__(self, cum, name='Edgeworth expanded normal', **kwds):
        if len(cum) < 2:
            raise ValueError('At least two cumulants are needed.')
        self._coef, self._mu, self._sigma = self._compute_coefs_pdf(cum)
        self._herm_pdf = HermiteE(self._coef)
        if self._coef.size > 2:
            self._herm_cdf = HermiteE(-self._coef[1:])
        else:
            self._herm_cdf = lambda x: 0.0
        r = np.real_if_close(self._herm_pdf.roots())
        r = (r - self._mu) / self._sigma
        if r[(np.imag(r) == 0) & (np.abs(r) < 4)].any():
            mesg = 'PDF has zeros at %s ' % r
            warnings.warn(mesg, RuntimeWarning)
        kwds.update({'name': name, 'momtype': 0})
        super(ExpandedNormal, self).__init__(**kwds)
