import numpy as np

def _make_index(prob, size):
    """
    Returns a boolean index for given probabilities.

    Notes
    -----
    prob = [.75,.25] means that there is a 75% chance of the first column
    being True and a 25% chance of the second column being True. The
    columns are mutually exclusive.
    """
    prob = np.array(prob)
    prob = prob / prob.sum()  # Normalize probabilities
    cum_prob = np.cumsum(prob)
    random_values = np.random.rand(size)
    index = (random_values[:, np.newaxis] < cum_prob).T
    return index

def mixture_rvs(prob, size, dist, kwargs=None):
    """
    Sample from a mixture of distributions.

    Parameters
    ----------
    prob : array_like
        Probability of sampling from each distribution in dist
    size : int
        The length of the returned sample.
    dist : array_like
        An iterable of distributions objects from scipy.stats.
    kwargs : tuple of dicts, optional
        A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and
        args to be passed to the respective distribution in dist.  If not
        provided, the distribution defaults are used.

    Examples
    --------
    Say we want 5000 random variables from mixture of normals with two
    distributions norm(-1,.5) and norm(1,.5) and we want to sample from the
    first with probability .75 and the second with probability .25.

    >>> from scipy import stats
    >>> prob = [.75,.25]
    >>> Y = mixture_rvs(prob, 5000, dist=[stats.norm, stats.norm],
    ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
    """
    index = _make_index(prob, size)
    if kwargs is None:
        kwargs = [{} for _ in dist]
    rvs = np.zeros(size)
    for i, (d, kw) in enumerate(zip(dist, kwargs)):
        rvs[index[i]] = d.rvs(size=index[i].sum(), **kw)
    return rvs

class MixtureDistribution:
    """univariate mixture distribution

    for simple case for now (unbound support)
    does not yet inherit from scipy.stats.distributions

    adding pdf to mixture_rvs, some restrictions on broadcasting
    Currently it does not hold any state, all arguments included in each method.
    """

    def pdf(self, x, prob, dist, kwargs=None):
        """
        pdf a mixture of distributions.

        Parameters
        ----------
        x : array_like
            Array containing locations where the PDF should be evaluated
        prob : array_like
            Probability of sampling from each distribution in dist
        dist : array_like
            An iterable of distributions objects from scipy.stats.
        kwargs : tuple of dicts, optional
            A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and
            args to be passed to the respective distribution in dist.  If not
            provided, the distribution defaults are used.

        Examples
        --------
        Say we want 5000 random variables from mixture of normals with two
        distributions norm(-1,.5) and norm(1,.5) and we want to sample from the
        first with probability .75 and the second with probability .25.

        >>> import numpy as np
        >>> from scipy import stats
        >>> from statsmodels.distributions.mixture_rvs import MixtureDistribution
        >>> x = np.arange(-4.0, 4.0, 0.01)
        >>> prob = [.75,.25]
        >>> mixture = MixtureDistribution()
        >>> Y = mixture.pdf(x, prob, dist=[stats.norm, stats.norm],
        ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
        """
        if kwargs is None:
            kwargs = [{} for _ in dist]
        prob = np.array(prob)
        prob = prob / prob.sum()  # Normalize probabilities
        pdf_values = np.zeros_like(x)
        for p, d, kw in zip(prob, dist, kwargs):
            pdf_values += p * d.pdf(x, **kw)
        return pdf_values

    def cdf(self, x, prob, dist, kwargs=None):
        """
        cdf of a mixture of distributions.

        Parameters
        ----------
        x : array_like
            Array containing locations where the CDF should be evaluated
        prob : array_like
            Probability of sampling from each distribution in dist
        size : int
            The length of the returned sample.
        dist : array_like
            An iterable of distributions objects from scipy.stats.
        kwargs : tuple of dicts, optional
            A tuple of dicts.  Each dict in kwargs can have keys loc, scale, and
            args to be passed to the respective distribution in dist.  If not
            provided, the distribution defaults are used.

        Examples
        --------
        Say we want 5000 random variables from mixture of normals with two
        distributions norm(-1,.5) and norm(1,.5) and we want to sample from the
        first with probability .75 and the second with probability .25.

        >>> import numpy as np
        >>> from scipy import stats
        >>> from statsmodels.distributions.mixture_rvs import MixtureDistribution
        >>> x = np.arange(-4.0, 4.0, 0.01)
        >>> prob = [.75,.25]
        >>> mixture = MixtureDistribution()
        >>> Y = mixture.pdf(x, prob, dist=[stats.norm, stats.norm],
        ...                 kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))
        """
        if kwargs is None:
            kwargs = [{} for _ in dist]
        prob = np.array(prob)
        prob = prob / prob.sum()  # Normalize probabilities
        cdf_values = np.zeros_like(x)
        for p, d, kw in zip(prob, dist, kwargs):
            cdf_values += p * d.cdf(x, **kw)
        return cdf_values

def mv_mixture_rvs(prob, size, dist, nvars, **kwargs):
    """
    Sample from a mixture of multivariate distributions.

    Parameters
    ----------
    prob : array_like
        Probability of sampling from each distribution in dist
    size : int
        The length of the returned sample.
    dist : array_like
        An iterable of distributions instances with callable method rvs.
    nvars : int
        dimension of the multivariate distribution, could be inferred instead
    kwargs : tuple of dicts, optional
        ignored

    Examples
    --------
    Say we want 2000 random variables from mixture of normals with two
    multivariate normal distributions, and we want to sample from the
    first with probability .4 and the second with probability .6.

    import statsmodels.sandbox.distributions.mv_normal as mvd

    cov3 = np.array([[ 1.  ,  0.5 ,  0.75],
                       [ 0.5 ,  1.5 ,  0.6 ],
                       [ 0.75,  0.6 ,  2.  ]])

    mu = np.array([-1, 0.0, 2.0])
    mu2 = np.array([4, 2.0, 2.0])
    mvn3 = mvd.MVNormal(mu, cov3)
    mvn32 = mvd.MVNormal(mu2, cov3/2., 4)
    rvs = mix.mv_mixture_rvs([0.4, 0.6], 2000, [mvn3, mvn32], 3)
    """
    index = _make_index(prob, size)
    rvs = np.zeros((size, nvars))
    for i, d in enumerate(dist):
        rvs[index[i]] = d.rvs(size=index[i].sum())
    return rvs
if __name__ == '__main__':
    from scipy import stats
    obs_dist = mixture_rvs([0.25, 0.75], size=10000, dist=[stats.norm, stats.beta], kwargs=(dict(loc=-1, scale=0.5), dict(loc=1, scale=1, args=(1, 0.5))))
    nobs = 10000
    mix = MixtureDistribution()
    mix_kwds = (dict(loc=-1, scale=0.25), dict(loc=1, scale=0.75))
    mrvs = mix.rvs([1 / 3.0, 2 / 3.0], size=nobs, dist=[stats.norm, stats.norm], kwargs=mix_kwds)
    grid = np.linspace(-4, 4, 100)
    mpdf = mix.pdf(grid, [1 / 3.0, 2 / 3.0], dist=[stats.norm, stats.norm], kwargs=mix_kwds)
    mcdf = mix.cdf(grid, [1 / 3.0, 2 / 3.0], dist=[stats.norm, stats.norm], kwargs=mix_kwds)
    doplot = 1
    if doplot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(mrvs, bins=50, normed=True, color='red')
        plt.title('histogram of sample and pdf')
        plt.plot(grid, mpdf, lw=2, color='black')
        plt.figure()
        plt.hist(mrvs, bins=50, normed=True, cumulative=True, color='red')
        plt.title('histogram of sample and pdf')
        plt.plot(grid, mcdf, lw=2, color='black')
        plt.show()
