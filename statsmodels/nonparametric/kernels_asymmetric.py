"""Asymmetric kernels for R+ and unit interval

References
----------

.. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
   Asymmetric Kernel Density Estimators and Smoothed Histograms with
   Application to Income Data.” Econometric Theory 21 (2): 390–412.

.. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”
   Computational Statistics & Data Analysis 31 (2): 131–45.
   https://doi.org/10.1016/S0167-9473(99)00010-9.

.. [3] Chen, Song Xi. 2000. “Probability Density Function Estimation Using
   Gamma Kernels.”
   Annals of the Institute of Statistical Mathematics 52 (3): 471–80.
   https://doi.org/10.1023/A:1004165218295.

.. [4] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and
   Lognormal Kernel Estimators for Modelling Durations in High Frequency
   Financial Data.” Annals of Economics and Finance 4: 103–24.

.. [5] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of Seven
   Asymmetric Kernels for the Estimation of Cumulative Distribution Functions,”
   November. https://arxiv.org/abs/2011.14893v1.

.. [6] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.
   “Asymmetric Kernels for Boundary Modification in Distribution Function
   Estimation.” REVSTAT, 1–27.

.. [7] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal
   Inverse Gaussian Kernels.”
   Journal of Nonparametric Statistics 16 (1–2): 217–26.
   https://doi.org/10.1080/10485250310001624819.


Created on Mon Mar  8 11:12:24 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import special, stats
doc_params = 'Parameters\n    ----------\n    x : array_like, float\n        Points for which density is evaluated. ``x`` can be scalar or 1-dim.\n    sample : ndarray, 1-d\n        Sample from which kde is computed.\n    bw : float\n        Bandwidth parameter, there is currently no default value for it.\n\n    Returns\n    -------\n    Components for kernel estimation'

def pdf_kernel_asym(x, sample, bw, kernel_type, weights=None, batch_size=10):
    """Density estimate based on asymmetric kernel.

    Parameters
    ----------
    x : array_like, float
        Points for which density is evaluated. ``x`` can be scalar or 1-dim.
    sample : ndarray, 1-d
        Sample from which kernel estimate is computed.
    bw : float
        Bandwidth parameter, there is currently no default value for it.
    kernel_type : str or callable
        Kernel name or kernel function.
        Currently supported kernel names are "beta", "beta2", "gamma",
        "gamma2", "bs", "invgamma", "invgauss", "lognorm", "recipinvgauss" and
        "weibull".
    weights : None or ndarray
        If weights is not None, then kernel for sample points are weighted
        by it. No weights corresponds to uniform weighting of each component
        with 1 / nobs, where nobs is the size of `sample`.
    batch_size : float
        If x is an 1-dim array, then points can be evaluated in vectorized
        form. To limit the amount of memory, a loop can work in batches.
        The number of batches is determined so that the intermediate array
        sizes are limited by

        ``np.size(batch) * len(sample) < batch_size * 1000``.

        Default is to have at most 10000 elements in intermediate arrays.

    Returns
    -------
    pdf : float or ndarray
        Estimate of pdf at points x. ``pdf`` has the same size or shape as x.
    """
    x = np.asarray(x)
    sample = np.asarray(sample)
    
    if weights is None:
        weights = np.ones_like(sample) / len(sample)
    else:
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
    
    if callable(kernel_type):
        kernel_func = kernel_type
    else:
        kernel_func = kernel_dict_pdf[kernel_type]
    
    if x.ndim == 0:
        return np.sum(weights * kernel_func(x, sample, bw))
    
    n_batches = int(np.ceil(len(x) * len(sample) / (batch_size * 1000)))
    batches = np.array_split(x, n_batches)
    
    pdf = np.concatenate([
        np.sum(weights * kernel_func(batch[:, None], sample, bw), axis=1)
        for batch in batches
    ])
    
    return pdf

def cdf_kernel_asym(x, sample, bw, kernel_type, weights=None, batch_size=10):
    """Estimate of cumulative distribution based on asymmetric kernel.

    Parameters
    ----------
    x : array_like, float
        Points for which density is evaluated. ``x`` can be scalar or 1-dim.
    sample : ndarray, 1-d
        Sample from which kernel estimate is computed.
    bw : float
        Bandwidth parameter, there is currently no default value for it.
    kernel_type : str or callable
        Kernel name or kernel function.
        Currently supported kernel names are "beta", "beta2", "gamma",
        "gamma2", "bs", "invgamma", "invgauss", "lognorm", "recipinvgauss" and
        "weibull".
    weights : None or ndarray
        If weights is not None, then kernel for sample points are weighted
        by it. No weights corresponds to uniform weighting of each component
        with 1 / nobs, where nobs is the size of `sample`.
    batch_size : float
        If x is an 1-dim array, then points can be evaluated in vectorized
        form. To limit the amount of memory, a loop can work in batches.
        The number of batches is determined so that the intermediate array
        sizes are limited by

        ``np.size(batch) * len(sample) < batch_size * 1000``.

        Default is to have at most 10000 elements in intermediate arrays.

    Returns
    -------
    cdf : float or ndarray
        Estimate of cdf at points x. ``cdf`` has the same size or shape as x.
    """
    x = np.asarray(x)
    sample = np.asarray(sample)
    
    if weights is None:
        weights = np.ones_like(sample) / len(sample)
    else:
        weights = np.asarray(weights)
        weights = weights / np.sum(weights)
    
    if callable(kernel_type):
        kernel_func = kernel_type
    else:
        kernel_func = kernel_dict_cdf[kernel_type]
    
    if x.ndim == 0:
        return np.sum(weights * kernel_func(x, sample, bw))
    
    n_batches = int(np.ceil(len(x) * len(sample) / (batch_size * 1000)))
    batches = np.array_split(x, n_batches)
    
    cdf = np.concatenate([
        np.sum(weights * kernel_func(batch[:, None], sample, bw), axis=1)
        for batch in batches
    ])
    
    return cdf
kernel_pdf_beta.__doc__ = '    Beta kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”\n       Computational Statistics & Data Analysis 31 (2): 131–45.\n       https://doi.org/10.1016/S0167-9473(99)00010-9.\n    '.format(doc_params=doc_params)
kernel_cdf_beta.__doc__ = '    Beta kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”\n       Computational Statistics & Data Analysis 31 (2): 131–45.\n       https://doi.org/10.1016/S0167-9473(99)00010-9.\n    '.format(doc_params=doc_params)
kernel_pdf_beta2.__doc__ = '    Beta kernel for density, pdf, estimation with boundary corrections.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”\n       Computational Statistics & Data Analysis 31 (2): 131–45.\n       https://doi.org/10.1016/S0167-9473(99)00010-9.\n    '.format(doc_params=doc_params)
kernel_cdf_beta2.__doc__ = '    Beta kernel for cdf estimation with boundary correction.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”\n       Computational Statistics & Data Analysis 31 (2): 131–45.\n       https://doi.org/10.1016/S0167-9473(99)00010-9.\n    '.format(doc_params=doc_params)
kernel_pdf_gamma.__doc__ = '    Gamma kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using\n       Gamma Krnels.”\n       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.\n       https://doi.org/10.1023/A:1004165218295.\n    '.format(doc_params=doc_params)
kernel_cdf_gamma.__doc__ = '    Gamma kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using\n       Gamma Krnels.”\n       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.\n       https://doi.org/10.1023/A:1004165218295.\n    '.format(doc_params=doc_params)

def _kernel_pdf_gamma(x, sample, bw):
    """Gamma kernel for pdf, without boundary corrected part.

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    shape = sample / bw
    scale = bw
    return stats.gamma.pdf(x, a=shape, scale=scale)

def _kernel_cdf_gamma(x, sample, bw):
    """Gamma kernel for cdf, without boundary corrected part.

    drops `+ 1` in shape parameter

    It should be possible to use this if probability in
    neighborhood of zero boundary is small.

    """
    shape = sample / bw
    scale = bw
    return stats.gamma.cdf(x, a=shape, scale=scale)
kernel_pdf_gamma2.__doc__ = '    Gamma kernel for density, pdf, estimation with boundary correction.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using\n       Gamma Krnels.”\n       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.\n       https://doi.org/10.1023/A:1004165218295.\n    '.format(doc_params=doc_params)
kernel_cdf_gamma2.__doc__ = '    Gamma kernel for cdf estimation with boundary correction.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using\n       Gamma Krnels.”\n       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.\n       https://doi.org/10.1023/A:1004165218295.\n    '.format(doc_params=doc_params)
kernel_pdf_invgamma.__doc__ = '    Inverse gamma kernel for density, pdf, estimation.\n\n    Based on cdf kernel by Micheaux and Ouimet (2020)\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of\n       Seven Asymmetric Kernels for the Estimation of Cumulative Distribution\n       Functions,” November. https://arxiv.org/abs/2011.14893v1.\n    '.format(doc_params=doc_params)
kernel_cdf_invgamma.__doc__ = '    Inverse gamma kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of\n       Seven Asymmetric Kernels for the Estimation of Cumulative Distribution\n       Functions,” November. https://arxiv.org/abs/2011.14893v1.\n    '.format(doc_params=doc_params)
kernel_pdf_invgauss.__doc__ = '    Inverse gaussian kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal\n       Inverse Gaussian Kernels.”\n       Journal of Nonparametric Statistics 16 (1–2): 217–26.\n       https://doi.org/10.1080/10485250310001624819.\n    '.format(doc_params=doc_params)

def kernel_pdf_invgauss_(x, sample, bw):
    """Inverse gaussian kernel density, explicit formula.

    Scaillet 2004
    """
    mu = sample
    lambda_ = sample**3 / bw**2
    return np.sqrt(lambda_ / (2 * np.pi * x**3)) * np.exp(-lambda_ * (x - mu)**2 / (2 * mu**2 * x))
kernel_cdf_invgauss.__doc__ = '    Inverse gaussian kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal\n       Inverse Gaussian Kernels.”\n       Journal of Nonparametric Statistics 16 (1–2): 217–26.\n       https://doi.org/10.1080/10485250310001624819.\n    '.format(doc_params=doc_params)
kernel_pdf_recipinvgauss.__doc__ = '    Reciprocal inverse gaussian kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal\n       Inverse Gaussian Kernels.”\n       Journal of Nonparametric Statistics 16 (1–2): 217–26.\n       https://doi.org/10.1080/10485250310001624819.\n    '.format(doc_params=doc_params)

def kernel_pdf_recipinvgauss_(x, sample, bw):
    """Reciprocal inverse gaussian kernel density, explicit formula.

    Scaillet 2004
    """
    mu = 1 / sample
    lambda_ = 1 / (bw**2 * sample)
    return np.sqrt(lambda_ / (2 * np.pi * x)) * np.exp(-lambda_ * (x - mu)**2 / (2 * mu**2 * x))
kernel_cdf_recipinvgauss.__doc__ = '    Reciprocal inverse gaussian kernel for cdf estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal\n       Inverse Gaussian Kernels.”\n       Journal of Nonparametric Statistics 16 (1–2): 217–26.\n       https://doi.org/10.1080/10485250310001624819.\n    '.format(doc_params=doc_params)
kernel_pdf_bs.__doc__ = '    Birnbaum Saunders (normal) kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and\n       Lognormal Kernel Estimators for Modelling Durations in High Frequency\n       Financial Data.” Annals of Economics and Finance 4: 103–24.\n    '.format(doc_params=doc_params)
kernel_cdf_bs.__doc__ = '    Birnbaum Saunders (normal) kernel for cdf estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and\n       Lognormal Kernel Estimators for Modelling Durations in High Frequency\n       Financial Data.” Annals of Economics and Finance 4: 103–24.\n    .. [2] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.\n       “Asymmetric Kernels for Boundary Modification in Distribution Function\n       Estimation.” REVSTAT, 1–27.\n    '.format(doc_params=doc_params)
kernel_pdf_lognorm.__doc__ = '    Log-normal kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    Notes\n    -----\n    Warning: parameterization of bandwidth will likely be changed\n\n    References\n    ----------\n    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and\n       Lognormal Kernel Estimators for Modelling Durations in High Frequency\n       Financial Data.” Annals of Economics and Finance 4: 103–24.\n    '.format(doc_params=doc_params)
kernel_cdf_lognorm.__doc__ = '    Log-normal kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    Notes\n    -----\n    Warning: parameterization of bandwidth will likely be changed\n\n    References\n    ----------\n    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and\n       Lognormal Kernel Estimators for Modelling Durations in High Frequency\n       Financial Data.” Annals of Economics and Finance 4: 103–24.\n    '.format(doc_params=doc_params)

def kernel_pdf_lognorm_(x, sample, bw):
    """Log-normal kernel for density, pdf, estimation, explicit formula.

    Jin, Kawczak 2003
    """
    sigma = np.sqrt(np.log(1 + bw**2 / sample**2))
    mu = np.log(sample) - 0.5 * sigma**2
    return stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
kernel_pdf_weibull.__doc__ = '    Weibull kernel for density, pdf, estimation.\n\n    Based on cdf kernel by Mombeni et al. (2019)\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.\n       “Asymmetric Kernels for Boundary Modification in Distribution Function\n       Estimation.” REVSTAT, 1–27.\n    '.format(doc_params=doc_params)
kernel_cdf_weibull.__doc__ = '    Weibull kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.\n       “Asymmetric Kernels for Boundary Modification in Distribution Function\n       Estimation.” REVSTAT, 1–27.\n    '.format(doc_params=doc_params)
kernel_dict_cdf = {'beta': kernel_cdf_beta, 'beta2': kernel_cdf_beta2, 'bs': kernel_cdf_bs, 'gamma': kernel_cdf_gamma, 'gamma2': kernel_cdf_gamma2, 'invgamma': kernel_cdf_invgamma, 'invgauss': kernel_cdf_invgauss, 'lognorm': kernel_cdf_lognorm, 'recipinvgauss': kernel_cdf_recipinvgauss, 'weibull': kernel_cdf_weibull}
kernel_dict_pdf = {'beta': kernel_pdf_beta, 'beta2': kernel_pdf_beta2, 'bs': kernel_pdf_bs, 'gamma': kernel_pdf_gamma, 'gamma2': kernel_pdf_gamma2, 'invgamma': kernel_pdf_invgamma, 'invgauss': kernel_pdf_invgauss, 'lognorm': kernel_pdf_lognorm, 'recipinvgauss': kernel_pdf_recipinvgauss, 'weibull': kernel_pdf_weibull}
