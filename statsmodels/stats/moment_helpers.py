"""helper functions conversion between moments

contains:

* conversion between central and non-central moments, skew, kurtosis and
  cummulants
* cov2corr : convert covariance matrix to correlation matrix


Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy.special import comb

def mc2mnc(mc):
    """convert central to non-central moments, uses recursive formula
    optionally adjusts first moment to return mean
    """
    mnc = [mc[0]]
    for k in range(1, len(mc)):
        mnc_k = sum(comb(k, i) * mc[i] * mc[0]**(k-i) for i in range(k+1))
        mnc.append(mnc_k)
    return np.array(mnc)

def mnc2mc(mnc, wmean=True):
    """convert non-central to central moments, uses recursive formula
    optionally adjusts first moment to return mean
    """
    mc = [mnc[0] if wmean else 0]
    for k in range(1, len(mnc)):
        mc_k = mnc[k] - sum(comb(k, i) * mc[i] * mnc[0]**(k-i) for i in range(1, k+1))
        mc.append(mc_k)
    return np.array(mc)

def cum2mc(kappa):
    """convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    References
    ----------
    Kenneth Lange: Numerical Analysis for Statisticians, page 40
    """
    mc = [kappa[0]]
    for n in range(1, len(kappa)):
        mc_n = kappa[n] + sum(comb(n-1, k-1) * kappa[k] * mc[n-k] for k in range(1, n))
        mc.append(mc_n)
    return np.array(mc)

def mnc2cum(mnc):
    """convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    https://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    """
    cum = [mnc[0]]
    for n in range(1, len(mnc)):
        cum_n = mnc[n] - sum(comb(n-1, k-1) * cum[k] * mnc[n-k] for k in range(1, n))
        cum.append(cum_n)
    return np.array(cum)

def mc2cum(mc):
    """
    just chained because I have still the test case
    """
    mnc = mc2mnc(mc)
    return mnc2cum(mnc)

def mvsk2mc(args):
    """convert mean, variance, skew, kurtosis to central moments"""
    mean, var, skew, kurt = args
    std = np.sqrt(var)
    mc = [mean, var, skew * std**3, kurt * var**2]
    return np.array(mc)

def mvsk2mnc(args):
    """convert mean, variance, skew, kurtosis to non-central moments"""
    mc = mvsk2mc(args)
    return mc2mnc(mc)

def mc2mvsk(args):
    """convert central moments to mean, variance, skew, kurtosis"""
    mean, var, mc3, mc4 = args
    std = np.sqrt(var)
    skew = mc3 / (std**3)
    kurt = mc4 / (var**2)
    return np.array([mean, var, skew, kurt])

def mnc2mvsk(args):
    """convert central moments to mean, variance, skew, kurtosis
    """
    mc = mnc2mc(args)
    return mc2mvsk(mc)

def cov2corr(cov, return_std=False):
    """
    convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires that
    division is defined elementwise. np.ma.array and np.matrix are allowed.
    """
    cov = np.asarray(cov)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    if return_std:
        return corr, std
    else:
        return corr

def corr2cov(corr, std):
    """
    convert correlation matrix to covariance matrix given standard deviation

    Parameters
    ----------
    corr : array_like, 2d
        correlation matrix, see Notes
    std : array_like, 1d
        standard deviation

    Returns
    -------
    cov : ndarray (subclass)
        covariance matrix

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that multiplication is defined elementwise. np.ma.array are allowed, but
    not matrices.
    """
    corr = np.asarray(corr)
    std = np.asarray(std)
    cov = corr * np.outer(std, std)
    return cov

def se_cov(cov):
    """
    get standard deviation from covariance matrix

    just a shorthand function np.sqrt(np.diag(cov))

    Parameters
    ----------
    cov : array_like, square
        covariance matrix

    Returns
    -------
    std : ndarray
        standard deviation from diagonal of cov
    """
    return np.sqrt(np.diag(cov))
