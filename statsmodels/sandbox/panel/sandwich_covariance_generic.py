"""covariance with (nobs,nobs) loop and general kernel

This is a general implementation that is not efficient for any special cases.
kernel is currently only for one continuous variable and any number of
categorical groups.

No spatial example, continuous is interpreted as time

Created on Wed Nov 30 08:20:44 2011

Author: Josef Perktold
License: BSD-3

"""
import numpy as np

def kernel(d1, d2, r=None, weights=None):
    """general product kernel

    hardcoded split for the example:
        cat1 is continuous (time), other categories are discrete

    weights is e.g. Bartlett for cat1
    r is (0,1) indicator vector for boolean weights 1{d1_i == d2_i}

    returns boolean if no continuous weights are used
    """
    if r is None:
        r = np.ones(d1.shape[1], dtype=bool)
    
    # Continuous part (time)
    if weights is not None:
        continuous_kernel = weights(abs(d1[0] - d2[0]))
    else:
        continuous_kernel = 1.0
    
    # Discrete part
    discrete_kernel = np.all(d1[1:] == d2[1:])
    
    # Combine continuous and discrete parts
    return continuous_kernel * discrete_kernel * np.prod(r[1:])

def aggregate_cov(x, d, r=None, weights=None):
    """sum of outer procuct over groups and time selected by r

    This is for a generic reference implementation, it uses a nobs-nobs double
    loop.

    Parameters
    ----------
    x : ndarray, (nobs,) or (nobs, k_vars)
        data, for robust standard error calculation, this is array of x_i * u_i
    d : ndarray, (nobs, n_groups)
        integer group labels, each column contains group (or time) indices
    r : ndarray, (n_groups,)
        indicator for which groups to include. If r[i] is zero, then
        this group is ignored. If r[i] is not zero, then the cluster robust
        standard errors include this group.
    weights : ndarray
        weights if the first group dimension uses a HAC kernel

    Returns
    -------
    cov : ndarray (k_vars, k_vars) or scalar
        covariance matrix aggregates over group kernels
    count : int
        number of terms added in sum, mainly returned for cross-checking

    Notes
    -----
    This uses `kernel` to calculate the weighted distance between two
    observations.

    """
    nobs = x.shape[0]
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    k_vars = x.shape[1]
    
    cov = np.zeros((k_vars, k_vars))
    count = 0
    
    for i in range(nobs):
        for j in range(nobs):
            k = kernel(d[i], d[j], r, weights)
            if k != 0:
                cov += k * np.outer(x[i], x[j])
                count += 1
    
    return cov, count

def S_all_hac(x, d, nlags=1):
    """HAC independent of categorical group membership
    """
    nobs = x.shape[0]
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    k_vars = x.shape[1]
    
    cov = np.zeros((k_vars, k_vars))
    
    for lag in range(nlags + 1):
        weight = 1 - lag / (nlags + 1)  # Bartlett kernel
        for i in range(nobs - lag):
            cov += weight * np.outer(x[i], x[i+lag])
            if lag > 0:
                cov += weight * np.outer(x[i+lag], x[i])
    
    return cov

def S_within_hac(x, d, nlags=1, groupidx=1):
    """HAC for observations within a categorical group
    """
    nobs = x.shape[0]
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    k_vars = x.shape[1]
    
    cov = np.zeros((k_vars, k_vars))
    groups = np.unique(d[:, groupidx])
    
    for group in groups:
        group_mask = d[:, groupidx] == group
        x_group = x[group_mask]
        nobs_group = x_group.shape[0]
        
        for lag in range(min(nlags + 1, nobs_group)):
            weight = 1 - lag / (nlags + 1)  # Bartlett kernel
            for i in range(nobs_group - lag):
                cov += weight * np.outer(x_group[i], x_group[i+lag])
                if lag > 0:
                    cov += weight * np.outer(x_group[i+lag], x_group[i])
    
    return cov

def S_white(x, d):
    """simple white heteroscedasticity robust covariance
    note: calculating this way is very inefficient, just for cross-checking
    """
    nobs = x.shape[0]
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    k_vars = x.shape[1]
    
    cov = np.zeros((k_vars, k_vars))
    
    for i in range(nobs):
        cov += np.outer(x[i], x[i])
    
    return cov
