"""
Created on Wed May 04 06:09:18 2011

@author: josef
"""
import numpy as np

def mean_residual_life(x, frac=None, alpha=0.05):
    """empirical mean residual life or expected shortfall

    Parameters
    ----------
    x : 1-dimensional array_like
    frac : list[float], optional
        All entries must be between 0 and 1
    alpha : float, default 0.05
        FIXME: not actually used.

    TODO:
        check formula for std of mean
        does not include case for all observations
        last observations std is zero
        vectorize loop using cumsum
        frac does not work yet
    """
    x = np.sort(x)[::-1]  # Sort in descending order
    n = len(x)
    
    if frac is None:
        thresholds = x
    else:
        thresholds = [np.percentile(x, 100 * (1 - f)) for f in frac]
    
    results = []
    for threshold in thresholds:
        exceedances = x[x > threshold]
        if len(exceedances) == 0:
            results.append((threshold, np.nan, np.nan, 0))
        else:
            mean = np.mean(exceedances - threshold)
            std = np.std(exceedances - threshold, ddof=1)
            count = len(exceedances)
            results.append((threshold, mean, std, count))
    
    return np.array(results)
expected_shortfall = mean_residual_life
if __name__ == '__main__':
    rvs = np.random.standard_t(5, size=10)
    res = mean_residual_life(rvs)
    print(res)
    rmean = [rvs[i:].mean() for i in range(len(rvs))]
    print(res[:, 2] - rmean[1:])
    res_frac = mean_residual_life(rvs, frac=[0.5])
    print(res_frac)
