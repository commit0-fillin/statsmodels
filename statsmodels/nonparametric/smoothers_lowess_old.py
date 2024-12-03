"""
Univariate lowess function, like in R.

References
----------
Hastie, Tibshirani, Friedman. (2009) The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition: Chapter 6.

Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots". Journal of the American Statistical Association 74 (368): 829-836.
"""
import numpy as np
from numpy.linalg import lstsq

def lowess(endog, exog, frac=2.0 / 3, it=3):
    """
    LOWESS (Locally Weighted Scatterplot Smoothing)

    A lowess function that outs smoothed estimates of endog
    at the given exog values from points (exog, endog)

    Parameters
    ----------
    endog : 1-D numpy array
        The y-values of the observed points
    exog : 1-D numpy array
        The x-values of the observed points
    frac : float
        Between 0 and 1. The fraction of the data used
        when estimating each y-value.
    it : int
        The number of residual-based reweightings
        to perform.

    Returns
    -------
    out: numpy array
        A numpy array with two columns. The first column
        is the sorted x values and the second column the
        associated estimated y-values.

    Notes
    -----
    This lowess function implements the algorithm given in the
    reference below using local linear estimates.

    Suppose the input data has N points. The algorithm works by
    estimating the true ``y_i`` by taking the frac*N closest points
    to ``(x_i,y_i)`` based on their x values and estimating ``y_i``
    using a weighted linear regression. The weight for ``(x_j,y_j)``
    is `_lowess_tricube` function applied to ``|x_i-x_j|``.

    If ``iter > 0``, then further weighted local linear regressions
    are performed, where the weights are the same as above
    times the `_lowess_bisquare` function of the residuals. Each iteration
    takes approximately the same amount of time as the original fit,
    so these iterations are expensive. They are most useful when
    the noise has extremely heavy tails, such as Cauchy noise.
    Noise with less heavy-tails, such as t-distributions with ``df > 2``,
    are less problematic. The weights downgrade the influence of
    points with large residuals. In the extreme case, points whose
    residuals are larger than 6 times the median absolute residual
    are given weight 0.

    Some experimentation is likely required to find a good
    choice of frac and iter for a particular dataset.

    References
    ----------
    Cleveland, W.S. (1979) "Robust Locally Weighted Regression
    and Smoothing Scatterplots". Journal of the American Statistical
    Association 74 (368): 829-836.

    Examples
    --------
    The below allows a comparison between how different the fits from
    `lowess` for different values of frac can be.

    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> lowess = sm.nonparametric.lowess
    >>> x = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=500)
    >>> y = np.sin(x) + np.random.normal(size=len(x))
    >>> z = lowess(y, x)
    >>> w = lowess(y, x, frac=1./3)

    This gives a similar comparison for when it is 0 vs not.

    >>> import scipy.stats as stats
    >>> x = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=500)
    >>> y = np.sin(x) + stats.cauchy.rvs(size=len(x))
    >>> z = lowess(y, x, frac= 1./3, it=0)
    >>> w = lowess(y, x, frac=1./3)
    """
    x, y = np.array(exog), np.array(endog)
    
    if exog.ndim != 1:
        raise ValueError("exog must be 1-dimensional")
    if endog.ndim != 1:
        raise ValueError("endog must be 1-dimensional")
    if len(exog) != len(endog):
        raise ValueError("exog and endog must have same length")
    
    n = len(exog)
    k = int(frac * n)
    k = max(min(n, k), 1)  # ensure 1 <= k <= n
    
    # Sort the data
    order = np.argsort(exog)
    x, y = x[order], y[order]
    
    fitted = _lowess_initial_fit(x, y, k, n)
    
    # Perform iterative reweighting
    for _ in range(it):
        fitted = _lowess_robustify_fit(x, y, fitted, k, n)
    
    return np.column_stack((x, fitted))

def _lowess_initial_fit(x_copy, y_copy, k, n):
    """
    The initial weighted local linear regression for lowess.

    Parameters
    ----------
    x_copy : 1-d ndarray
        The x-values/exogenous part of the data being smoothed
    y_copy : 1-d ndarray
        The y-values/ endogenous part of the data being smoothed
   k : int
        The number of data points which affect the linear fit for
        each estimated point
    n : int
        The total number of points

    Returns
    -------
    fitted : 1-d ndarray
        The fitted y-values
    weights : 2-d ndarray
        An n by k array. The contribution to the weights in the
        local linear fit coming from the distances between the
        x-values

   """
    fitted = np.zeros(n)
    weights = np.zeros((n, k))

    for i in range(n):
        left = max(0, i - k // 2)
        right = min(n, left + k)
        left = max(0, right - k)
        x_neighborhood = x_copy[left:right]
        y_neighborhood = y_copy[left:right]

        weights_i = np.abs(x_neighborhood - x_copy[i])
        max_weight = np.max(weights_i)
        if max_weight > 0:
            weights_i /= max_weight
        _lowess_tricube(weights_i)
        weights[i, :len(weights_i)] = weights_i

        X = np.column_stack((np.ones_like(x_neighborhood), x_neighborhood - x_copy[i]))
        W = np.diag(weights_i)
        beta = lstsq(np.dot(W, X), np.dot(W, y_neighborhood), rcond=None)[0]
        fitted[i] = beta[0]

    return fitted, weights

def _lowess_wt_standardize(weights, new_entries, x_copy_i, width):
    """
    The initial phase of creating the weights.
    Subtract the current x_i and divide by the width.

    Parameters
    ----------
    weights : ndarray
        The memory where (new_entries - x_copy_i)/width will be placed
    new_entries : ndarray
        The x-values of the k closest points to x[i]
    x_copy_i : float
        x[i], the i'th point in the (sorted) x values
    width : float
        The maximum distance between x[i] and any point in new_entries

    Returns
    -------
    Nothing. The modifications are made to weight in place.
    """
    weights[:] = (new_entries - x_copy_i) / width

def _lowess_robustify_fit(x_copy, y_copy, fitted, k, n):
    """
    Additional weighted local linear regressions, performed if
    iter>0. They take into account the sizes of the residuals,
    to eliminate the effect of extreme outliers.

    Parameters
    ----------
    x_copy : 1-d ndarray
        The x-values/exogenous part of the data being smoothed
    y_copy : 1-d ndarray
        The y-values/ endogenous part of the data being smoothed
    fitted : 1-d ndarray
        The fitted y-values from the previous iteration
    k : int
        The number of data points which affect the linear fit for
        each estimated point
    n : int
        The total number of points

   Returns
    -------
    new_fitted : 1-d ndarray
        The updated fitted y-values
    """
    residuals = y_copy - fitted
    s = np.median(np.abs(residuals))
    if s == 0:
        return fitted

    for i in range(n):
        left = max(0, i - k // 2)
        right = min(n, left + k)
        left = max(0, right - k)
        x_neighborhood = x_copy[left:right]
        y_neighborhood = y_copy[left:right]

        weights = np.abs(x_neighborhood - x_copy[i])
        max_weight = np.max(weights)
        if max_weight > 0:
            weights /= max_weight
        _lowess_tricube(weights)

        arg_sort = np.argsort(x_neighborhood)
        x_neighborhood = x_neighborhood[arg_sort]
        y_neighborhood = y_neighborhood[arg_sort]
        weights = weights[arg_sort]

        res = y_neighborhood - fitted[left:right]
        weights *= _lowess_bisquare(res / (6 * s))

        X = np.column_stack((np.ones_like(x_neighborhood), x_neighborhood - x_copy[i]))
        W = np.diag(weights)
        beta = lstsq(np.dot(W, X), np.dot(W, y_neighborhood), rcond=None)[0]
        fitted[i] = beta[0]

    return fitted

def _lowess_update_nn(x, cur_nn, i):
    """
    Update the endpoints of the nearest neighbors to
    the ith point.

    Parameters
    ----------
    x : iterable
        The sorted points of x-values
    cur_nn : list of length 2
        The two current indices between which are the
        k closest points to x[i]. (The actual value of
        k is irrelevant for the algorithm.
    i : int
        The index of the current value in x for which
        the k closest points are desired.

    Returns
    -------
    Nothing. It modifies cur_nn in place.
    """
    while cur_nn[0] > 0 and x[i] - x[cur_nn[0]-1] < x[cur_nn[1]] - x[i]:
        cur_nn[1] = cur_nn[0]
        cur_nn[0] -= 1
    while cur_nn[1] < len(x) - 1 and x[cur_nn[1]+1] - x[i] < x[i] - x[cur_nn[0]]:
        cur_nn[0] = cur_nn[1]
        cur_nn[1] += 1

def _lowess_tricube(t):
    """
    The _tricube function applied to a numpy array.
    The tricube function is (1-abs(t)**3)**3.

    Parameters
    ----------
    t : ndarray
        Array the tricube function is applied to elementwise and
        in-place.

    Returns
    -------
    Nothing
    """
    t = np.abs(t)
    t = np.where(t > 1, 0, (1 - t**3)**3)

def _lowess_mycube(t):
    """
    Fast matrix cube

    Parameters
    ----------
    t : ndarray
        Array that is cubed, elementwise and in-place

    Returns
    -------
    Nothing
    """
    t **= 3

def _lowess_bisquare(t):
    """
    The bisquare function applied to a numpy array.
    The bisquare function is (1-t**2)**2.

    Parameters
    ----------
    t : ndarray
        array bisquare function is applied to, element-wise and in-place.

    Returns
    -------
    result : ndarray
        The result of applying the bisquare function to t
    """
    t = np.abs(t)
    result = (1 - t**2)**2
    result[t > 1] = 0
    return result
