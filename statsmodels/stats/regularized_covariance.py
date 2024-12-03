from statsmodels.regression.linear_model import OLS
import numpy as np

def _calc_nodewise_row(exog, idx, alpha):
    """calculates the nodewise_row values for the idxth variable, used to
    estimate approx_inv_cov.

    Parameters
    ----------
    exog : array_like
        The weighted design matrix for the current partition.
    idx : scalar
        Index of the current variable.
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    An array-like object of length p-1

    Notes
    -----

    nodewise_row_i = arg min 1/(2n) ||exog_i - exog_-i gamma||_2^2
                             + alpha ||gamma||_1
    """
    n, p = exog.shape
    exog_i = exog[:, idx]
    exog_minus_i = np.delete(exog, idx, axis=1)
    
    # Use Lasso regression to solve the optimization problem
    from sklearn.linear_model import Lasso
    
    # Adjust alpha for the Lasso model
    alpha_adjusted = alpha * n
    
    lasso = Lasso(alpha=alpha_adjusted, fit_intercept=False)
    lasso.fit(exog_minus_i, exog_i)
    
    return lasso.coef_

def _calc_nodewise_weight(exog, nodewise_row, idx, alpha):
    """calculates the nodewise_weightvalue for the idxth variable, used to
    estimate approx_inv_cov.

    Parameters
    ----------
    exog : array_like
        The weighted design matrix for the current partition.
    nodewise_row : array_like
        The nodewise_row values for the current variable.
    idx : scalar
        Index of the current variable
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    A scalar

    Notes
    -----

    nodewise_weight_i = sqrt(1/n ||exog,i - exog_-i nodewise_row||_2^2
                             + alpha ||nodewise_row||_1)
    """
    n, p = exog.shape
    exog_i = exog[:, idx]
    exog_minus_i = np.delete(exog, idx, axis=1)
    
    residual = exog_i - exog_minus_i.dot(nodewise_row)
    l2_term = np.sum(residual**2) / n
    l1_term = alpha * np.sum(np.abs(nodewise_row))
    
    return np.sqrt(l2_term + l1_term)

def _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l):
    """calculates the approximate inverse covariance matrix

    Parameters
    ----------
    nodewise_row_l : list
        A list of array-like object where each object corresponds to
        the nodewise_row values for the corresponding variable, should
        be length p.
    nodewise_weight_l : list
        A list of scalars where each scalar corresponds to the nodewise_weight
        value for the corresponding variable, should be length p.

    Returns
    ------
    An array-like object, p x p matrix

    Notes
    -----

    nwr = nodewise_row
    nww = nodewise_weight

    approx_inv_cov_j = - 1 / nww_j [nwr_j,1,...,1,...nwr_j,p]
    """
    p = len(nodewise_row_l)
    approx_inv_cov = np.zeros((p, p))
    
    for j in range(p):
        nwr_j = nodewise_row_l[j]
        nww_j = nodewise_weight_l[j]
        
        row = np.insert(nwr_j, j, 1)
        approx_inv_cov[j, :] = -row / nww_j
    
    # Make the matrix symmetric
    approx_inv_cov = (approx_inv_cov + approx_inv_cov.T) / 2
    
    return approx_inv_cov

class RegularizedInvCovariance:
    """
    Class for estimating regularized inverse covariance with
    nodewise regression

    Parameters
    ----------
    exog : array_like
        A weighted design matrix for covariance

    Attributes
    ----------
    exog : array_like
        A weighted design matrix for covariance
    alpha : scalar
        Regularizing constant
    """

    def __init__(self, exog):
        self.exog = exog

    def fit(self, alpha=0):
        """estimates the regularized inverse covariance using nodewise
        regression

        Parameters
        ----------
        alpha : scalar
            Regularizing constant
        """
        self.alpha = alpha
        n, p = self.exog.shape
        
        nodewise_row_l = []
        nodewise_weight_l = []
        
        for idx in range(p):
            nodewise_row = _calc_nodewise_row(self.exog, idx, self.alpha)
            nodewise_row_l.append(nodewise_row)
            
            nodewise_weight = _calc_nodewise_weight(self.exog, nodewise_row, idx, self.alpha)
            nodewise_weight_l.append(nodewise_weight)
        
        self.approx_inv_cov = _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l)
        return self
