"""
A collection of smooth penalty functions.

Penalties on vectors take a vector argument and return a scalar
penalty.  The gradient of the penalty is a vector with the same shape
as the input value.

Penalties on covariance matrices take two arguments: the matrix and
its inverse, both in unpacked (square) form.  The returned penalty is
a scalar, and the gradient is returned as a vector that contains the
gradient with respect to the free elements in the lower triangle of
the covariance matrix.

All penalties are subtracted from the log-likelihood, so greater
penalty values correspond to a greater degree of penalization.

The penaties should be smooth so that they can be subtracted from log
likelihood functions and optimized using standard methods (i.e. L1
penalties do not belong here).
"""
import numpy as np

class Penalty:
    """
    A class for representing a scalar-value penalty.

    Parameters
    ----------
    weights : array_like
        A vector of weights that determines the weight of the penalty
        for each parameter.

    Notes
    -----
    The class has a member called `alpha` that scales the weights.
    """

    def __init__(self, weights=1.0):
        self.weights = weights
        self.alpha = 1.0

    def func(self, params):
        """
        A penalty function on a vector of parameters.

        Parameters
        ----------
        params : array_like
            A vector of parameters.

        Returns
        -------
        A scalar penaty value; greater values imply greater
        penalization.
        """
        params = np.asarray(params)
        return np.sum(self.alpha * self.weights * params**2)

    def deriv(self, params):
        """
        The gradient of a penalty function.

        Parameters
        ----------
        params : array_like
            A vector of parameters

        Returns
        -------
        The gradient of the penalty with respect to each element in
        `params`.
        """
        params = np.asarray(params)
        return 2 * self.alpha * self.weights * params

    def _null_weights(self, params):
        """work around for Null model

        This will not be needed anymore when we can use `self._null_drop_keys`
        as in DiscreteModels.
        TODO: check other models
        """
        if np.size(self.weights) > 1:
            return self.weights[:np.size(params)]
        else:
            return self.weights

class NonePenalty(Penalty):
    """
    A penalty that does not penalize.
    """

    def __init__(self, **kwds):
        super().__init__()
        if kwds:
            import warnings
            warnings.warn('keyword arguments are be ignored')

class L2(Penalty):
    """
    The L2 (ridge) penalty.
    """

    def __init__(self, weights=1.0):
        super().__init__(weights)

class L2Univariate(Penalty):
    """
    The L2 (ridge) penalty applied to each parameter.
    """

    def __init__(self, weights=None):
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

class PseudoHuber(Penalty):
    """
    The pseudo-Huber penalty.
    """

    def __init__(self, dlt, weights=1.0):
        super().__init__(weights)
        self.dlt = dlt

class SCAD(Penalty):
    """
    The SCAD penalty of Fan and Li.

    The SCAD penalty is linear around zero as a L1 penalty up to threshold tau.
    The SCAD penalty is constant for values larger than c*tau.
    The middle segment is quadratic and connect the two segments with a continuous
    derivative.
    The penalty is symmetric around zero.

    Parameterization follows Boo, Johnson, Li and Tan 2011.
    Fan and Li use lambda instead of tau, and a instead of c. Fan and Li
    recommend setting c=3.7.

    f(x) = { tau |x|                                        if 0 <= |x| < tau
           { -(|x|^2 - 2 c tau |x| + tau^2) / (2 (c - 1))   if tau <= |x| < c tau
           { (c + 1) tau^2 / 2                              if c tau <= |x|

    Parameters
    ----------
    tau : float
        slope and threshold for linear segment
    c : float
        factor for second threshold which is c * tau
    weights : None or array
        weights for penalty of each parameter. If an entry is zero, then the
        corresponding parameter will not be penalized.

    References
    ----------
    Buu, Anne, Norman J. Johnson, Runze Li, and Xianming Tan. "New variable
    selection methods for zero‐inflated count data with applications to the
    substance abuse field."
    Statistics in medicine 30, no. 18 (2011): 2326-2340.

    Fan, Jianqing, and Runze Li. "Variable selection via nonconcave penalized
    likelihood and its oracle properties."
    Journal of the American statistical Association 96, no. 456 (2001):
    1348-1360.
    """

    def __init__(self, tau, c=3.7, weights=1.0):
        super().__init__(weights)
        self.tau = tau
        self.c = c

    def deriv2(self, params):
        """Second derivative of function

        This returns scalar or vector in same shape as params, not a square
        Hessian. If the return is 1 dimensional, then it is the diagonal of
        the Hessian.
        """
        params = np.abs(params)
        tau, c = self.tau, self.c
        result = np.zeros_like(params)
        mask1 = params < tau
        mask2 = (tau <= params) & (params < c * tau)
        result[mask1] = 0
        result[mask2] = -1 / (c - 1)
        return result

class SCADSmoothed(SCAD):
    """
    The SCAD penalty of Fan and Li, quadratically smoothed around zero.

    This follows Fan and Li 2001 equation (3.7).

    Parameterization follows Boo, Johnson, Li and Tan 2011
    see docstring of SCAD

    Parameters
    ----------
    tau : float
        slope and threshold for linear segment
    c : float
        factor for second threshold
    c0 : float
        threshold for quadratically smoothed segment
    restriction : None or array
        linear constraints for

    Notes
    -----
    TODO: Use delegation instead of subclassing, so smoothing can be added to
    all penalty classes.
    """

    def __init__(self, tau, c=3.7, c0=None, weights=1.0, restriction=None):
        super().__init__(tau, c=c, weights=weights)
        self.tau = tau
        self.c = c
        self.c0 = c0 if c0 is not None else tau * 0.1
        if self.c0 > tau:
            raise ValueError('c0 cannot be larger than tau')
        c0 = self.c0
        weights = self.weights
        self.weights = 1.0
        deriv_c0 = super(SCADSmoothed, self).deriv(c0)
        value_c0 = super(SCADSmoothed, self).func(c0)
        self.weights = weights
        self.aq1 = value_c0 - 0.5 * deriv_c0 * c0
        self.aq2 = 0.5 * deriv_c0 / c0
        self.restriction = restriction

class ConstraintsPenalty:
    """
    Penalty applied to linear transformation of parameters

    Parameters
    ----------
    penalty: instance of penalty function
        currently this requires an instance of a univariate, vectorized
        penalty class
    weights : None or ndarray
        weights for adding penalties of transformed params
    restriction : None or ndarray
        If it is not None, then restriction defines a linear transformation
        of the parameters. The penalty function is applied to each transformed
        parameter independently.

    Notes
    -----
    `restrictions` allows us to impose penalization on contrasts or stochastic
    constraints of the original parameters.
    Examples for these contrast are difference penalities or all pairs
    penalties.
    """

    def __init__(self, penalty, weights=None, restriction=None):
        self.penalty = penalty
        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights
        if restriction is not None:
            restriction = np.asarray(restriction)
        self.restriction = restriction

    def func(self, params):
        """evaluate penalty function at params

        Parameter
        ---------
        params : ndarray
            array of parameters at which derivative is evaluated

        Returns
        -------
        deriv2 : ndarray
            value(s) of penalty function
        """
        if self.restriction is not None:
            params = np.dot(self.restriction, params)
        return np.sum(self.weights * self.penalty.func(params))

    def deriv(self, params):
        """first derivative of penalty function w.r.t. params

        Parameter
        ---------
        params : ndarray
            array of parameters at which derivative is evaluated

        Returns
        -------
        deriv2 : ndarray
            array of first partial derivatives
        """
        if self.restriction is not None:
            transformed_params = np.dot(self.restriction, params)
            grad = self.penalty.deriv(transformed_params)
            return np.dot(self.restriction.T, self.weights * grad)
        else:
            return self.weights * self.penalty.deriv(params)
    grad = deriv

    def deriv2(self, params):
        """second derivative of penalty function w.r.t. params

        Parameter
        ---------
        params : ndarray
            array of parameters at which derivative is evaluated

        Returns
        -------
        deriv2 : ndarray, 2-D
            second derivative matrix
        """
        if self.restriction is not None:
            transformed_params = np.dot(self.restriction, params)
            hess = np.diag(self.weights * self.penalty.deriv2(transformed_params))
            return np.dot(self.restriction.T, np.dot(hess, self.restriction))
        else:
            return np.diag(self.weights * self.penalty.deriv2(params))

class L2ConstraintsPenalty(ConstraintsPenalty):
    """convenience class of ConstraintsPenalty with L2 penalization
    """

    def __init__(self, weights=None, restriction=None, sigma_prior=None):
        if sigma_prior is not None:
            raise NotImplementedError('sigma_prior is not implemented yet')
        penalty = L2Univariate()
        super(L2ConstraintsPenalty, self).__init__(penalty, weights=weights, restriction=restriction)

class CovariancePenalty:

    def __init__(self, weight):
        self.weight = weight

    def func(self, mat, mat_inv):
        """
        Parameters
        ----------
        mat : square matrix
            The matrix to be penalized.
        mat_inv : square matrix
            The inverse of `mat`.

        Returns
        -------
        A scalar penalty value
        """
        return self.weight * (np.trace(mat) + np.trace(mat_inv) - 2 * mat.shape[0])

    def deriv(self, mat, mat_inv):
        """
        Parameters
        ----------
        mat : square matrix
            The matrix to be penalized.
        mat_inv : square matrix
            The inverse of `mat`.

        Returns
        -------
        A vector containing the gradient of the penalty
        with respect to each element in the lower triangle
        of `mat`.
        """
        n = mat.shape[0]
        grad = self.weight * (np.eye(n) - np.dot(mat_inv, mat_inv))
        return grad[np.tril_indices(n)]

class PSD(CovariancePenalty):
    """
    A penalty that converges to +infinity as the argument matrix
    approaches the boundary of the domain of symmetric, positive
    definite matrices.
    """
