"""
Linear Algebra solvers and other helpers
"""
import numpy as np
__all__ = ['logdet_symm', 'stationary_solve', 'transf_constraints', 'matrix_sqrt']

def logdet_symm(m, check_symm=False):
    """
    Return log(det(m)) asserting positive definiteness of m.

    Parameters
    ----------
    m : array_like
        2d array that is positive-definite (and symmetric)

    Returns
    -------
    logdet : float
        The log-determinant of m.
    """
    if check_symm and not np.allclose(m, m.T):
        raise ValueError("Input matrix is not symmetric")
    
    # Use Cholesky decomposition to compute log-determinant
    L = np.linalg.cholesky(m)
    return 2 * np.sum(np.log(np.diag(L)))

def stationary_solve(r, b):
    """
    Solve a linear system for a Toeplitz correlation matrix.

    A Toeplitz correlation matrix represents the covariance of a
    stationary series with unit variance.

    Parameters
    ----------
    r : array_like
        A vector describing the coefficient matrix.  r[0] is the first
        band next to the diagonal, r[1] is the second band, etc.
    b : array_like
        The right-hand side for which we are solving, i.e. we solve
        Tx = b and return b, where T is the Toeplitz coefficient matrix.

    Returns
    -------
    The solution to the linear system.
    """
    from scipy import linalg
    r = np.asarray(r)
    b = np.asarray(b)
    n = len(b)
    
    # Construct the first column of the Toeplitz matrix
    c = np.r_[1, r[:n-1]]
    
    # Solve the system using Levinson recursion
    return linalg.solve_toeplitz(c, b)

def transf_constraints(constraints):
    """use QR to get transformation matrix to impose constraint

    Parameters
    ----------
    constraints : ndarray, 2-D
        restriction matrix with one constraints in rows

    Returns
    -------
    transf : ndarray
        transformation matrix to reparameterize so that constraint is
        imposed

    Notes
    -----
    This is currently and internal helper function for GAM.
    API not stable and will most likely change.

    The code for this function was taken from patsy spline handling, and
    corresponds to the reparameterization used by Wood in R's mgcv package.

    See Also
    --------
    statsmodels.base._constraints.TransformRestriction : class to impose
        constraints by reparameterization used by `_fit_constrained`.
    """
    Q, R = np.linalg.qr(constraints.T)
    m, n = constraints.shape
    transf = Q[:, m:]
    return transf

def matrix_sqrt(mat, inverse=False, full=False, nullspace=False, threshold=1e-15):
    """matrix square root for symmetric matrices

    Usage is for decomposing a covariance function S into a square root R
    such that

        R' R = S if inverse is False, or
        R' R = pinv(S) if inverse is True

    Parameters
    ----------
    mat : array_like, 2-d square
        symmetric square matrix for which square root or inverse square
        root is computed.
        There is no checking for whether the matrix is symmetric.
        A warning is issued if some singular values are negative, i.e.
        below the negative of the threshold.
    inverse : bool
        If False (default), then the matrix square root is returned.
        If inverse is True, then the matrix square root of the inverse
        matrix is returned.
    full : bool
        If full is False (default, then the square root has reduce number
        of rows if the matrix is singular, i.e. has singular values below
        the threshold.
    nullspace : bool
        If nullspace is true, then the matrix square root of the null space
        of the matrix is returned.
    threshold : float
        Singular values below the threshold are dropped.

    Returns
    -------
    msqrt : ndarray
        matrix square root or square root of inverse matrix.
    """
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    
    if np.any(s < -threshold):
        import warnings
        warnings.warn("Some singular values are negative.")
    
    mask = s > threshold
    s_sqrt = np.sqrt(s[mask])
    
    if inverse:
        s_sqrt = 1 / s_sqrt
    
    if nullspace:
        s_sqrt = np.sqrt(1 - s[mask]**2 / s[0]**2)
    
    if full:
        msqrt = U * s_sqrt
    else:
        msqrt = (U[:, mask] * s_sqrt).dot(Vt[mask])
    
    return msqrt
