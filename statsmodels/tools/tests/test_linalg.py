from statsmodels.tools import linalg
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import toeplitz


def test_stationary_solve_1d():
    b = np.random.uniform(size=10)
    r = np.random.uniform(size=9)
    t = np.concatenate((np.r_[1], r))
    tmat = toeplitz(t)
    soln = np.linalg.solve(tmat, b)
    soln1 = linalg.stationary_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)


def test_stationary_solve_2d():
    b = np.random.uniform(size=(10, 2))
    r = np.random.uniform(size=9)
    t = np.concatenate((np.r_[1], r))
    tmat = toeplitz(t)
    soln = np.linalg.solve(tmat, b)
    soln1 = linalg.stationary_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)

def test_logdet_symm():
    # Test with a known positive definite matrix
    m = np.array([[2, 1], [1, 2]])
    expected_logdet = np.log(3)
    assert_allclose(linalg.logdet_symm(m), expected_logdet, rtol=1e-7)

    # Test with check_symm=True
    assert_allclose(linalg.logdet_symm(m, check_symm=True), expected_logdet, rtol=1e-7)

    # Test with non-symmetric matrix
    m_nonsym = np.array([[2, 1], [3, 2]])
    np.testing.assert_raises(ValueError, linalg.logdet_symm, m_nonsym, check_symm=True)

def test_transf_constraints():
    constraints = np.array([[1, 1, 0], [0, 1, 1]])
    transf = linalg.transf_constraints(constraints)
    
    # Check that the transformation matrix is orthogonal to the constraints
    assert_allclose(constraints.dot(transf), np.zeros((2, 1)), atol=1e-7)
    
    # Check that the transformation matrix has the correct shape
    assert transf.shape == (3, 1)

def test_matrix_sqrt():
    # Test with a known positive definite matrix
    m = np.array([[4, 1], [1, 4]])
    expected_sqrt = np.array([[1.93649167, 0.24206146],
                              [0.24206146, 1.93649167]])
    
    # Test regular square root
    assert_allclose(linalg.matrix_sqrt(m), expected_sqrt, rtol=1e-7)
    
    # Test inverse square root
    inv_expected_sqrt = np.linalg.inv(expected_sqrt)
    assert_allclose(linalg.matrix_sqrt(m, inverse=True), inv_expected_sqrt, rtol=1e-7)
    
    # Test with singular matrix
    m_singular = np.array([[1, 1], [1, 1]])
    sqrt_singular = linalg.matrix_sqrt(m_singular, threshold=1e-10)
    assert sqrt_singular.shape == (2, 1)
    
    # Test nullspace
    nullspace_sqrt = linalg.matrix_sqrt(m, nullspace=True)
    assert_allclose(nullspace_sqrt.T.dot(nullspace_sqrt), 
                    np.eye(2) - m/np.linalg.norm(m, 2), rtol=1e-7)
