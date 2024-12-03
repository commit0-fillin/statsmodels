"""Nonlinear Transformation classes


Created on Sat Apr 16 16:06:11 2011

Author: Josef Perktold
License : BSD
"""
import numpy as np

class TransformFunction:

    def __call__(self, x):
        return self.func(x)

class SquareFunc(TransformFunction):
    """class to hold quadratic function with inverse function and derivative

    using instance methods instead of class methods, if we want extension
    to parametrized function
    """

class NegSquareFunc(TransformFunction):
    """negative quadratic function

    """

class AbsFunc(TransformFunction):
    """class for absolute value transformation
    """

class LogFunc(TransformFunction):
    def func(self, x):
        return np.log(x)
    
    def inverse(self, x):
        return np.exp(x)
    
    def derivative(self, x):
        return 1 / x

class ExpFunc(TransformFunction):
    def func(self, x):
        return np.exp(x)
    
    def inverse(self, x):
        return np.log(x)
    
    def derivative(self, x):
        return np.exp(x)

class BoxCoxNonzeroFunc(TransformFunction):

    def __init__(self, lamda):
        self.lamda = lamda

class AffineFunc(TransformFunction):

    def __init__(self, constant, slope):
        self.constant = constant
        self.slope = slope

class ChainFunc(TransformFunction):

    def __init__(self, finn, fout):
        self.finn = finn
        self.fout = fout
if __name__ == '__main__':
    absf = AbsFunc()
    absf.func(5) == 5
    absf.func(-5) == 5
    absf.inverseplus(5) == 5
    absf.inverseminus(5) == -5
    chainf = ChainFunc(AffineFunc(1, 2), BoxCoxNonzeroFunc(2))
    print(chainf.func(3.0))
    chainf2 = ChainFunc(BoxCoxNonzeroFunc(2), AffineFunc(1, 2))
    print(chainf.func(3.0))
