""" Transformation Classes as generators for Archimedean copulas


Created on Wed Jan 27 14:33:40 2021

Author: Josef Perktold
License: BSD-3

"""
import warnings
import numpy as np
from scipy.special import expm1, gamma

class Transforms:

    def __init__(self):
        # Initialize any necessary attributes here
        # For now, we'll keep it as pass since no specific attributes are required
        pass

class TransfFrank(Transforms):
    pass

class TransfClayton(Transforms):
    pass

class TransfGumbel(Transforms):
    """
    requires theta >=1
    """

class TransfIndep(Transforms):
    pass

class _TransfPower(Transforms):
    """generic multivariate Archimedean copula with additional power transforms

    Nelson p.144, equ. 4.5.2

    experimental, not yet tested and used
    """

    def __init__(self, transform):
        self.transform = transform
