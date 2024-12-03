"""
Created on Fri Oct 22 08:13:38 2010

Author: josef-pktd
License: BSD (3-clause)
"""
import numpy as np
from numpy import polynomial as npp

class LagPolynomial(npp.Polynomial):

    def flip(self):
        """reverse polynomial coefficients
        """
        return LagPolynomial(self.coef[::-1])

    def div(self, other, maxlag=None):
        """padded division, pads numerator with zeros to maxlag
        """
        if maxlag is None:
            maxlag = max(len(self.coef), len(other.coef))
        
        num = self.pad(maxlag - 1).coef
        den = other.pad(maxlag - 1).coef
        
        quotient, remainder = npp.polynomial.polydiv(num, den)
        return LagPolynomial(quotient), LagPolynomial(remainder)
ar = LagPolynomial([1, -0.8])
arpad = ar.pad(10)
ma = LagPolynomial([1, 0.1])
mapad = ma.pad(10)
unit = LagPolynomial([1])
