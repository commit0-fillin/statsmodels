"""

Special functions for copulas not available in scipy

Created on Jan. 27, 2023
"""
import numpy as np
from scipy.special import factorial

class Sterling1:
    """Stirling numbers of the first kind
    """

    def __init__(self):
        self._cache = {}

    def __call__(self, n, k):
        key = str(n) + ',' + str(k)
        if key in self._cache.keys():
            return self._cache[key]
        if n == k == 0:
            return 1
        if n > 0 and k == 0:
            return 0
        if k > n:
            return 0
        result = sterling1(n - 1, k - 1) + (n - 1) * sterling1(n - 1, k)
        self._cache[key] = result
        return result

    def clear_cache(self):
        """clear cache of Sterling numbers
        """
        self._cache.clear()
sterling1 = Sterling1()

class Sterling2:
    """Stirling numbers of the second kind
    """

    def __init__(self):
        self._cache = {}

    def __call__(self, n, k):
        key = str(n) + ',' + str(k)
        if key in self._cache.keys():
            return self._cache[key]
        if n == k == 0:
            return 1
        if n > 0 and k == 0 or (n == 0 and k > 0):
            return 0
        if n == k:
            return 1
        if k > n:
            return 0
        result = k * sterling2(n - 1, k) + sterling2(n - 1, k - 1)
        self._cache[key] = result
        return result

    def clear_cache(self):
        """clear cache of Sterling numbers
        """
        pass
sterling2 = Sterling2()

def li3(z):
    """Polylogarithm for negative integer order -3

    Li(-3, z)
    """
    z = np.asarray(z)
    return z * (1 + z) / (1 - z)**4

def li4(z):
    """Polylogarithm for negative integer order -4

    Li(-4, z)
    """
    z = np.asarray(z)
    return z * (1 + z) * (1 + 4*z + z**2) / (1 - z)**6

def lin(n, z):
    """Polylogarithm for negative integer order -n

    Li(-n, z)

    https://en.wikipedia.org/wiki/Polylogarithm#Particular_values
    """
    z = np.asarray(z)
    result = np.zeros_like(z, dtype=float)
    
    for k in range(n + 1):
        result += sterling2(n + 1, k + 1) * factorial(k) * (z / (1 - z))**(k + 1)
    
    return (-1)**(n + 1) * result
