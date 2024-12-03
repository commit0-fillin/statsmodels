"""Examples of non-linear functions for non-parametric regression

Created on Sat Jan 05 20:21:22 2013

Author: Josef Perktold
"""
import numpy as np

def fg1(x):
    """Fan and Gijbels example function 1

    """
    return 2 + 2 * np.cos(x)

def fg1eu(x):
    """Eubank similar to Fan and Gijbels example function 1

    """
    return 2 + 2 * np.exp(-16 * x**2)

def fg2(x):
    """Fan and Gijbels example function 2

    """
    return np.sin(2 * x) + 2 * np.exp(-16 * x**2)

def func1(x):
    """made up example with sin, square

    """
    return np.sin(x) + 0.5 * x**2
doc = {'description': "Base Class for Univariate non-linear example\n\n    Does not work on it's own.\n    needs additional at least self.func\n", 'ref': ''}

class _UnivariateFunction:
    __doc__ = '%(description)s\n\n    Parameters\n    ----------\n    nobs : int\n        number of observations to simulate\n    x : None or 1d array\n        If x is given then it is used for the exogenous variable instead of\n        creating a random sample\n    distr_x : None or distribution instance\n        Only used if x is None. The rvs method is used to create a random\n        sample of the exogenous (explanatory) variable.\n    distr_noise : None or distribution instance\n        The rvs method is used to create a random sample of the errors.\n\n    Attributes\n    ----------\n    x : ndarray, 1-D\n        exogenous or explanatory variable. x is sorted.\n    y : ndarray, 1-D\n        endogenous or response variable\n    y_true : ndarray, 1-D\n        expected values of endogenous or response variable, i.e. values of y\n        without noise\n    func : callable\n        underlying function (defined by subclass)\n\n    %(ref)s\n    '

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        if x is None:
            if distr_x is None:
                x = np.random.normal(loc=0, scale=self.s_x, size=nobs)
            else:
                x = distr_x.rvs(size=nobs)
            x.sort()
        self.x = x
        if distr_noise is None:
            noise = np.random.normal(loc=0, scale=self.s_noise, size=nobs)
        else:
            noise = distr_noise.rvs(size=nobs)
        if hasattr(self, 'het_scale'):
            noise *= self.het_scale(self.x)
        self.y_true = y_true = self.func(x)
        self.y = y_true + noise

    def plot(self, scatter=True, ax=None):
        """plot the mean function and optionally the scatter of the sample

        Parameters
        ----------
        scatter : bool
            If true, then add scatterpoints of sample to plot.
        ax : None or matplotlib axis instance
            If None, then a matplotlib.pyplot figure is created, otherwise
            the given axis, ax, is used.

        Returns
        -------
        Figure
            This is either the created figure instance or the one associated
            with ax if ax is given.

        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(self.x, self.y_true, 'r-', label='True function')
        if scatter:
            ax.scatter(self.x, self.y, alpha=0.5, label='Observed data')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

        return fig
doc = {'description': 'Fan and Gijbels example function 1\n\nlinear trend plus a hump\n', 'ref': '\nReferences\n----------\nFan, Jianqing, and Irene Gijbels. 1992. "Variable Bandwidth and Local\nLinear Regression Smoothers."\nThe Annals of Statistics 20 (4) (December): 2008-2036. doi:10.2307/2242378.\n\n'}

class UnivariateFanGijbels1(_UnivariateFunction):
    __doc__ = _UnivariateFunction.__doc__ % doc

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        self.s_x = 1.0
        self.s_noise = 0.7
        self.func = fg1
        super(self.__class__, self).__init__(nobs=nobs, x=x, distr_x=distr_x, distr_noise=distr_noise)
doc['description'] = 'Fan and Gijbels example function 2\n\nsin plus a hump\n'

class UnivariateFanGijbels2(_UnivariateFunction):
    __doc__ = _UnivariateFunction.__doc__ % doc

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        self.s_x = 1.0
        self.s_noise = 0.5
        self.func = fg2
        super(self.__class__, self).__init__(nobs=nobs, x=x, distr_x=distr_x, distr_noise=distr_noise)

class UnivariateFanGijbels1EU(_UnivariateFunction):
    """

    Eubank p.179f
    """

    def __init__(self, nobs=50, x=None, distr_x=None, distr_noise=None):
        if distr_x is None:
            from scipy import stats
            distr_x = stats.uniform
        self.s_noise = 0.15
        self.func = fg1eu
        super(self.__class__, self).__init__(nobs=nobs, x=x, distr_x=distr_x, distr_noise=distr_noise)

class UnivariateFunc1(_UnivariateFunction):
    """

    made up, with sin and quadratic trend
    """

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        if x is None and distr_x is None:
            from scipy import stats
            distr_x = stats.uniform(-2, 4)
        else:
            nobs = x.shape[0]
        self.s_noise = 2.0
        self.func = func1
        super(UnivariateFunc1, self).__init__(nobs=nobs, x=x, distr_x=distr_x, distr_noise=distr_noise)
