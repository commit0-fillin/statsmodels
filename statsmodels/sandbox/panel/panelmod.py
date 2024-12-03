"""
Sandbox Panel Estimators

References
-----------

Baltagi, Badi H. `Econometric Analysis of Panel Data.` 4th ed. Wiley, 2008.
"""
from functools import reduce
import numpy as np
from statsmodels.regression.linear_model import GLS
from scipy import stats
__all__ = ['PanelModel']
from pandas import Panel

def group(X):
    """
    Returns unique numeric values for groups without sorting.

    Examples
    --------
    >>> X = np.array(['a','a','b','c','b','c'])
    >>> group(X)
    >>> g
    array([ 0.,  0.,  1.,  2.,  1.,  2.])
    """
    unique_values = {}
    result = np.zeros(len(X), dtype=float)
    counter = 0
    for i, x in enumerate(X):
        if x not in unique_values:
            unique_values[x] = counter
            counter += 1
        result[i] = unique_values[x]
    return result

def repanel_cov(groups, sigmas):
    """calculate error covariance matrix for random effects model

    Parameters
    ----------
    groups : ndarray, (nobs, nre) or (nobs,)
        array of group/category observations
    sigma : ndarray, (nre+1,)
        array of standard deviations of random effects,
        last element is the standard deviation of the
        idiosyncratic error

    Returns
    -------
    omega : ndarray, (nobs, nobs)
        covariance matrix of error
    omegainv : ndarray, (nobs, nobs)
        inverse covariance matrix of error
    omegainvsqrt : ndarray, (nobs, nobs)
        squareroot inverse covariance matrix of error
        such that omega = omegainvsqrt * omegainvsqrt.T

    Notes
    -----
    This does not use sparse matrices and constructs nobs by nobs
    matrices. Also, omegainvsqrt is not sparse, i.e. elements are non-zero
    """
    nobs = len(groups)
    if groups.ndim == 1:
        groups = groups.reshape(-1, 1)
    nre = groups.shape[1]

    # Construct the covariance matrix
    omega = np.zeros((nobs, nobs))
    for i in range(nre):
        omega += (sigmas[i]**2) * (groups[:, i:i+1] == groups[:, i:i+1].T)
    omega += (sigmas[-1]**2) * np.eye(nobs)

    # Calculate inverse and square root inverse
    eigvals, eigvecs = np.linalg.eigh(omega)
    omegainv = np.dot(eigvecs, np.dot(np.diag(1/eigvals), eigvecs.T))
    omegainvsqrt = np.dot(eigvecs, np.dot(np.diag(1/np.sqrt(eigvals)), eigvecs.T))

    return omega, omegainv, omegainvsqrt

class PanelData(Panel):
    pass

class PanelModel:
    """
    An abstract statistical model class for panel (longitudinal) datasets.

    Parameters
    ----------
    endog : array_like or str
        If a pandas object is used then endog should be the name of the
        endogenous variable as a string.
#    exog
#    panel_arr
#    time_arr
    panel_data : pandas.Panel object

    Notes
    -----
    If a pandas object is supplied it is assumed that the major_axis is time
    and that the minor_axis has the panel variable.
    """

    def __init__(self, endog=None, exog=None, panel=None, time=None, xtnames=None, equation=None, panel_data=None):
        if panel_data is None:
            self.initialize(endog, exog, panel, time, xtnames, equation)

    def initialize(self, endog, exog, panel, time, xtnames, equation):
        """
        Initialize plain array model.

        See PanelModel
        """
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)
        self.panel = np.asarray(panel)
        self.time = np.asarray(time)
        self.xtnames = xtnames or ['panel', 'time']
        self.equation = equation

        if self.exog.ndim == 1:
            self.exog = self.exog.reshape(-1, 1)

        self.nobs, self.k_vars = self.exog.shape
        self.n_panels = len(np.unique(self.panel))
        self.n_periods = len(np.unique(self.time))

        if len(self.endog) != self.nobs:
            raise ValueError("endog and exog must have the same number of observations")
        if len(self.panel) != self.nobs or len(self.time) != self.nobs:
            raise ValueError("panel and time must have the same length as endog")

    def _group_mean(self, X, index='oneway', counts=False, dummies=False):
        """
        Get group means of X by time or by panel.

        index default is panel
        """
        if index == 'oneway':
            groups = self.panel
        elif index == 'time':
            groups = self.time
        else:
            raise ValueError("index must be 'oneway' or 'time'")

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_vars = X.shape[1]

        group_means = np.zeros((n_groups, n_vars))
        group_counts = np.zeros(n_groups)

        for i, group in enumerate(unique_groups):
            mask = (groups == group)
            group_means[i] = X[mask].mean(axis=0)
            group_counts[i] = mask.sum()

        if dummies:
            dummy_matrix = (groups[:, None] == unique_groups).astype(float)
            if counts:
                return group_means, group_counts, dummy_matrix
            else:
                return group_means, dummy_matrix
        elif counts:
            return group_means, group_counts
        else:
            return group_means

    def fit(self, model=None, method=None, effects='oneway'):
        """
        method : LSDV, demeaned, MLE, GLS, BE, FE, optional
        model :
                between
                fixed
                random
                pooled
                [gmm]
        effects :
                oneway
                time
                twoway
        femethod : demeaned (only one implemented)
                   WLS
        remethod :
                swar -
                amemiya
                nerlove
                walhus


        Notes
        -----
        This is unfinished.  None of the method arguments work yet.
        Only oneway effects should work.
        """
        if model == 'pooled':
            return self._fit_pooled()
        elif model == 'between':
            return self._fit_between(effects)
        elif model == 'fixed':
            return self._fit_fixed(effects)
        elif model == 'random':
            return self._fit_random(effects)
        else:
            raise ValueError("Unsupported model type")

    def _fit_pooled(self):
        return GLS(self.endog, self.exog).fit()

    def _fit_between(self, effects):
        if effects == 'oneway':
            group_means_y = self._group_mean(self.endog)
            group_means_X = self._group_mean(self.exog)
        elif effects == 'time':
            group_means_y = self._group_mean(self.endog, index='time')
            group_means_X = self._group_mean(self.exog, index='time')
        else:
            raise ValueError("effects must be 'oneway' or 'time'")
        
        return GLS(group_means_y, group_means_X).fit()

    def _fit_fixed(self, effects):
        if effects == 'oneway':
            demeaned_y = self.endog - self._group_mean(self.endog)
            demeaned_X = self.exog - self._group_mean(self.exog)
        elif effects == 'time':
            demeaned_y = self.endog - self._group_mean(self.endog, index='time')
            demeaned_X = self.exog - self._group_mean(self.exog, index='time')
        elif effects == 'twoway':
            demeaned_y = self.endog - self._group_mean(self.endog) - self._group_mean(self.endog, index='time') + self.endog.mean()
            demeaned_X = self.exog - self._group_mean(self.exog) - self._group_mean(self.exog, index='time') + self.exog.mean(axis=0)
        else:
            raise ValueError("effects must be 'oneway', 'time', or 'twoway'")
        
        return GLS(demeaned_y, demeaned_X).fit()

    def _fit_random(self, effects):
        # This is a simplified random effects model using GLS
        # A more sophisticated implementation would use the appropriate variance components
        if effects == 'oneway':
            group_means_y = self._group_mean(self.endog)
            group_means_X = self._group_mean(self.exog)
            between_resid = group_means_y - np.dot(group_means_X, np.linalg.lstsq(group_means_X, group_means_y, rcond=None)[0])
            sigma_u = np.var(between_resid)
            sigma_e = np.var(self.endog - np.dot(self.exog, np.linalg.lstsq(self.exog, self.endog, rcond=None)[0]))
            theta = 1 - np.sqrt(sigma_e / (self.n_periods * sigma_u + sigma_e))
            y_star = self.endog - theta * self._group_mean(self.endog)
            X_star = self.exog - theta * self._group_mean(self.exog)
        else:
            raise ValueError("Only oneway random effects are implemented")
        
        return GLS(y_star, X_star).fit()

class SURPanel(PanelModel):
    pass

class SEMPanel(PanelModel):
    pass

class DynamicPanel(PanelModel):
    pass
if __name__ == '__main__':
    import numpy.lib.recfunctions as nprf
    import pandas
    from pandas import Panel
    import statsmodels.api as sm
    data = sm.datasets.grunfeld.load()
    endog = data.endog[:-20]
    fullexog = data.exog[:-20]
    panel_arr = nprf.append_fields(fullexog, 'investment', endog, float, usemask=False)
    panel_df = pandas.DataFrame(panel_arr)
    panel_panda = panel_df.set_index(['year', 'firm']).to_panel()
    exog = fullexog[['value', 'capital']].view(float).reshape(-1, 2)
    exog = sm.add_constant(exog, prepend=False)
    panel = group(fullexog['firm'])
    year = fullexog['year']
    panel_mod = PanelModel(endog, exog, panel, year, xtnames=['firm', 'year'], equation='invest value capital')
    panel_ols = panel_mod.fit(model='pooled')
    panel_be = panel_mod.fit(model='between', effects='oneway')
    panel_fe = panel_mod.fit(model='fixed', effects='oneway')
    panel_bet = panel_mod.fit(model='between', effects='time')
    panel_fet = panel_mod.fit(model='fixed', effects='time')
    panel_fe2 = panel_mod.fit(model='fixed', effects='twoways')
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    nobs = groups.shape[0]
    groupuniq = np.unique(groups)
    periods = np.array([0, 1, 2, 1, 2, 0, 1, 2])
    perioduniq = np.unique(periods)
    dummygr = (groups[:, None] == groupuniq).astype(float)
    dummype = (periods[:, None] == perioduniq).astype(float)
    sigma = 1.0
    sigmagr = np.sqrt(2.0)
    sigmape = np.sqrt(3.0)
    dummyall = np.c_[sigmagr * dummygr, sigmape * dummype]
    omega = np.dot(dummyall, dummyall.T) + sigma * np.eye(nobs)
    print(omega)
    print(np.linalg.cholesky(omega))
    ev, evec = np.linalg.eigh(omega)
    omegainv = np.dot(evec, (1 / ev * evec).T)
    omegainv2 = np.linalg.inv(omega)
    omegacomp = np.dot(evec, (ev * evec).T)
    print(np.max(np.abs(omegacomp - omega)))
    print(np.max(np.abs(np.dot(omegainv, omega) - np.eye(nobs))))
    omegainvhalf = evec / np.sqrt(ev)
    print(np.max(np.abs(np.dot(omegainvhalf, omegainvhalf.T) - omegainv)))
    sigmas2 = np.array([sigmagr, sigmape, sigma])
    groups2 = np.column_stack((groups, periods))
    omega_, omegainv_, omegainvhalf_ = repanel_cov(groups2, sigmas2)
    print(np.max(np.abs(omega_ - omega)))
    print(np.max(np.abs(omegainv_ - omegainv)))
    print(np.max(np.abs(omegainvhalf_ - omegainvhalf)))
    Pgr = reduce(np.dot, [dummygr, np.linalg.inv(np.dot(dummygr.T, dummygr)), dummygr.T])
    Qgr = np.eye(nobs) - Pgr
    print(np.max(np.abs(np.dot(Qgr, groups))))
