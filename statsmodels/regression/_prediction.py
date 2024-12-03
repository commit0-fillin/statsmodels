"""
Created on Fri Dec 19 11:29:18 2014

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
import pandas as pd

class PredictionResults:
    """
    Results class for predictions.

    Parameters
    ----------
    predicted_mean : ndarray
        The array containing the prediction means.
    var_pred_mean : ndarray
        The array of the variance of the prediction means.
    var_resid : ndarray
        The array of residual variances.
    df : int
        The degree of freedom used if dist is 't'.
    dist : {'norm', 't', object}
        Either a string for the normal or t distribution or another object
        that exposes a `ppf` method.
    row_labels : list[str]
        Row labels used in summary frame.
    """

    def __init__(self, predicted_mean, var_pred_mean, var_resid, df=None, dist=None, row_labels=None):
        self.predicted = predicted_mean
        self.var_pred = var_pred_mean
        self.df = df
        self.var_resid = var_resid
        self.row_labels = row_labels
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self.df,)
        else:
            self.dist = dist
            self.dist_args = ()

    def conf_int(self, obs=False, alpha=0.05):
        """
        Returns the confidence interval of the value, `effect` of the
        constraint.

        This is currently only available for t and z tests.

        Parameters
        ----------
        obs : bool, optional
            If True, returns prediction interval for observations.
            If False, returns confidence interval for the mean.
            Default is False.
        alpha : float, optional
            The significance level for the confidence interval.
            ie., The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        ci : ndarray, (k_constraints, 2)
            The array has the lower and the upper limit of the confidence
            interval in the columns.
        """
        std_errors = np.sqrt(self.var_pred + (self.var_resid if obs else 0))
        q = self.dist.ppf(1 - alpha / 2, *self.dist_args)
        lower = self.predicted - q * std_errors
        upper = self.predicted + q * std_errors
        return np.column_stack((lower, upper))

def get_prediction(self, exog=None, transform=True, weights=None, row_labels=None, pred_kwds=None):
    """
    Compute prediction results.

    Parameters
    ----------
    exog : array_like, optional
        The values for which you want to predict.
    transform : bool, optional
        If the model was fit via a formula, do you want to pass
        exog through the formula. Default is True. E.g., if you fit
        a model y ~ log(x1) + log(x2), and transform is True, then
        you can pass a data structure that contains x1 and x2 in
        their original form. Otherwise, you'd need to log the data
        first.
    weights : array_like, optional
        Weights interpreted as in WLS, used for the variance of the predicted
        residual.
    row_labels : list
        A list of row labels to use.  If not provided, read `exog` is
        available.
    pred_kwds : dict, optional
        Additional keyword arguments to be passed to the model's predict
        method.

    Returns
    -------
    linear_model.PredictionResults
        The prediction results instance contains prediction and prediction
        variance and can on demand calculate confidence intervals and summary
        tables for the prediction of the mean and of new observations.
    """
    if pred_kwds is None:
        pred_kwds = {}
    
    if exog is not None:
        if transform:
            exog = self.model.apply_transform(exog)
        exog = np.asarray(exog)
    else:
        exog = self.model.exog
    
    if weights is None:
        weights = getattr(self.model, 'weights', None)
    
    predicted_mean = self.model.predict(exog, **pred_kwds)
    var_pred_mean = self.model.predict_var(exog)
    var_resid = self.model.scale
    
    if weights is not None:
        var_resid = var_resid / weights
    
    df = getattr(self.model, 'df_resid', np.inf)
    dist = getattr(self.model, 'distribution', stats.norm)
    
    if row_labels is None:
        row_labels = getattr(exog, 'index', None)
    
    return PredictionResults(predicted_mean, var_pred_mean, var_resid, 
                             df=df, dist=dist, row_labels=row_labels)
