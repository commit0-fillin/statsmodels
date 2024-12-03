import numpy as np
import pandas as pd
from scipy import stats

class PredictionResults:
    """
    Prediction results

    Parameters
    ----------
    predicted_mean : {ndarray, Series, DataFrame}
        The predicted mean values
    var_pred_mean : {ndarray, Series, DataFrame}
        The variance of the predicted mean values
    dist : {None, "norm", "t", rv_frozen}
        The distribution to use when constructing prediction intervals.
        Default is normal.
    df : int, optional
        The degree of freedom parameter for the t. Not used if dist is None,
        "norm" or a callable.
    row_labels : {Sequence[Hashable], pd.Index}
        Row labels to use for the summary frame. If None, attempts to read the
        index of ``predicted_mean``
    """

    def __init__(self, predicted_mean, var_pred_mean, dist=None, df=None, row_labels=None):
        self._predicted_mean = np.asarray(predicted_mean)
        self._var_pred_mean = np.asarray(var_pred_mean)
        self._df = df
        self._row_labels = row_labels
        if row_labels is None:
            self._row_labels = getattr(predicted_mean, 'index', None)
        self._use_pandas = self._row_labels is not None
        if dist != 't' and df is not None:
            raise ValueError('df must be None when dist is not "t"')
        if dist is None or dist == 'norm':
            self.dist = stats.norm
            self.dist_args = ()
        elif dist == 't':
            self.dist = stats.t
            self.dist_args = (self._df,)
        elif isinstance(dist, stats.distributions.rv_frozen):
            self.dist = dist
            self.dist_args = ()
        else:
            raise ValueError('dist must be a None, "norm", "t" or a callable.')

    @property
    def row_labels(self):
        """The row labels used in pandas-types."""
        return self._row_labels

    @property
    def predicted_mean(self):
        """The predicted mean"""
        if self._use_pandas:
            return pd.Series(self._predicted_mean, index=self._row_labels)
        return self._predicted_mean

    @property
    def var_pred_mean(self):
        """The variance of the predicted mean"""
        if self._use_pandas:
            return pd.Series(self._var_pred_mean, index=self._row_labels)
        return self._var_pred_mean

    @property
    def se_mean(self):
        """The standard deviation of the predicted mean"""
        se = np.sqrt(self.var_pred_mean)
        if self._use_pandas:
            return pd.Series(se, index=self._row_labels)
        return se

    @property
    def tvalues(self):
        """The ratio of the predicted mean to its standard deviation"""
        tvalues = self.predicted_mean / self.se_mean
        if self._use_pandas:
            return pd.Series(tvalues, index=self._row_labels)
        return tvalues

    def t_test(self, value=0, alternative='two-sided'):
        """
        z- or t-test for hypothesis that mean is equal to value

        Parameters
        ----------
        value : array_like
            value under the null hypothesis
        alternative : str
            'two-sided', 'larger', 'smaller'

        Returns
        -------
        stat : ndarray
            test statistic
        pvalue : ndarray
            p-value of the hypothesis test, the distribution is given by
            the attribute of the instance, specified in `__init__`. Default
            if not specified is the normal distribution.
        """
        stat = (self.predicted_mean - value) / self.se_mean
        
        if alternative == 'two-sided':
            pvalue = 2 * (1 - self.dist.cdf(np.abs(stat), *self.dist_args))
        elif alternative == 'larger':
            pvalue = 1 - self.dist.cdf(stat, *self.dist_args)
        elif alternative == 'smaller':
            pvalue = self.dist.cdf(stat, *self.dist_args)
        else:
            raise ValueError("alternative must be 'two-sided', 'larger' or 'smaller'")
        
        return stat, pvalue

    def conf_int(self, alpha=0.05):
        """
        Confidence interval construction for the predicted mean.

        This is currently only available for t and z tests.

        Parameters
        ----------
        alpha : float, optional
            The significance level for the prediction interval.
            The default `alpha` = .05 returns a 95% confidence interval.

        Returns
        -------
        pi : {ndarray, DataFrame}
            The array has the lower and the upper limit of the prediction
            interval in the columns.
        """
        q = self.dist.ppf(1 - alpha / 2, *self.dist_args)
        lower = self.predicted_mean - q * self.se_mean
        upper = self.predicted_mean + q * self.se_mean
        
        if self._use_pandas:
            return pd.DataFrame({'lower': lower, 'upper': upper}, index=self._row_labels)
        return np.column_stack((lower, upper))

    def summary_frame(self, alpha=0.05):
        """
        Summary frame of mean, variance and confidence interval.

        Returns
        -------
        DataFrame
            DataFrame containing four columns:

            * mean
            * mean_se
            * mean_ci_lower
            * mean_ci_upper

        Notes
        -----
        Fixes alpha to 0.05 so that the confidence interval should have 95%
        coverage.
        """
        ci = self.conf_int(alpha=alpha)
        if self._use_pandas:
            return pd.DataFrame({
                'mean': self.predicted_mean,
                'mean_se': self.se_mean,
                'mean_ci_lower': ci['lower'],
                'mean_ci_upper': ci['upper']
            }, index=self._row_labels)
        else:
            return pd.DataFrame({
                'mean': self.predicted_mean,
                'mean_se': self.se_mean,
                'mean_ci_lower': ci[:, 0],
                'mean_ci_upper': ci[:, 1]
            })
