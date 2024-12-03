"""Analyze a set of multiple variables with a linear models

multiOLS:
    take a model and test it on a series of variables defined over a
    pandas dataset, returning a summary for each variable

multigroup:
    take a boolean vector and the definition of several groups of variables
    and test if the group has a fraction of true values higher than the
    rest. It allows to test if the variables in the group are significantly
    more significant than outside the group.
"""
from patsy import dmatrix
import pandas as pd
from statsmodels.api import OLS
from statsmodels.api import stats
import numpy as np
import logging

def _model2dataframe(model_endog, model_exog, model_type=OLS, **kwargs):
    """return a series containing the summary of a linear model

    All the exceding parameters will be redirected to the linear model
    """
    model = model_type(model_endog, model_exog, **kwargs)
    results = model.fit()
    
    summary = pd.Series({
        'params': results.params,
        'pvalues': results.pvalues,
        'bse': results.bse,
        'rsquared': results.rsquared,
        'rsquared_adj': results.rsquared_adj,
        'fvalue': results.fvalue,
        'f_pvalue': results.f_pvalue
    })
    
    return summary

def multiOLS(model, dataframe, column_list=None, method='fdr_bh', alpha=0.05, subset=None, model_type=OLS, **kwargs):
    """apply a linear model to several endogenous variables on a dataframe

    Take a linear model definition via formula and a dataframe that will be
    the environment of the model, and apply the linear model to a subset
    (or all) of the columns of the dataframe. It will return a dataframe
    with part of the information from the linear model summary.

    Parameters
    ----------
    model : str
        formula description of the model
    dataframe : pandas.dataframe
        dataframe where the model will be evaluated
    column_list : list[str], optional
        Names of the columns to analyze with the model.
        If None (Default) it will perform the function on all the
        eligible columns (numerical type and not in the model definition)
    model_type : model class, optional
        The type of model to be used. The default is the linear model.
        Can be any linear model (OLS, WLS, GLS, etc..)
    method : str, optional
        the method used to perform the pvalue correction for multiple testing.
        default is the Benjamini/Hochberg, other available methods are:

            `bonferroni` : one-step correction
            `sidak` : on-step correction
            `holm-sidak` :
            `holm` :
            `simes-hochberg` :
            `hommel` :
            `fdr_bh` : Benjamini/Hochberg
            `fdr_by` : Benjamini/Yekutieli

    alpha : float, optional
        the significance level used for the pvalue correction (default 0.05)
    subset : bool array
        the selected rows to be used in the regression

    all the other parameters will be directed to the model creation.

    Returns
    -------
    summary : pandas.DataFrame
        a dataframe containing an extract from the summary of the model
        obtained for each columns. It will give the model complexive f test
        result and p-value, and the regression value and standard deviarion
        for each of the regressors. The DataFrame has a hierachical column
        structure, divided as:

            - params: contains the parameters resulting from the models. Has
            an additional column named _f_test containing the result of the
            F test.
            - pval: the pvalue results of the models. Has the _f_test column
            for the significativity of the whole test.
            - adj_pval: the corrected pvalues via the multitest function.
            - std: uncertainties of the model parameters
            - statistics: contains the r squared statistics and the adjusted
            r squared.

    Notes
    -----
    The main application of this function is on system biology to perform
    a linear model testing of a lot of different parameters, like the
    different genetic expression of several genes.

    See Also
    --------
    statsmodels.stats.multitest
        contains several functions to perform the multiple p-value correction

    Examples
    --------
    Using the longley data as dataframe example

    >>> import statsmodels.api as sm
    >>> data = sm.datasets.longley.load_pandas()
    >>> df = data.exog
    >>> df['TOTEMP'] = data.endog

    This will perform the specified linear model on all the
    other columns of the dataframe
    >>> multiOLS('GNP + 1', df)

    This select only a certain subset of the columns
    >>> multiOLS('GNP + 0', df, ['GNPDEFL', 'TOTEMP', 'POP'])

    It is possible to specify a trasformation also on the target column,
    conforming to the patsy formula specification
    >>> multiOLS('GNP + 0', df, ['I(GNPDEFL**2)', 'center(TOTEMP)'])

    It is possible to specify the subset of the dataframe
    on which perform the analysis
    >> multiOLS('GNP + 1', df, subset=df.GNPDEFL > 90)

    Even a single column name can be given without enclosing it in a list
    >>> multiOLS('GNP + 0', df, 'GNPDEFL')
    """
    if column_list is None:
        column_list = [col for col in dataframe.columns if col not in model.split() and np.issubdtype(dataframe[col].dtype, np.number)]
    elif isinstance(column_list, str):
        column_list = [column_list]

    results = {}
    for column in column_list:
        y = dataframe[column]
        X = dmatrix(model, dataframe, return_type='dataframe')
        
        if subset is not None:
            y = y[subset]
            X = X[subset]
        
        model_results = _model2dataframe(y, X, model_type, **kwargs)
        results[column] = model_results

    summary = pd.DataFrame(results).T
    
    # Perform multiple testing correction
    pvalues = summary['pvalues']
    reject, pvals_corrected, _, _ = stats.multipletests(pvalues, alpha=alpha, method=method)
    summary['adj_pvals'] = pvals_corrected

    # Reorganize the DataFrame structure
    summary = pd.DataFrame({
        'params': summary['params'],
        'pval': summary['pvalues'],
        'adj_pval': summary['adj_pvals'],
        'std': summary['bse'],
        'statistics': pd.DataFrame({
            'rsquared': summary['rsquared'],
            'rsquared_adj': summary['rsquared_adj']
        })
    })

    summary['params']['_f_test'] = summary['fvalue']
    summary['pval']['_f_test'] = summary['f_pvalue']

    return summary

def _test_group(pvalues, group_name, group, exact=True):
    """test if the objects in the group are different from the general set.

    The test is performed on the pvalues set (ad a pandas series) over
    the group specified via a fisher exact test.
    """
    in_group = pvalues.index.isin(group)
    significant = pvalues < 0.05

    contingency_table = pd.crosstab(in_group, significant)
    
    if exact:
        _, p_value = stats.fisher_exact(contingency_table)
    else:
        _, p_value, _, _ = stats.chi2_contingency(contingency_table)
    
    odds_ratio = (contingency_table.loc[True, True] * contingency_table.loc[False, False]) / \
                 (contingency_table.loc[True, False] * contingency_table.loc[False, True])
    
    return pd.Series({
        'pvals': p_value,
        'increase': np.log(odds_ratio),
        '_in_sign': contingency_table.loc[True, True],
        '_in_non': contingency_table.loc[True, False],
        '_out_sign': contingency_table.loc[False, True],
        '_out_non': contingency_table.loc[False, False]
    }, name=group_name)

def multigroup(pvals, groups, exact=True, keep_all=True, alpha=0.05):
    """Test if the given groups are different from the total partition.

    Given a boolean array test if each group has a proportion of positives
    different than the complexive proportion.
    The test can be done as an exact Fisher test or approximated as a
    Chi squared test for more speed.

    Parameters
    ----------
    pvals : pandas series of boolean
        the significativity of the variables under analysis
    groups : dict of list
        the name of each category of variables under exam.
        each one is a list of the variables included
    exact : bool, optional
        If True (default) use the fisher exact test, otherwise
        use the chi squared test for contingencies tables.
        For high number of elements in the array the fisher test can
        be significantly slower than the chi squared.
    keep_all : bool, optional
        if False it will drop those groups where the fraction
        of positive is below the expected result. If True (default)
         it will keep all the significant results.
    alpha : float, optional
        the significativity level for the pvalue correction
        on the whole set of groups (not inside the groups themselves).

    Returns
    -------
    result_df: pandas dataframe
        for each group returns:

            pvals - the fisher p value of the test
            adj_pvals - the adjusted pvals
            increase - the log of the odd ratio between the
                internal significant ratio versus the external one
            _in_sign - significative elements inside the group
            _in_non - non significative elements inside the group
            _out_sign - significative elements outside the group
            _out_non - non significative elements outside the group

    Notes
    -----
    This test allow to see if a category of variables is generally better
    suited to be described for the model. For example to see if a predictor
    gives more information on demographic or economical parameters,
    by creating two groups containing the endogenous variables of each
    category.

    This function is conceived for medical dataset with a lot of variables
    that can be easily grouped into functional groups. This is because
    The significativity of a group require a rather large number of
    composing elements.

    Examples
    --------
    A toy example on a real dataset, the Guerry dataset from R
    >>> url = "https://raw.githubusercontent.com/vincentarelbundock/"
    >>> url = url + "Rdatasets/csv/HistData/Guerry.csv"
    >>> df = pd.read_csv(url, index_col='dept')

    evaluate the relationship between the various paramenters whith the Wealth
    >>> pvals = multiOLS('Wealth', df)['adj_pvals', '_f_test']

    define the groups
    >>> groups = {}
    >>> groups['crime'] = ['Crime_prop', 'Infanticide',
    ...     'Crime_parents', 'Desertion', 'Crime_pers']
    >>> groups['religion'] = ['Donation_clergy', 'Clergy', 'Donations']
    >>> groups['wealth'] = ['Commerce', 'Lottery', 'Instruction', 'Literacy']

    do the analysis of the significativity
    >>> multigroup(pvals < 0.05, groups)
    """
    results = []
    for group_name, group in groups.items():
        result = _test_group(pvals, group_name, group, exact)
        results.append(result)
    
    result_df = pd.DataFrame(results)
    
    # Perform multiple testing correction
    reject, pvals_corrected, _, _ = stats.multipletests(result_df['pvals'], alpha=alpha, method='fdr_bh')
    result_df['adj_pvals'] = pvals_corrected
    
    if not keep_all:
        result_df = result_df[result_df['increase'] > 0]
    
    return result_df.sort_values('adj_pvals')
