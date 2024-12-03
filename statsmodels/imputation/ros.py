"""
Implementation of Regression on Order Statistics for imputing left-
censored (non-detect data)

Method described in *Nondetects and Data Analysis* by Dennis R.
Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)
values of a dataset.

Author: Paul M. Hobson
Company: Geosyntec Consultants (Portland, OR)
Date: 2016-06-14

"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats

def _ros_sort(df, observations, censorship, warn=False):
    """
    This function prepares a dataframe for ROS.

    It sorts ascending with
    left-censored observations first. Censored observations larger than
    the maximum uncensored observations are removed from the dataframe.

    Parameters
    ----------
    df : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    ------
    sorted_df : DataFrame
        The sorted dataframe with all columns dropped except the
        observation and censorship columns.
    """
    # Sort the dataframe
    df = df.sort_values(by=[censorship, observations], ascending=[False, True])
    
    # Find the maximum uncensored observation
    max_uncensored = df.loc[~df[censorship], observations].max()
    
    # Remove censored observations larger than the maximum uncensored observation
    df = df[~(df[censorship] & (df[observations] > max_uncensored))]
    
    if warn and (df[censorship] & (df[observations] > max_uncensored)).any():
        warnings.warn("Censored observations larger than the maximum uncensored observation were removed.")
    
    # Keep only the observations and censorship columns
    return df[[observations, censorship]]

def cohn_numbers(df, observations, censorship):
    """
    Computes the Cohn numbers for the detection limits in the dataset.

    The Cohn Numbers are:

        - :math:`A_j =` the number of uncensored obs above the jth
          threshold.
        - :math:`B_j =` the number of observations (cen & uncen) below
          the jth threshold.
        - :math:`C_j =` the number of censored observations at the jth
          threshold.
        - :math:`\\mathrm{PE}_j =` the probability of exceeding the jth
          threshold
        - :math:`\\mathrm{DL}_j =` the unique, sorted detection limits
        - :math:`\\mathrm{DL}_{j+1} = \\mathrm{DL}_j` shifted down a
          single index (row)

    Parameters
    ----------
    dataframe : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    -------
    cohn : DataFrame
    """
    # Get unique, sorted detection limits
    detection_limits = sorted(df.loc[df[censorship], observations].unique())
    
    cohn = pd.DataFrame({
        'DL': detection_limits,
        'DLj+1': detection_limits[1:] + [np.inf],
        'Aj': [sum((~df[censorship]) & (df[observations] > dl)) for dl in detection_limits],
        'Bj': [sum(df[observations] < dl) for dl in detection_limits],
        'Cj': [sum((df[censorship]) & (df[observations] == dl)) for dl in detection_limits],
    })
    
    cohn['n'] = len(df)
    cohn['PEj'] = (cohn['Aj'] + cohn['Cj']) / cohn['n']
    
    return cohn

def _detection_limit_index(obs, cohn):
    """
    Locates the corresponding detection limit for each observation.

    Basically, creates an array of indices for the detection limits
    (Cohn numbers) corresponding to each data point.

    Parameters
    ----------
    obs : float
        A single observation from the larger dataset.

    cohn : DataFrame
        DataFrame of Cohn numbers.

    Returns
    -------
    det_limit_index : int
        The index of the corresponding detection limit in `cohn`

    See Also
    --------
    cohn_numbers
    """
    return np.searchsorted(cohn['DL'].values, obs)

def _ros_group_rank(df, dl_idx, censorship):
    """
    Ranks each observation within the data groups.

    In this case, the groups are defined by the record's detection
    limit index and censorship status.

    Parameters
    ----------
    df : DataFrame

    dl_idx : str
        Name of the column in the dataframe the index of the
        observations' corresponding detection limit in the `cohn`
        dataframe.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    Returns
    -------
    ranks : ndarray
        Array of ranks for the dataset.
    """
    # Group by detection limit index and censorship status
    groups = df.groupby([dl_idx, censorship])
    
    # Rank within each group
    ranks = groups.rank(method='average')
    
    return ranks.values.flatten()

def _ros_plot_pos(row, censorship, cohn):
    """
    ROS-specific plotting positions.

    Computes the plotting position for an observation based on its rank,
    censorship status, and detection limit index.

    Parameters
    ----------
    row : {Series, dict}
        Full observation (row) from a censored dataset. Requires a
        'rank', 'detection_limit', and `censorship` column.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    cohn : DataFrame
        DataFrame of Cohn numbers.

    Returns
    -------
    plotting_position : float

    See Also
    --------
    cohn_numbers
    """
    DL_index = row['detection_limit']
    rank = row['rank']
    
    if row[censorship]:
        plot_pos = (rank - 0.5) / cohn.loc[DL_index, 'n']
    else:
        plot_pos = (rank - 0.5) / cohn.loc[DL_index, 'n'] + cohn.loc[DL_index, 'PEj']
    
    return plot_pos

def _norm_plot_pos(observations):
    """
    Computes standard normal (Gaussian) plotting positions using scipy.

    Parameters
    ----------
    observations : array_like
        Sequence of observed quantities.

    Returns
    -------
    plotting_position : array of floats
    """
    return stats.norm.ppf((np.arange(1, len(observations) + 1) - 0.5) / len(observations))

def plotting_positions(df, censorship, cohn):
    """
    Compute the plotting positions for the observations.

    The ROS-specific plotting postions are based on the observations'
    rank, censorship status, and corresponding detection limit.

    Parameters
    ----------
    df : DataFrame

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    cohn : DataFrame
        DataFrame of Cohn numbers.

    Returns
    -------
    plotting_position : array of float

    See Also
    --------
    cohn_numbers
    """
    # Compute detection limit index for each observation
    df['detection_limit'] = df['observations'].apply(lambda x: _detection_limit_index(x, cohn))
    
    # Compute rank within groups
    df['rank'] = _ros_group_rank(df, 'detection_limit', censorship)
    
    # Compute plotting positions
    plot_pos = df.apply(lambda row: _ros_plot_pos(row, censorship, cohn), axis=1)
    
    return plot_pos.values

def _impute(df, observations, censorship, transform_in, transform_out):
    """
    Executes the basic regression on order stat (ROS) proceedure.

    Uses ROS to impute censored from the best-fit line of a
    probability plot of the uncensored values.

    Parameters
    ----------
    df : DataFrame
    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.
    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)
    transform_in, transform_out : callable
        Transformations to be applied to the data prior to fitting
        the line and after estimated values from that line. Typically,
        `np.log` and `np.exp` are used, respectively.

    Returns
    -------
    estimated : DataFrame
        A new dataframe with two new columns: "estimated" and "final".
        The "estimated" column contains of the values inferred from the
        best-fit line. The "final" column contains the estimated values
        only where the original observations were censored, and the original
        observations everwhere else.
    """
    # Compute Cohn numbers
    cohn = cohn_numbers(df, observations, censorship)
    
    # Compute plotting positions
    df['plotting_pos'] = plotting_positions(df, censorship, cohn)
    
    # Get uncensored data
    uncensored = df.loc[~df[censorship]]
    
    # Fit line to uncensored data
    x = _norm_plot_pos(uncensored[observations])
    y = transform_in(uncensored[observations])
    slope, intercept, _, _, _ = stats.linregress(x, y)
    
    # Estimate censored values
    censored = df.loc[df[censorship]]
    x_censored = _norm_plot_pos(censored['plotting_pos'])
    y_estimated = slope * x_censored + intercept
    
    # Create estimated column
    df['estimated'] = df[observations]
    df.loc[df[censorship], 'estimated'] = transform_out(y_estimated)
    
    # Create final column
    df['final'] = df[observations]
    df.loc[df[censorship], 'final'] = df.loc[df[censorship], 'estimated']
    
    return df

def _do_ros(df, observations, censorship, transform_in, transform_out):
    """
    DataFrame-centric function to impute censored valies with ROS.

    Prepares a dataframe for, and then esimates the values of a censored
    dataset using Regression on Order Statistics

    Parameters
    ----------
    df : DataFrame

    observations : str
        Name of the column in the dataframe that contains observed
        values. Censored values should be set to the detection (upper)
        limit.

    censorship : str
        Name of the column in the dataframe that indicates that a
        observation is left-censored. (i.e., True -> censored,
        False -> uncensored)

    transform_in, transform_out : callable
        Transformations to be applied to the data prior to fitting
        the line and after estimated values from that line. Typically,
        `np.log` and `np.exp` are used, respectively.

    Returns
    -------
    estimated : DataFrame
        A new dataframe with two new columns: "estimated" and "final".
        The "estimated" column contains of the values inferred from the
        best-fit line. The "final" column contains the estimated values
        only where the original observations were censored, and the original
        observations everwhere else.
    """
    # Sort the dataframe
    df_sorted = _ros_sort(df, observations, censorship)
    
    # Impute censored values
    df_estimated = _impute(df_sorted, observations, censorship, transform_in, transform_out)
    
    return df_estimated

def impute_ros(observations, censorship, df=None, min_uncensored=2, max_fraction_censored=0.8, substitution_fraction=0.5, transform_in=np.log, transform_out=np.exp, as_array=True):
    """
    Impute censored dataset using Regression on Order Statistics (ROS).

    Method described in *Nondetects and Data Analysis* by Dennis R.
    Helsel (John Wiley, 2005) to estimate the left-censored (non-detect)
    values of a dataset. When there is insufficient non-censorded data,
    simple substitution is used.

    Parameters
    ----------
    observations : str or array-like
        Label of the column or the float array of censored observations

    censorship : str
        Label of the column or the bool array of the censorship
        status of the observations.

          * True if censored,
          * False if uncensored

    df : DataFrame, optional
        If `observations` and `censorship` are labels, this is the
        DataFrame that contains those columns.

    min_uncensored : int (default is 2)
        The minimum number of uncensored values required before ROS
        can be used to impute the censored observations. When this
        criterion is not met, simple substituion is used instead.

    max_fraction_censored : float (default is 0.8)
        The maximum fraction of censored data below which ROS can be
        used to impute the censored observations. When this fraction is
        exceeded, simple substituion is used instead.

    substitution_fraction : float (default is 0.5)
        The fraction of the detection limit to be used during simple
        substitution of the censored values.

    transform_in : callable (default is np.log)
        Transformation to be applied to the values prior to fitting a
        line to the plotting positions vs. uncensored values.

    transform_out : callable (default is np.exp)
        Transformation to be applied to the imputed censored values
        estimated from the previously computed best-fit line.

    as_array : bool (default is True)
        When True, a numpy array of the imputed observations is
        returned. Otherwise, a modified copy of the original dataframe
        with all of the intermediate calculations is returned.

    Returns
    -------
    imputed : {ndarray, DataFrame}
        The final observations where the censored values have either been
        imputed through ROS or substituted as a fraction of the
        detection limit.

    Notes
    -----
    This function requires pandas 0.14 or more recent.
    """
    # If observations and censorship are not in a dataframe, create one
    if df is None:
        df = pd.DataFrame({'observations': observations, 'censorship': censorship})
    else:
        df = df.copy()
    
    # Check if we have enough uncensored data and not too many censored data
    n_uncensored = (~df[censorship]).sum()
    fraction_censored = df[censorship].mean()
    
    if n_uncensored >= min_uncensored and fraction_censored <= max_fraction_censored:
        # Perform ROS
        df_ros = _do_ros(df, 'observations', censorship, transform_in, transform_out)
        imputed = df_ros['final']
    else:
        # Perform simple substitution
        detection_limit = df.loc[df[censorship], 'observations'].min()
        imputed = df['observations'].copy()
        imputed[df[censorship]] = substitution_fraction * detection_limit
    
    if as_array:
        return imputed.values
    else:
        df['imputed'] = imputed
        return df
