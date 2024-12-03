import matplotlib.pyplot as plt
import numpy as np

def plot_scree(eigenvals, total_var, ncomp=None, x_label='factor'):
    """
    Plot of the ordered eigenvalues and variance explained for the loadings

    Parameters
    ----------
    eigenvals : array_like
        The eigenvalues
    total_var : float
        the total variance (for plotting percent variance explained)
    ncomp : int, optional
        Number of factors to include in the plot.  If None, will
        included the same as the number of maximum possible loadings
    x_label : str
        label of x-axis

    Returns
    -------
    Figure
        Handle to the figure.
    """
    eigenvals = np.asarray(eigenvals)
    if ncomp is None:
        ncomp = len(eigenvals)
    else:
        ncomp = min(ncomp, len(eigenvals))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot eigenvalues
    ax1.plot(range(1, ncomp + 1), eigenvals[:ncomp], 'bo-')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Scree Plot')
    
    # Plot cumulative variance explained
    variance_explained = eigenvals / total_var * 100
    cumulative_variance = np.cumsum(variance_explained)
    ax2.plot(range(1, ncomp + 1), cumulative_variance[:ncomp], 'ro-')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    
    plt.tight_layout()
    return fig

def plot_loadings(loadings, col_names=None, row_names=None, loading_pairs=None, percent_variance=None, title='Factor patterns'):
    """
    Plot factor loadings in 2-d plots

    Parameters
    ----------
    loadings : array like
        Each column is a component (or factor)
    col_names : a list of strings
        column names of `loadings`
    row_names : a list of strings
        row names of `loadings`
    loading_pairs : None or a list of tuples
        Specify plots. Each tuple (i, j) represent one figure, i and j is
        the loading number for x-axis and y-axis, respectively. If `None`,
        all combinations of the loadings will be plotted.
    percent_variance : array_like
        The percent variance explained by each factor.

    Returns
    -------
    figs : a list of figure handles
    """
    loadings = np.asarray(loadings)
    n_factors = loadings.shape[1]
    
    if col_names is None:
        col_names = [f'Factor {i+1}' for i in range(n_factors)]
    if row_names is None:
        row_names = [f'Var {i+1}' for i in range(loadings.shape[0])]
    
    if loading_pairs is None:
        loading_pairs = [(i, j) for i in range(n_factors) for j in range(i+1, n_factors)]
    
    figs = []
    for i, j in loading_pairs:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(loadings[:, i], loadings[:, j])
        
        for k, (x, y) in enumerate(zip(loadings[:, i], loadings[:, j])):
            ax.annotate(row_names[k], (x, y), xytext=(5, 5), textcoords='offset points')
        
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
        
        xlabel = f'{col_names[i]}'
        ylabel = f'{col_names[j]}'
        if percent_variance is not None:
            xlabel += f' ({percent_variance[i]:.1f}%)'
            ylabel += f' ({percent_variance[j]:.1f}%)'
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}\n{col_names[i]} vs {col_names[j]}')
        
        plt.tight_layout()
        figs.append(fig)
    
    return figs
