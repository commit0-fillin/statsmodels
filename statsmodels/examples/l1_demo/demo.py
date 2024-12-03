from optparse import OptionParser
import statsmodels.api as sm
import scipy as sp
from scipy import linalg
from scipy import stats
docstr = '\nDemonstrates l1 regularization for likelihood models.\nUse different models by setting mode = mnlogit, logit, or probit.\n\nExamples\n-------\n$ python demo.py --get_l1_slsqp_results  logit\n\n>>> import demo\n>>> demo.run_demo(\'logit\')\n\nThe Story\n---------\nThe maximum likelihood (ML) solution works well when the number of data\npoints is large and the noise is small.  When the ML solution starts\n"breaking", the regularized solution should do better.\n\nThe l1 Solvers\n--------------\nThe solvers are slower than standard Newton, and sometimes have\n    convergence issues Nonetheless, the final solution makes sense and\n    is often better than the ML solution.\nThe standard l1 solver is fmin_slsqp and is included with scipy.  It\n    sometimes has trouble verifying convergence when the data size is\n    large.\nThe l1_cvxopt_cp solver is part of CVXOPT and this package needs to be\n    installed separately.  It works well even for larger data sizes.\n'

def main():
    """
    Provides a CLI for the demo.
    """
    parser = OptionParser(docstr)
    parser.add_option("--get_l1_slsqp_results", action="store_true",
                      dest="get_l1_slsqp_results", default=False)
    parser.add_option("--get_l1_cvxopt_results", action="store_true",
                      dest="get_l1_cvxopt_results", default=False)
    parser.add_option("--print_summaries", action="store_true",
                      dest="print_summaries", default=False)
    parser.add_option("--save_arrays", action="store_true",
                      dest="save_arrays", default=False)
    parser.add_option("--load_old_arrays", action="store_true",
                      dest="load_old_arrays", default=False)

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error("Incorrect number of arguments")

    mode = args[0]
    if mode not in ['logit', 'mnlogit', 'probit']:
        parser.error("Mode must be 'logit', 'mnlogit', or 'probit'")

    run_demo(mode, get_l1_slsqp_results=options.get_l1_slsqp_results,
             get_l1_cvxopt_results=options.get_l1_cvxopt_results,
             print_summaries=options.print_summaries,
             save_arrays=options.save_arrays,
             load_old_arrays=options.load_old_arrays)

def run_demo(mode, base_alpha=0.01, N=500, get_l1_slsqp_results=False, get_l1_cvxopt_results=False, num_nonconst_covariates=10, noise_level=0.2, cor_length=2, num_zero_params=8, num_targets=3, print_summaries=False, save_arrays=False, load_old_arrays=False):
    """
    Run the demo and print results.

    Parameters
    ----------
    mode : str
        either 'logit', 'mnlogit', or 'probit'
    base_alpha :  Float
        Size of regularization param (the param actually used will
        automatically scale with data size in this demo)
    N : int
        Number of data points to generate for fit
    get_l1_slsqp_results : bool,
        Do an l1 fit using slsqp.
    get_l1_cvxopt_results : bool
        Do an l1 fit using cvxopt
    num_nonconst_covariates : int
        Number of covariates that are not constant
        (a constant will be prepended)
    noise_level : float (non-negative)
        Level of the noise relative to signal
    cor_length : float (non-negative)
        Correlation length of the (Gaussian) independent variables
    num_zero_params : int
        Number of parameters equal to zero for every target in logistic
        regression examples.
    num_targets : int
        Number of choices for the endogenous response in multinomial logit
        example
    print_summaries : bool
        print the full fit summary.
    save_arrays : bool
        Save exog/endog/true_params to disk for future use.
    load_old_arrays
        Load exog/endog/true_params arrays from disk.
    """
    if load_old_arrays:
        exog = sp.load('exog.npy')
        endog = sp.load('endog.npy')
        true_params = sp.load('true_params.npy')
    else:
        exog = get_exog(N, num_nonconst_covariates, cor_length)
        true_params = sp.randn(exog.shape[1])
        true_params[:num_zero_params] = 0

        if mode == 'logit':
            endog = get_logit_endog(true_params, exog, noise_level)
        elif mode == 'probit':
            endog = get_probit_endog(true_params, exog, noise_level)
        elif mode == 'mnlogit':
            true_params = sp.repeat(true_params, num_targets - 1).reshape(-1, num_targets - 1)
            true_params[:num_zero_params, :] = 0
            endog = get_logit_endog(true_params, exog, noise_level)
        else:
            raise ValueError("Invalid mode. Choose 'logit', 'probit', or 'mnlogit'.")

    if save_arrays:
        sp.save('exog.npy', exog)
        sp.save('endog.npy', endog)
        sp.save('true_params.npy', true_params)

    alpha = base_alpha * N / 500.

    if mode == 'logit':
        model = sm.Logit(endog, exog)
    elif mode == 'probit':
        model = sm.Probit(endog, exog)
    elif mode == 'mnlogit':
        model = sm.MNLogit(endog, exog)

    results_str = run_solvers(model, true_params, alpha, get_l1_slsqp_results, get_l1_cvxopt_results, print_summaries)
    print(results_str)

def run_solvers(model, true_params, alpha, get_l1_slsqp_results, get_l1_cvxopt_results, print_summaries):
    """
    Runs the solvers using the specified settings and returns a result string.
    Works the same for any l1 penalized likelihood model.
    """
    results = []
    
    # Run ML estimation
    ml_results = model.fit(disp=0)
    results.append(('ML', ml_results))

    # Run L1 regularized estimation with SLSQP
    if get_l1_slsqp_results:
        l1_results_slsqp = model.fit_regularized(method='l1', alpha=alpha, disp=0)
        results.append(('L1 (SLSQP)', l1_results_slsqp))

    # Run L1 regularized estimation with CVXOPT
    if get_l1_cvxopt_results:
        try:
            l1_results_cvxopt = model.fit_regularized(method='l1_cvxopt_cp', alpha=alpha, disp=0)
            results.append(('L1 (CVXOPT)', l1_results_cvxopt))
        except ImportError:
            print("CVXOPT not available. Skipping L1 (CVXOPT) estimation.")

    return get_summary_str(results, true_params, get_l1_slsqp_results, get_l1_cvxopt_results, print_summaries)

def get_summary_str(results, true_params, get_l1_slsqp_results, get_l1_cvxopt_results, print_summaries):
    """
    Gets a string summarizing the results.
    """
    summary = []
    for name, result in results:
        rmse = get_RMSE(result, true_params)
        summary.append(f"{name} RMSE: {rmse:.4f}")
        
        if print_summaries:
            summary.append(str(result.summary()))
        
        summary.append("\nEstimated parameters:")
        summary.append(str(result.params))
        summary.append("\nTrue parameters:")
        summary.append(str(true_params))
        summary.append("\n")

    return "\n".join(summary)

def get_RMSE(results, true_params):
    """
    Gets the (normalized) root mean square error.
    """
    params = results.params
    if params.ndim == 2 and true_params.ndim == 2:
        # For multinomial logit
        diff = params - true_params
        mse = (diff ** 2).mean()
    else:
        # For binary logit and probit
        mse = ((params - true_params) ** 2).mean()
    return sp.sqrt(mse) / sp.absolute(true_params).mean()

def get_logit_endog(true_params, exog, noise_level):
    """
    Gets an endogenous response that is consistent with the true_params,
        perturbed by noise at noise_level.
    """
    if true_params.ndim == 2:
        # Multinomial logit
        linear_predictor = sp.dot(exog, true_params)
        probs = sp.exp(linear_predictor) / (1 + sp.exp(linear_predictor).sum(axis=1, keepdims=True))
        probs = sp.column_stack((1 - probs.sum(axis=1), probs))
    else:
        # Binary logit
        linear_predictor = sp.dot(exog, true_params)
        probs = 1 / (1 + sp.exp(-linear_predictor))

    noise = sp.random.normal(0, noise_level, size=probs.shape)
    noisy_probs = probs + noise
    noisy_probs = sp.clip(noisy_probs, 0, 1)
    noisy_probs /= noisy_probs.sum(axis=1, keepdims=True)

    return sp.random.multinomial(1, noisy_probs[i]) for i in range(len(noisy_probs))

def get_probit_endog(true_params, exog, noise_level):
    """
    Gets an endogenous response that is consistent with the true_params,
        perturbed by noise at noise_level.
    """
    linear_predictor = sp.dot(exog, true_params)
    probs = stats.norm.cdf(linear_predictor)

    noise = sp.random.normal(0, noise_level, size=probs.shape)
    noisy_probs = sp.clip(probs + noise, 0, 1)

    return sp.random.binomial(1, noisy_probs)

def get_exog(N, num_nonconst_covariates, cor_length):
    """
    Returns an exog array with correlations determined by cor_length.
    The covariance matrix of exog will have (asymptotically, as
    :math:'N\\to\\inf')
    .. math:: Cov[i,j] = \\exp(-|i-j| / cor_length)

    Higher cor_length makes the problem more ill-posed, and easier to screw
        up with noise.
    BEWARE:  With very long correlation lengths, you often get a singular KKT
        matrix (during the l1_cvxopt_cp fit)
    """
    cov = sp.zeros((num_nonconst_covariates, num_nonconst_covariates))
    for i in range(num_nonconst_covariates):
        for j in range(num_nonconst_covariates):
            cov[i, j] = sp.exp(-abs(i - j) / cor_length)

    exog = sp.random.multivariate_normal(sp.zeros(num_nonconst_covariates), cov, size=N)
    exog = sp.column_stack((sp.ones(N), exog))  # Add constant term
    return exog
if __name__ == '__main__':
    main()
