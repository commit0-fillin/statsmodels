"""
For comparison with sklearn.linear_model.LogisticRegression

Computes a regularzation path with both packages.  The coefficient values in
    either path are related by a "constant" in the sense that for any fixed
    value of the constraint C and log likelihood, there exists an l1
    regularization constant alpha such that the optimal solutions should be
    the same.  Note that alpha(C) is a nonlinear function in general.  Here we
    find alpha(C) by finding a reparameterization of the statsmodels path that
    makes the paths match up.  An equation is available, but to use it I would
    need to hack the sklearn code to extract the gradient of the log
    likelihood.


The results "prove" that the regularization paths are the same.  Note that
    finding the reparameterization is non-trivial since the coefficient paths
    are NOT monotonic.  As a result, the paths do not match up perfectly.
"""
from statsmodels.compat.python import lrange
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
use_spector = False
if use_spector:
    spector_data = sm.datasets.spector.load()
    X = spector_data.exog
    Y = spector_data.endog
else:
    raise Exception('The anes96 dataset is now loaded in as a short version that cannot be used here')
    anes96_data = sm.datasets.anes96.load_pandas()
    Y = anes96_data.exog.vote
N = 200
K = X.shape[1]
logit_mod = sm.Logit(Y, X)
sm_coeff = np.zeros((N, K))
if use_spector:
    alphas = 1 / np.logspace(-1, 2, N)
else:
    alphas = 1 / np.logspace(-3, 2, N)
for n, alpha in enumerate(alphas):
    logit_res = logit_mod.fit_regularized(method='l1', alpha=alpha, disp=False, trim_mode='off')
    sm_coeff[n, :] = logit_res.params
sk_coeff = np.zeros((N, K))
if use_spector:
    Cs = np.logspace(-0.45, 2, N)
else:
    Cs = np.logspace(-2.6, 0, N)
for n, C in enumerate(Cs):
    clf = linear_model.LogisticRegression(C=C, penalty='l1', fit_intercept=False)
    clf.fit(X, Y)
    sk_coeff[n, :] = clf.coef_
sk_special_X = np.fabs(sk_coeff[:, 2])
sm_special_X = np.fabs(sm_coeff[:, 2])
s = np.zeros(N)
s = np.searchsorted(sk_special_X, sm_special_X)
plt.figure(2)
plt.clf()
plt.grid()
plt.xlabel('Index in sklearn simulation')
plt.ylabel('Coefficient value')
plt.title('Regularization Paths')
colors = ['b', 'r', 'k', 'g', 'm', 'c', 'y']
for coeff, name in [(sm_coeff, 'sm'), (sk_coeff, 'sk')]:
    if name == 'sk':
        ltype = 'x'
        t = lrange(N)
    else:
        ltype = 'o'
        t = s
    for i in range(K):
        plt.plot(t, coeff[:, i], ltype + colors[i], label=name + '-X' + str(i))
plt.legend(loc='best')
plt.show()