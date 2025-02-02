"""
Author: Padarn Wilson

Performance of normal reference plug-in estimator vs silverman. Sample is drawn
from a mixture of gaussians. Distribution has been chosen to be reasoanbly close
to normal.
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.nonparametric.api as npar
from statsmodels.distributions.mixture_rvs import mixture_rvs
np.random.seed(12345)
x = mixture_rvs([0.1, 0.9], size=200, dist=[stats.norm, stats.norm], kwargs=(dict(loc=0, scale=0.5), dict(loc=1, scale=0.5)))
kde = npar.KDEUnivariate(x)
kernel_names = ['Gaussian', 'Epanechnikov', 'Biweight', 'Triangular', 'Triweight', 'Cosine']
kernel_switch = ['gau', 'epa', 'tri', 'biw', 'triw', 'cos']
fig = plt.figure()
for ii, kn in enumerate(kernel_switch):
    ax = fig.add_subplot(2, 3, ii + 1)
    ax.hist(x, bins=20, density=True, alpha=0.25)
    kde.fit(kernel=kn, bw='silverman', fft=False)
    ax.plot(kde.support, kde.density)
    kde.fit(kernel=kn, bw='normal_reference', fft=False)
    ax.plot(kde.support, kde.density)
    ax.plot(kde.support, true_pdf(kde.support), color='black', linestyle='--')
    ax.set_title(kernel_names[ii])
ax.legend(['silverman', 'normal reference', 'true pdf'], loc='lower right')
ax.set_title('200 points')
plt.show()