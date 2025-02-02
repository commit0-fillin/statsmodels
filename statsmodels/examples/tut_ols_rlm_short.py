"""Examples: comparing OLS and RLM

robust estimators and outliers

RLM is less influenced by outliers than OLS and has estimated slope
closer to true slope and not tilted like OLS.

Note: uncomment plt.show() to display graphs
"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
np.random.seed(98765789)
nsample = 50
x1 = np.linspace(0, 20, nsample)
X = np.c_[x1, np.ones(nsample)]
sig = 0.3
beta = [0.5, 5.0]
y_true2 = np.dot(X, beta)
y2 = y_true2 + sig * 1.0 * np.random.normal(size=nsample)
y2[[39, 41, 43, 45, 48]] -= 5
plt.figure()
plt.plot(x1, y2, 'o', x1, y_true2, 'b-')
res2 = sm.OLS(y2, X).fit()
print('OLS: parameter estimates: slope, constant')
print(res2.params)
print('standard deviation of parameter estimates')
print(res2.bse)
prstd, iv_l, iv_u = wls_prediction_std(res2)
plt.plot(x1, res2.fittedvalues, 'r-')
plt.plot(x1, iv_u, 'r--')
plt.plot(x1, iv_l, 'r--')
resrlm2 = sm.RLM(y2, X).fit()
print('\nRLM: parameter estimates: slope, constant')
print(resrlm2.params)
print('standard deviation of parameter estimates')
print(resrlm2.bse)
plt.plot(x1, resrlm2.fittedvalues, 'g.-')
plt.title('Data with Outliers; blue: true, red: OLS, green: RLM')
plt.show()