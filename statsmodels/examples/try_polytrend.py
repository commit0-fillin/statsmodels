import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import statsmodels.api as sm
from statsmodels.datasets.macrodata import data
dta = data.load()
gdp = np.log(dta.data['realgdp'])
maxorder = 20
polybase = special.chebyt
polybase = special.legendre
t = np.linspace(-1, 1, len(gdp))
exog = np.column_stack([polybase(i)(t) for i in range(maxorder)])
fitted = [sm.OLS(gdp, exog[:, :maxr]).fit().fittedvalues for maxr in range(2, maxorder)]
print((np.corrcoef(exog[:, 1:6], rowvar=0) * 10000).astype(int))
plt.figure()
plt.plot(gdp, 'o')
for i in range(maxorder - 2):
    plt.plot(fitted[i])
plt.figure()
for i in range(maxorder - 4, maxorder - 2):
    plt.plot(gdp - fitted[i])
    plt.title(str(i + 2))
plt.figure()
plt.plot(gdp, '.')
plt.plot(fitted[-1], lw=2, color='r')
plt.plot(fitted[0], lw=2, color='g')
plt.title('GDP and Polynomial Trend')
plt.figure()
plt.plot(gdp - fitted[-1], lw=2, color='r')
plt.plot(gdp - fitted[0], lw=2, color='g')
plt.title('Residual GDP minus Polynomial Trend (green: linear, red: legendre(20))')
ex2 = t[:, None] ** np.arange(6)
q2, r2 = np.linalg.qr(ex2, mode='full')
np.max(np.abs(np.dot(q2.T, q2) - np.eye(6)))
plt.figure()
plt.plot(q2, lw=2)
plt.show()