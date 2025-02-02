import numpy as np
from statsmodels.sandbox.tools import cross_val
if __name__ == '__main__':
    import statsmodels.api as sm
    from statsmodels.datasets.stackloss import load
    data = load()
    data.exog = sm.tools.add_constant(data.exog, prepend=False)
    resols = sm.OLS(data.endog, data.exog).fit()
    print('\n OLS leave 1 out')
    for inidx, outidx in cross_val.LeaveOneOut(len(data.endog)):
        res = sm.OLS(data.endog[inidx], data.exog[inidx, :]).fit()
        print(data.endog[outidx], res.model.predict(res.params, data.exog[outidx, :], end=' '))
        print(data.endog[outidx] - res.model.predict(res.params, data.exog[outidx, :]))
    print('\n OLS leave 2 out')
    resparams = []
    for inidx, outidx in cross_val.LeavePOut(len(data.endog), 2):
        res = sm.OLS(data.endog[inidx], data.exog[inidx, :]).fit()
        resparams.append(res.params)
    resparams = np.array(resparams)
    print(resparams)
    doplots = 1
    if doplots:
        from matplotlib.font_manager import FontProperties
        import matplotlib.pyplot as plt
        plt.figure()
        figtitle = 'Leave2out parameter estimates'
        t = plt.gcf().text(0.5, 0.95, figtitle, horizontalalignment='center', fontproperties=FontProperties(size=16))
        for i in range(resparams.shape[1]):
            plt.subplot(4, 2, i + 1)
            plt.hist(resparams[:, i], bins=10)
        plt.show()
    for inidx, outidx in cross_val.KStepAhead(20, 2):
        print(inidx)
        print(np.ones(20)[inidx].sum(), np.arange(20)[inidx][-4:])
        print(outidx)
        print(np.nonzero(np.ones(20)[outidx])[0][()])