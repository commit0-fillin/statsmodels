from statsmodels.compat.pandas import FUTURE_STACK
from collections import defaultdict
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
if __name__ == '__main__':
    from black import FileMode, TargetVersion, format_file_contents
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    PATH = os.environ.get('PSS_PATH', '..')
    print(f'Processing {PATH}')
    files = glob.glob(os.path.join(PATH, '*.npz'))
    groups = defaultdict(list)
    for f in files:
        keys = f.split('-')
        key = (int(keys[2]), int(keys[4]), keys[6] == 'True')
        if key[0] == 0:
            continue
        with np.load(f) as contents:
            idx = (100 * contents['percentiles']).astype(int)
            s = pd.Series(contents['q'], index=idx)
        groups[key].append(s)
    final = {}
    quantiles = (90, 95, 99, 99.9)
    crit_vals = {}
    ordered_keys = sorted(groups.keys())
    for key in ordered_keys:
        final[key] = pd.concat(groups[key], axis=1)
        cv = []
        for q in quantiles:
            cv.append(final[key].loc[int(100 * q)].mean())
        crit_vals[key] = cv
    df = pd.DataFrame(crit_vals).T
    df.index.names = ('k', 'case', 'I1')
    df.columns = quantiles
    for key, row in df.iterrows():
        crit_vals[key] = [round(val, 7) for val in list(row)]
    large_p = {}
    small_p = {}
    transform = {}
    max_stat = {}
    threshold = {}
    hp = 2
    for key in final:
        print(key)
        data = final[key]
        score = {}
        lr = LinearRegression(fit_intercept=False)
        for lp in (2, 3):
            for cut in range(40, data.shape[0] - 40):
                for log in (True, False):
                    cv = KFold(shuffle=True, random_state=20210903)
                    x, y = setup_regressors(data, lp, hp, cut, log)
                    k = (lp, hp, cut, log)
                    score[k] = cross_val_score(lr, x, y, scoring='neg_mean_absolute_error', cv=cv).sum()
        idx = pd.Series(score).idxmax()
        lp, hp, cut, log = idx
        assert log
        x, y = setup_regressors(data, lp, hp, cut, log)
        lr = lr.fit(x, y)
        large = lr.coef_[:1 + lp]
        if lp == 2:
            large = np.array(large.tolist() + [0.0])
        large_p[key] = large.tolist()
        small_p[key] = lr.coef_[1 + lp:].tolist()
        transform[key] = log
        max_stat[key] = np.inf
        threshold[key] = data.iloc[cut].mean()
        if small_p[key][2] < 0:
            max_stat[key] = small_p[key][1] / (-2 * small_p[key][2])
    for key in large_p:
        large_p[key] = [round(val, 5) for val in large_p[key]]
        small_p[key] = [round(val, 5) for val in small_p[key]]
    raw_code = f'\n#!/usr/bin/env python\n# coding: utf-8\n\n"""\nCritical value polynomials and related quantities for the bounds test of\n\nPesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches\n   to the analysis of level relationships. Journal of applied econometrics,\n   16(3), 289-326.\n\nThese were computed using 32,000,000 simulations for each key using the\nmethodology of PSS, who only used 40,000. The asymptotic P-value response\nfunctions were computed based on the simulated value. Critical values\nare the point estimates for the respective quantiles. The simulation code\nis contained in pss.py. The output files from this function are then\ntransformed using pss-process.py.\n\nThe format of the keys are (k, case, I1) where\n\n* k is is the number of x variables included in the model (0 is an ADF)\n* case is 1, 2, 3, 4 or 5 and corresponds to the PSS paper\n* I1 is True if X contains I1 variables and False if X is stationary\n\nThe parameters are for polynomials of order 3 (large) or 2 (small).\nstat_star is the value where the switch between large and small occurs.\nStat values less then stat_star use large_p, while values above use\nsmall_p. In all cases the stat is logged prior to computing the p-value\nso that the p-value is\n\n1 - Phi(c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3)\n\nwhere x = np.log(stat) and Phi() is the normal cdf.\n\nWhen this the models, the polynomial is evaluated at the natural log of the\ntest statistic and then the normal CDF of this value is computed to produce\nthe p-value.\n"""\n\n__all__ = ["large_p", "small_p", "crit_vals", "crit_percentiles", "stat_star"]\n\nlarge_p = {large_p}\n\nsmall_p = {small_p}\n\nstat_star = {threshold}\n\ncrit_percentiles = {quantiles}\n\ncrit_vals = {crit_vals}\n'
    targets = {TargetVersion.PY37, TargetVersion.PY38, TargetVersion.PY39}
    fm = FileMode(target_versions=targets, line_length=79)
    formatted_code = format_file_contents(raw_code, fast=False, mode=fm)
    with open('../pss_critical_values.py', 'w', newline='\n', encoding='utf-8') as out:
        out.write(formatted_code)