"""
Created on Thu Aug 12 14:59:03 2010

Warning: not tried out or tested yet, Done

Author: josef-pktd
"""
import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats.distributions import rv_continuous
import matplotlib.pyplot as plt
from numpy import where, inf
from numpy import abs as np_abs

class genpareto2_gen(rv_continuous):
    pass
genpareto2 = genpareto2_gen(a=0.0, name='genpareto', longname='A generalized Pareto', shapes='c')
shape, loc, scale = (0.5, 0, 1)
rv = np.arange(5)
quant = [0.01, 0.1, 0.5, 0.9, 0.99]
for method, x in [('pdf', rv), ('cdf', rv), ('sf', rv), ('ppf', quant), ('isf', quant)]:
    print(getattr(genpareto2, method)(x, shape, loc, scale))
    print(getattr(stats.genpareto, method)(x, -shape, loc, scale))
print(genpareto2.stats(shape, loc, scale, moments='mvsk'))
print(stats.genpareto.stats(-shape, loc, scale, moments='mvsk'))
print(genpareto2.entropy(shape, loc, scale))
print(stats.genpareto.entropy(-shape, loc, scale))

def paramstopot(thresh, shape, scale):
    """transform shape scale for peak over threshold

    y = x-u|x>u ~ GPD(k, sigma-k*u) if x ~ GPD(k, sigma)
    notation of de Zea Bermudez, Kotz
    k, sigma is shape, scale
    """
    new_scale = scale - shape * thresh
    return shape, new_scale

def meanexcess(thresh, shape, scale):
    """mean excess function of genpareto

    assert are inequality conditions in de Zea Bermudez, Kotz
    """
    assert shape < 1, "Shape parameter must be less than 1"
    assert scale > 0, "Scale parameter must be positive"
    assert thresh >= 0, "Threshold must be non-negative"
    
    return (scale + shape * thresh) / (1 - shape)
print(meanexcess(5, -0.5, 10))
print(meanexcess(5, -2, 10))
data = genpareto2.rvs(-0.75, scale=5, size=1000)
tmp = meanexcess_plot(data, params=(-0.75, 5), plot=1)
print(tmp[1][-20:])
print(tmp[0][-20:])
ds, me, mc = meanexcess_emp(1.0 * np.arange(1, 10))
print(ds)
print(me)
print(mc)
print(meanexcess_dist(stats.norm, lb=0.5))
print(meanexcess_dist(stats.norm, lb=[-np.inf, -0.5, 0, 0.5]))
rvs = stats.norm.rvs(size=100000)
rvs = rvs - rvs.mean()
print(rvs.mean(), rvs[rvs > -0.5].mean(), rvs[rvs > 0].mean(), rvs[rvs > 0.5].mean())
'\n[ 1.   0.5  0.   0.   0. ]\n[ 1.   0.5  0.   0.   0. ]\n[ 0.    0.75  1.    1.    1.  ]\n[ 0.    0.75  1.    1.    1.  ]\n[ 1.    0.25  0.    0.    0.  ]\n[ 1.    0.25  0.    0.    0.  ]\n[ 0.01002513  0.1026334   0.58578644  1.36754447  1.8       ]\n[ 0.01002513  0.1026334   0.58578644  1.36754447  1.8       ]\n[ 1.8         1.36754447  0.58578644  0.1026334   0.01002513]\n[ 1.8         1.36754447  0.58578644  0.1026334   0.01002513]\n(array(0.66666666666666674), array(0.22222222222222243), array(0.56568542494923058), array(-0.60000000000032916))\n(array(0.66666666666666674), array(0.22222222222222243), array(0.56568542494923058), array(-0.60000000000032916))\n0.5\n0.5\n25.0\nshape > -1 does not hold\n-20\n[  41.4980671    42.83145298   44.24197578   45.81622844   47.57145212\n   49.52692287   51.70553275   54.0830766    56.61358997   59.53409167\n   62.8970042    66.73494156   71.04227973   76.24015612   82.71835988\n   89.79611663   99.4252195   106.2372462    94.83432424    0.        ]\n[  15.79736355   16.16373531   17.44204268   17.47968055   17.73264951\n   18.23939099   19.02638455   20.79746264   23.7169161    24.48807136\n   25.90496638   28.35556795   32.27623618   34.65714495   37.37093362\n   47.32957609   51.27970515   78.98913941  129.04309012  189.66864848]\n>>> np.arange(10)\narray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])\n>>> meanexcess_emp(np.arange(10))\n(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([4, 4, 5, 5, 5, 6, 6, 5, 4, 0]), array([9, 8, 8, 7, 7, 6, 6, 5, 5, 4]))\n>>> meanexcess_emp(1*np.arange(10))\n(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([4, 4, 5, 5, 5, 6, 6, 5, 4, 0]), array([9, 8, 8, 7, 7, 6, 6, 5, 5, 4]))\n>>> meanexcess_emp(1.*np.arange(10))\n(array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]), array([ 4.5       ,  4.88888889,  5.25      ,  5.57142857,  5.83333333,\n        6.        ,  6.        ,  5.66666667,  4.5       ,  0.        ]), array([ 9. ,  8.5,  8. ,  7.5,  7. ,  6.5,  6. ,  5.5,  5. ,  4.5]))\n>>> meanexcess_emp(0.5**np.arange(10))\n(array([ 0.00195313,  0.00390625,  0.0078125 ,  0.015625  ,  0.03125   ,\n        0.0625    ,  0.125     ,  0.25      ,  0.5       ,  1.        ]), array([ 0.19960938,  0.22135417,  0.24804688,  0.28125   ,  0.32291667,\n        0.375     ,  0.4375    ,  0.5       ,  0.5       ,  0.        ]), array([ 1.        ,  0.75      ,  0.58333333,  0.46875   ,  0.3875    ,\n        0.328125  ,  0.28348214,  0.24902344,  0.22178819,  0.19980469]))\n>>> meanexcess_emp(np.arange(10)**0.5)\n(array([ 0.        ,  1.        ,  1.41421356,  1.73205081,  2.        ,\n        2.23606798,  2.44948974,  2.64575131,  2.82842712,  3.        ]), array([ 1.93060005,  2.03400006,  2.11147337,  2.16567659,  2.19328936,\n        2.18473364,  2.11854461,  1.94280904,  1.5       ,  0.        ]), array([ 3.        ,  2.91421356,  2.82472615,  2.73091704,  2.63194723,\n        2.52662269,  2.41311242,  2.28825007,  2.14511117,  1.93060005]))\n>>> meanexcess_emp(np.arange(10)**-2)\n(array([-2147483648,           0,           0,           0,           0,\n                 0,           0,           0,           0,           1]), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), array([         1,          0,          0,          0,          0,\n                0,          0,          0,          0, -214748365]))\n>>> meanexcess_emp(np.arange(10)**(-0.5))\n(array([ 0.33333333,  0.35355339,  0.37796447,  0.40824829,  0.4472136 ,\n        0.5       ,  0.57735027,  0.70710678,  1.        ,         Inf]), array([ Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  NaN]), array([ Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf]))\n>>> np.arange(10)**(-0.5)\narray([        Inf,  1.        ,  0.70710678,  0.57735027,  0.5       ,\n        0.4472136 ,  0.40824829,  0.37796447,  0.35355339,  0.33333333])\n>>> meanexcess_emp(np.arange(1,10)**(-0.5))\n(array([ 0.33333333,  0.35355339,  0.37796447,  0.40824829,  0.4472136 ,\n        0.5       ,  0.57735027,  0.70710678,  1.        ]), array([ 0.4857152 ,  0.50223543,  0.51998842,  0.53861177,  0.55689141,\n        0.57111426,  0.56903559,  0.5       ,  0.        ]), array([ 1.        ,  0.85355339,  0.76148568,  0.69611426,  0.64633413,\n        0.60665316,  0.57398334,  0.5464296 ,  0.52275224]))\n>>> meanexcess_emp(np.arange(1,10))\n(array([1, 2, 3, 4, 5, 6, 7, 8, 9]), array([4, 5, 5, 5, 6, 6, 5, 4, 0]), array([9, 8, 8, 7, 7, 6, 6, 5, 5]))\n>>> meanexcess_emp(1.*np.arange(1,10))\n(array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]), array([ 4.88888889,  5.25      ,  5.57142857,  5.83333333,  6.        ,\n        6.        ,  5.66666667,  4.5       ,  0.        ]), array([ 9. ,  8.5,  8. ,  7.5,  7. ,  6.5,  6. ,  5.5,  5. ]))\n>>> datasorted = np.sort(1.*np.arange(1,10))\n>>> (datasorted[::-1].cumsum()-datasorted[::-1])\narray([  0.,   9.,  17.,  24.,  30.,  35.,  39.,  42.,  44.])\n>>> datasorted[::-1].cumsum()\narray([  9.,  17.,  24.,  30.,  35.,  39.,  42.,  44.,  45.])\n>>> datasorted[::-1]\narray([ 9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.])\n>>>\n'
