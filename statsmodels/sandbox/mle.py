"""What's the origin of this file? It is not ours.
Does not run because of missing mtx files, now included

changes: JP corrections to imports so it runs, comment out print
"""
import numpy as np
from numpy import dot, outer, random
from scipy import io, linalg, optimize
from scipy.sparse import eye as speye
import matplotlib.pyplot as plt

def Rp(v):
    """ Gradient """
    global A, B
    Av = A.dot(v)
    Bv = B.dot(v)
    vBv = v.dot(Bv)
    return 2 * (Av * vBv - Bv * v.dot(Av)) / (vBv ** 2)

def Rpp(v):
    """ Hessian """
    global A, B
    Av = A.dot(v)
    Bv = B.dot(v)
    vAv = v.dot(Av)
    vBv = v.dot(Bv)
    return 2 * (A * vBv + B * vAv - outer(Av, Bv) - outer(Bv, Av)) / (vBv ** 2) - \
           4 * (vAv * vBv) * outer(Bv, Bv) / (vBv ** 4)
A = io.mmread('nos4.mtx')
n = A.shape[0]
B = speye(n, n)
random.seed(1)
v_0 = random.rand(n)
print('try fmin_bfgs')
full_output = 1
data = []
v, fopt, gopt, Hopt, func_calls, grad_calls, warnflag, allvecs = optimize.fmin_bfgs(R, v_0, fprime=Rp, full_output=full_output, retall=1)
if warnflag == 0:
    plt.semilogy(np.arange(0, len(data)), data)
    print('Rayleigh quotient BFGS', R(v))
print('fmin_bfgs OK')
print('try fmin_ncg')
data = []
v, fopt, fcalls, gcalls, hcalls, warnflag, allvecs = optimize.fmin_ncg(R, v_0, fprime=Rp, fhess=Rpp, full_output=full_output, retall=1)
if warnflag == 0:
    plt.figure()
    plt.semilogy(np.arange(0, len(data)), data)
    print('Rayleigh quotient NCG', R(v))
