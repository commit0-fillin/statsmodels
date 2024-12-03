import numpy as np

def forrt(X, m=None):
    """
    RFFT with order like Munro (1976) FORTT routine.
    """
    n = len(X)
    if m is None:
        m = n
    Y = np.fft.rfft(X, n=m)
    return Y[:(m // 2 + 1)]

def revrt(X, m=None):
    """
    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.
    """
    n = len(X)
    if m is None:
        m = (n - 1) * 2
    Y = np.zeros(m // 2 + 1, dtype=complex)
    Y[:n] = X
    return np.fft.irfft(Y, n=m)

def silverman_transform(bw, M, RANGE):
    """
    FFT of Gaussian kernel following to Silverman AS 176.

    Notes
    -----
    Underflow is intentional as a dampener.
    """
    r = np.arange(M)
    lamda = 2 * np.pi / RANGE
    return np.exp(-0.5 * (bw * lamda * r) ** 2)

def counts(x, v):
    """
    Counts the number of elements of x that fall within the grid points v

    Notes
    -----
    Using np.digitize and np.bincount
    """
    indices = np.digitize(x, v)
    return np.bincount(indices, minlength=len(v)+1)[1:-1]
