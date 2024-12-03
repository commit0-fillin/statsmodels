import numpy as np
import scipy.stats as stats
_future_warn = 'Passing `None` as the seed currently return the NumPy singleton RandomState\n(np.random.mtrand._rand). After release 0.13 this will change to using the\ndefault generator provided by NumPy (np.random.default_rng()). If you need\nreproducible draws, you should pass a seeded np.random.Generator, e.g.,\n\nimport numpy as np\nseed = 32839283923801\nrng = np.random.default_rng(seed)"\n'

def check_random_state(seed=None):
    """
    Turn `seed` into a random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[ints], `numpy.random.Generator`,
            `numpy.random.RandomState`, `scipy.stats.qmc.QMCEngine`}, optional

        If `seed` is None fresh, unpredictable entropy will be pulled
        from the OS and `numpy.random.Generator` is used.
        If `seed` is an int or ``array_like[ints]``, a new ``Generator``
        instance is used, seeded with `seed`.
        If `seed` is already a ``Generator``, ``RandomState`` or
        `scipy.stats.qmc.QMCEngine` instance then
        that instance is used.

        `scipy.stats.qmc.QMCEngine` requires SciPy >=1.7. It also means
        that the generator only have the method ``random``.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`,
            `scipy.stats.qmc.QMCEngine`}

        Random number generator.
    """
    if seed is None:
        return np.random.default_rng()
    elif isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    elif isinstance(seed, (list, tuple, np.ndarray)):
        return np.random.default_rng(seed)
    elif isinstance(seed, (np.random.Generator, np.random.RandomState)):
        return seed
    elif hasattr(seed, 'random'):
        # Check if it's a QMCEngine instance
        if hasattr(seed, 'random_base2'):
            return seed
    else:
        raise ValueError(f"Invalid seed: {seed}")
