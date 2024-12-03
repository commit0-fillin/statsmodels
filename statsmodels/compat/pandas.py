from typing import Optional
import numpy as np
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import Appender, Substitution, cache_readonly, deprecate_kwarg
__all__ = ['assert_frame_equal', 'assert_index_equal', 'assert_series_equal', 'data_klasses', 'frequencies', 'is_numeric_dtype', 'testing', 'cache_readonly', 'deprecate_kwarg', 'Appender', 'Substitution', 'is_int_index', 'is_float_index', 'make_dataframe', 'to_numpy', 'PD_LT_1_0_0', 'get_cached_func', 'get_cached_doc', 'call_cached_func', 'PD_LT_1_4', 'PD_LT_2', 'MONTH_END', 'QUARTER_END', 'YEAR_END', 'FUTURE_STACK']
version = parse(pd.__version__)
PD_LT_2_2_0 = version < Version('2.1.99')
PD_LT_2_1_0 = version < Version('2.0.99')
PD_LT_1_0_0 = version < Version('0.99.0')
PD_LT_1_4 = version < Version('1.3.99')
PD_LT_2 = version < Version('1.9.99')
try:
    from pandas.api.types import is_numeric_dtype
except ImportError:
    from pandas.core.common import is_numeric_dtype
try:
    from pandas.tseries import offsets as frequencies
except ImportError:
    from pandas.tseries import frequencies
data_klasses = (pd.Series, pd.DataFrame)
try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing
assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal

def is_int_index(index: pd.Index) -> bool:
    """
    Check if an index is integral

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if is an index with a standard integral type
    """
    return index.dtype.kind in 'iu'

def is_float_index(index: pd.Index) -> bool:
    """
    Check if an index is floating

    Parameters
    ----------
    index : pd.Index
        Any numeric index

    Returns
    -------
    bool
        True if an index with a standard numpy floating dtype
    """
    return index.dtype.kind == 'f'
try:
    from pandas._testing import makeDataFrame as make_dataframe
except ImportError:
    import string

    def rands_array(nchars, size, dtype='O'):
        """
        Generate an array of byte strings.
        """
        chars = np.array(list(string.ascii_letters + string.digits))
        retval = (chars[np.random.randint(0, len(chars), size=(size, nchars))]
                  .view((str, nchars))
                  .astype(dtype))
        return retval

    def make_dataframe():
        """
        Simple version of pandas._testing.makeDataFrame
        """
        index = pd.date_range('1/1/2000', periods=100)
        data = {
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100),
            'D': np.random.randn(100)
        }
        return pd.DataFrame(data, index=index)

def to_numpy(po: pd.DataFrame) -> np.ndarray:
    """
    Workaround legacy pandas lacking to_numpy

    Parameters
    ----------
    po : Pandas object

    Returns
    -------
    ndarray
        A numpy array
    """
    if hasattr(po, 'to_numpy'):
        return po.to_numpy()
    else:
        return po.values
MONTH_END = 'M' if PD_LT_2_2_0 else 'ME'
QUARTER_END = 'Q' if PD_LT_2_2_0 else 'QE'
YEAR_END = 'Y' if PD_LT_2_2_0 else 'YE'
FUTURE_STACK = {} if PD_LT_2_1_0 else {'future_stack': True}
