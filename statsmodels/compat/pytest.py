from __future__ import annotations
from typing import Tuple, Type, Union
import warnings
from _pytest.recwarn import WarningsChecker
from pytest import warns
__all__ = ['pytest_warns']

class NoWarningsChecker:

    def __init__(self):
        self.cw = warnings.catch_warnings(record=True)
        self.rec = []

    def __enter__(self):
        self.rec = self.cw.__enter__()

    def __exit__(self, type, value, traceback):
        if self.rec:
            warnings = [w.category.__name__ for w in self.rec]
            joined = '\\n'.join(warnings)
            raise AssertionError(f'Function is marked as not warning but the following warnings were found: \n{joined}')

def pytest_warns(warning: Type[Warning] | Tuple[Type[Warning], ...] | None) -> Union[WarningsChecker, NoWarningsChecker]:
    """
    A context manager for asserting warnings in a block of code.

    Parameters
    ----------
    warning : {None, Warning, Tuple[Warning]}
        None if no warning is produced, or a single or multiple Warnings

    Returns
    -------
    cm : Union[WarningsChecker, NoWarningsChecker]
        A context manager that can be used to check for warnings.

    """
    if warning is None:
        return NoWarningsChecker()
    else:
        return warns(warning)
