"""
Compatibility tools for differences between Python 2 and 3
"""
import sys
from typing import TYPE_CHECKING
PY37 = sys.version_info[:2] == (3, 7)
asunicode = lambda x, _: str(x)
__all__ = ['asunicode', 'asstr', 'asbytes', 'Literal', 'lmap', 'lzip', 'lrange', 'lfilter', 'with_metaclass']

def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    class metaclass(type):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})
if sys.version_info >= (3, 8):
    from typing import Literal
elif TYPE_CHECKING:
    from typing_extensions import Literal
else:
    from typing import Any as Literal
