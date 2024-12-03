"""
Handle file opening for read/write
"""
from pathlib import Path
from numpy.lib._iotools import _is_string_like

class EmptyContextManager:
    """
    This class is needed to allow file-like object to be used as
    context manager, but without getting closed.
    """

    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        """When entering, return the embedded object"""
        return self._obj

    def __exit__(self, *args):
        """Do not hide anything"""
        return False

    def __getattr__(self, name):
        return getattr(self._obj, name)

def get_file_obj(fname, mode='r', encoding=None):
    """
    Light wrapper to handle strings, path objects and let files (anything else)
    pass through.

    It also handle '.gz' files.

    Parameters
    ----------
    fname : str, path object or file-like object
        File to open / forward
    mode : str
        Argument passed to the 'open' or 'gzip.open' function
    encoding : str
        For Python 3 only, specify the encoding of the file

    Returns
    -------
    A file-like object that is always a context-manager. If the `fname` was
    already a file-like object, the returned context manager *will not
    close the file*.
    """
    import gzip
    
    if _is_string_like(fname):
        fname = Path(fname)
    
    if isinstance(fname, Path):
        if fname.suffix == '.gz':
            return gzip.open(fname, mode=mode)
        else:
            return open(fname, mode=mode, encoding=encoding)
    elif hasattr(fname, 'read') or hasattr(fname, 'write'):
        return EmptyContextManager(fname)
    else:
        raise ValueError(f"File object {fname} is not recognized")
