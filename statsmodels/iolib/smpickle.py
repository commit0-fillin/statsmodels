"""Helper files for pickling"""
from statsmodels.iolib.openfile import get_file_obj

def save_pickle(obj, fname):
    """
    Save the object to file via pickling.

    Parameters
    ----------
    fname : {str, pathlib.Path}
        Filename to pickle to
    """
    pass

def load_pickle(fname):
    """
    Load a previously saved object

    .. warning::

       Loading pickled models is not secure against erroneous or maliciously
       constructed data. Never unpickle data received from an untrusted or
       unauthenticated source.

    Parameters
    ----------
    fname : {str, pathlib.Path}
        Filename to unpickle

    Notes
    -----
    This method can be used to load *both* models and results.
    """
    pass