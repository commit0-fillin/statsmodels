import numpy as np

def rainbow(n):
    """
    Returns a list of colors sampled at equal intervals over the spectrum.

    Parameters
    ----------
    n : int
        The number of colors to return

    Returns
    -------
    R : (n,3) array
        An of rows of RGB color values

    Notes
    -----
    Converts from HSV coordinates (0, 1, 1) to (1, 1, 1) to RGB. Based on
    the Sage function of the same name.
    """
    if n == 0:
        return np.empty((0, 3))
    
    hue = np.linspace(0, 1, n+1)[:-1]
    saturation = np.ones_like(hue)
    value = np.ones_like(hue)
    
    c = value * saturation
    x = c * (1 - np.abs((hue * 6) % 2 - 1))
    m = value - c
    
    hue_idx = (hue * 6).astype(int)
    
    r = np.select([hue_idx < 1, hue_idx < 2, hue_idx < 3, hue_idx < 4, hue_idx < 5], 
                  [c, x, 0, 0, x], default=c)
    g = np.select([hue_idx < 1, hue_idx < 2, hue_idx < 3, hue_idx < 4, hue_idx < 5], 
                  [x, c, c, x, 0], default=0)
    b = np.select([hue_idx < 1, hue_idx < 2, hue_idx < 3, hue_idx < 4, hue_idx < 5], 
                  [0, 0, x, c, c], default=x)
    
    return np.column_stack((r + m, g + m, b + m))
