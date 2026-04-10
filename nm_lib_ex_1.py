"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

Later modified by Zejing Wang
"""


import numpy as np

def deriv_fwd(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    """
    Returns the forward derivative of hh array with respect to xx array.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.

    Returns
    -------
    `array`
        The forward derivative of hh respect to xx. The last
        grid point is ill (or missing) calculated.
    """
    dh = np.full_like(hh, np.nan, dtype=float)
    dh[:-1] = (hh[1:] - hh[:-1]) / (xx[1:] - xx[:-1])
    return dh


def order_conv(hh: np.ndarray, hh2: np.ndarray, hh4: np.ndarray, **kwargs) -> np.ndarray:
    """
    Computes the order of convergence of a derivative function

    Parameters
    ----------
    hh : `array`
        A function that depends on xx.
    hh2 : `array`
        A function that depends on xx but with twice the number of grid points than hh.
    hh4 : `array`
        A function that depends on xx but with twice the number of grid points than hh2.
    Returns
    -------
    `array`
        The order of convergence.
    """
    hh2c = hh2[::2]
    hh4c = hh4[::4]

    n = min(len(hh), len(hh2c), len(hh4c))
    hh = hh[:n]
    hh2c = hh2c[:n]
    hh4c = hh4c[:n]

    e1 = np.abs(hh - hh2c)
    e2 = np.abs(hh2c - hh4c)

    order = np.full(n, np.nan, dtype=float)
    mask = (e1 > 0) & (e2 > 0)
    order[mask] = np.log2(e1[mask] / e2[mask])

    return order

def deriv_4tho(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    """
    Returns the 4th order derivative of hh with respect to xx.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.

    Returns
    -------
    `array`
        The centered 4th order derivative of hh with respect to xx.
        The last and first two grid points are ill-calculated.
    """
    dx = xx[1] - xx[0]

    dh = np.full_like(hh, np.nan, dtype=float)
    dh[2:-2] = (hh[:-4] - 8.0 * hh[1:-3] + 8.0 * hh[3:-1] - hh[4:]) / (12.0 * dx)

    return dh
