"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

Later modified by Zejing Wang
"""

import numpy as np

from nm_lib.nm_ex.nm_lib_ex_1 import deriv_fwd




def deriv_bck(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dh = np.full_like(hh, np.nan, dtype=float)
    dh[1:] = (hh[1:] - hh[:-1]) / (xx[1:] - xx[:-1])
    return dh


def deriv_cent(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dh = np.full_like(hh, np.nan, dtype=float)
    dh[1:-1] = (hh[2:] - hh[:-2]) / (xx[2:] - xx[:-2])
    return dh

def step_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    **kwargs,
):
    r"""
    Right-hand side of Burger's eq. where a is u, i.e., hh.

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function`
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_fwd(x, y)

    Returns
    -------
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x}
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dx = xx[1] - xx[0]
    bnd_type = kwargs.get("bnd_type", "wrap")

    if bnd_type == "wrap":
        # periodic extension
        xx_ext = np.empty(len(xx) + 2, dtype=float)
        hh_ext = np.empty(len(hh) + 2, dtype=float)

        xx_ext[1:-1] = xx
        xx_ext[0] = xx[0] - dx
        xx_ext[-1] = xx[-1] + dx

        hh_ext[1:-1] = hh
        hh_ext[0] = hh[-1]
        hh_ext[-1] = hh[0]

        dh_ext = ddx(xx_ext, hh_ext)
        dh = dh_ext[1:-1]
    else:
        dh = ddx(xx, hh)

    rhs = -hh * dh
    return rhs


def evolv_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being u.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of time iterations.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98.
    ddx : `lambda function`
        Allows to change the space derivative function.
    bnd_type : `string`
        It allows one to select the type of boundaries.
        By default, 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dx = xx[1] - xx[0]

    t = np.zeros(nt + 1, dtype=float)
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        umax = np.max(np.abs(unnt[n]))
        if umax < 1e-14:
            dt = cfl_cut * dx
        else:
            dt = cfl_cut * dx / umax

        rhs = step_uadv_burgers(
            xx,
            unnt[n],
            cfl_cut=cfl_cut,
            ddx=ddx,
            bnd_type=bnd_type,
        )

        unnt[n + 1] = unnt[n] + dt * rhs
        t[n + 1] = t[n] + dt

    return t, unnt
