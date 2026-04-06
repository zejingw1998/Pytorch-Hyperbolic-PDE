"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

import numpy as np

from nm_lib.nm_ex.nm_lib_ex_1 import deriv_fwd


def step_adv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    a: float,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    **kwargs,
) -> np.ndarray:
    r"""
    Right-hand side of Burger's eq. where a can be a constant or a function that
    depends on xx.

    Requires
    ----------
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default, clf_cut=0.98.
    ddx : `lambda function`
        Allows the selection of the type of spatial derivative.
        By default lambda x,y: deriv_fwd(x, y)

    Returns
    -------
    `array`
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., a \frac{\partial u}{\partial x}
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    rhs = np.zeros_like(hh, dtype=float)

    dx = xx[1] - xx[0]

    # forward derivative on interior points
    if np.isscalar(a):
        rhs[:-1] = -a * (hh[1:] - hh[:-1]) / dx
        rhs[-1] = -a * (hh[0] - hh[-1]) / dx
    else:
        a = np.asarray(a, dtype=float)
        rhs[:-1] = -a[:-1] * (hh[1:] - hh[:-1]) / dx
        rhs[-1] = -a[-1] * (hh[0] - hh[-1]) / dx

    return rhs


def cfl_adv_burger(a: float, x: np.ndarray) -> float:
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and
    Lewy's condition for the advective term in Burger's equation.

    Parameters
    ----------
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    x : `array`
        Spatial axis.

    Returns
    -------
    `float`
        min(dx/|a|)
    """
    x = np.asarray(x, dtype=float)

    dx = x[1] - x[0]

    if np.isscalar(a):
        if np.isclose(a, 0.0):
            return np.inf
        return dx / np.abs(a)
    else:
        a = np.asarray(a, dtype=float)
        aa = np.abs(a)
        aa = aa[aa > 0.0]

        if aa.size == 0:
            return np.inf

        return np.min(dx / aa)


def evolv_adv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: float,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a fix constant or array.
    Requires
    ----------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    nt : `int`
        Number of time iterations.
    a : `float` or `array`
        Either constant or array, which multiplies the right-hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_fwd(x, y).
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default 'wrap'.
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels
        will need to be updated with the boundary information.
        By default [0,1].

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dt = cfl_cut * cfl_adv_burger(a, xx)

    t = np.arange(nt + 1, dtype=float) * dt
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)

    unnt[0] = hh.copy()

    for n in range(nt):
        rhs = step_adv_burgers(xx, unnt[n], a=a, cfl_cut=cfl_cut, ddx=ddx)

        unnt[n + 1] = unnt[n] + dt * rhs

        if bnd_type == "wrap":
            # keep periodicity consistent
            unnt[n + 1, -1] = unnt[n + 1, 0]

    return t, unnt