"""
Created on Fri Jul 02 10:25:17 2021

Original author: Juan Martinez Sykora
Later modified by Zejing Wang
"""

import numpy as np

from nm_lib.nm_ex.nm_lib_ex_1 import deriv_fwd
from nm_lib.nm_ex.nm_lib_ex_2 import step_adv_burgers


def _maxabs(c):
    c = np.asarray(c, dtype=float)
    return float(np.max(np.abs(c)))


def _split_dt(xx, a, b=None, cfl_cut=0.98):
    xx = np.asarray(xx, dtype=float)
    dx = xx[1] - xx[0]

    if b is None:
        cmax = _maxabs(a)
    else:
        csum = np.asarray(a, dtype=float) + np.asarray(b, dtype=float)
        cmax = max(_maxabs(a), _maxabs(b), _maxabs(csum))

    if cmax < 1e-14:
        return cfl_cut * dx

    return cfl_cut * dx / cmax


def _lax_step(xx, u, c, dt):
    xx = np.asarray(xx, dtype=float)
    u = np.asarray(u, dtype=float)
    c = np.asarray(c, dtype=float)

    dx = xx[1] - xx[0]
    up = np.roll(u, -1)
    um = np.roll(u, 1)
    # one Lax step for u_t = -c u_x with periodic BC
    lam = c * dt / (2.0 * dx)
    return 0.5 * (up + um) - lam * (up - um)


def _deriv_cent_periodic(xx, u):
    xx = np.asarray(xx, dtype=float)
    u = np.asarray(u, dtype=float)

    dx = xx[1] - xx[0]
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def _adv_rhs(xx, u, c):
    # right-hand side of u_t = -c u_x
    return -np.asarray(c, dtype=float) * _deriv_cent_periodic(xx, u)

def ops_Lax_LL_Add(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    b: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Additive operator splitting with Lax for both split operators.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dt = _split_dt(xx, a, b, cfl_cut)

    t = np.arange(nt + 1, dtype=float) * dt
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        u = unnt[n]

        ua = _lax_step(xx, u, a, dt)
        ub = _lax_step(xx, u, b, dt)

        unnt[n + 1] = ua + ub - u

    return t, unnt


def ops_Lax_LL_Lie(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    b: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Lie-Trotter splitting with Lax for both split operators.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dt = _split_dt(xx, a, b, cfl_cut)

    t = np.arange(nt + 1, dtype=float) * dt
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        u = unnt[n]

        u1 = _lax_step(xx, u, a, dt)
        u2 = _lax_step(xx, u1, b, dt)

        unnt[n + 1] = u2

    return t, unnt


def ops_Lax_LL_Strange(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    b: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Strang splitting with Lax for both split operators.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dt = _split_dt(xx, a, b, cfl_cut)

    t = np.arange(nt + 1, dtype=float) * dt
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        u = unnt[n]

        u1 = _lax_step(xx, u, a, dt / 2.0)
        u2 = _lax_step(xx, u1, b, dt)
        u3 = _lax_step(xx, u2, a, dt / 2.0)

        unnt[n + 1] = u3

    return t, unnt

def osp_Lax_LH_Strange(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    b: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Strang splitting with Lax for A and Hyman for B.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dt = _split_dt(xx, a, b, cfl_cut)

    t = np.arange(nt + 1, dtype=float) * dt
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    # Hyman needs memory from previous steps
    # this means the B-step is not a pure one-step operator
    fold = None
    dtold = None

    for n in range(nt):
        u = unnt[n]

        # A half-step with Lax
        u1 = _lax_step(xx, u, a, dt / 2.0)

        # B full-step with Hyman
        u2, fold, dtold = hyman(
            xx,
            u1,
            dth=dt,
            a=b,
            fold=fold,
            dtold=dtold,
            cfl_cut=cfl_cut,
            ddx=ddx,
            bnd_type=bnd_type,
            bnd_limits=bnd_limits,
        )

        # A half-step with Lax
        u3 = _lax_step(xx, u2, a, dt / 2.0)

        unnt[n + 1] = u3

    return t, unnt


def hyman(
    xx: np.ndarray,
    f: np.ndarray,
    dth: float,
    a: np.ndarray,
    fold: np.ndarray = None,
    dtold: float | None = None,
    cfl_cut: float = 0.8,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
):
    """
    Hyman predictor-corrector method.

    Important:
    This method is not a pure one-step map, because it needs the history
    variables fold and dtold.
    """
    xx = np.asarray(xx, dtype=float)
    f = np.asarray(f, dtype=float)

    if fold is not None:
        fold = np.asarray(fold, dtype=float)

    # df/dt = -a f_x
    dfdt = _adv_rhs(xx, f, a)

    # first call: initialize the history
    if fold is None:
        fold = np.copy(f)

        # first Hyman-style startup step
        f = 0.5 * (np.roll(f, 1) + np.roll(f, -1)) + dth * dfdt
        dtold = dth

        return f, fold, dtold

    # later calls: use the stored history
    ratio = dth / dtold
    a1 = ratio**2
    b1 = dth * (1.0 + ratio)
    a2 = 2.0 * (1.0 + ratio) / (2.0 + 3.0 * ratio)
    b2 = dth * (1.0 + ratio**2) / (2.0 + 3.0 * ratio)
    c2 = dth * (1.0 + ratio) / (2.0 + 3.0 * ratio)

    f, fold, fsav = hyman_pred(f, fold, dfdt, a1, b1, a2, b2)

    # recompute rhs at the predicted state
    dfdt_new = _adv_rhs(xx, f, a)

    f = hyman_corr(fsav, dfdt_new, c2)
    dtold = dth

    return f, fold, dtold


def hyman_corr(fsav: np.ndarray, dfdt: np.ndarray, c2: float) -> np.ndarray:
    """
    Hyman Corrector step

    Parameters
    ----------
    fsav : `array`
        A function that depends on xx from the interpolated step.
    dfdt : `array`
        A function that depends on xx. The right-hand side of the time derivative.
    c2: `float`
        Coefficient.

    Returns
    -------
    corrector : `array`
        A function of the Hyman corrector step
    """
    return fsav + c2 * dfdt


def hyman_pred(
    f: np.ndarray,
    fold: np.ndarray,
    dfdt: np.ndarray,
    a1: float,
    b1: float,
    a2: float,
    b2: float,
):
    """
    Hyman Predictor step

    Parameters
    ----------
    f : `array`
        A function that depends on xx.
    fold : `array`
        A function that depends on xx from the previous step.
    dfdt : `array`
        A function that depends on xx. The right-hand side of the time derivative.
    a1: `float`
        Coefficient.
    b1: `float`
        Coefficient.
    a2: `float`
        Coefficient.
    b2: `float`
        Coefficient.

    Returns
    -------
    f : `array`
        A function that depends on xx.
    fold : `array`
        A function that depends on xx from the previous step.
    fsav : `array`
        A function that depends on xx from the interpolated step.
    """
    fsav = np.copy(f)
    tempvar = f + a1 * (fold - f) + b1 * dfdt
    fold = np.copy(fsav)
    fsav = tempvar + a2 * (fsav - tempvar) + b2 * dfdt
    f = tempvar

    return f, fold, fsav
