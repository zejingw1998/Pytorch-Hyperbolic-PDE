"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

import numpy as np

from nm_lib.nm_ex.nm_lib_ex_1 import deriv_fwd


def evolv_Lax_adv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a fix constant or array.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dx = xx[1] - xx[0]

    if np.isscalar(a):
        amax = abs(a)
    else:
        a = np.asarray(a, dtype=float)
        amax = np.max(np.abs(a))

    if amax < 1e-14:
        dt = cfl_cut * dx
    else:
        dt = cfl_cut * dx / amax

    t = np.arange(nt + 1, dtype=float) * dt
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        u = unnt[n]
        u_new = np.empty_like(u)

        if bnd_type == "wrap":
            for j in range(len(u)):
                jp = (j + 1) % len(u)
                jm = (j - 1) % len(u)

                if np.isscalar(a):
                    lam = a * dt / (2.0 * dx)
                else:
                    lam = a[j] * dt / (2.0 * dx)

                u_new[j] = (u[jm] + u[j] + u[jp]) / 3.0 - lam * (u[jp] - u[jm])
        else:
            u_new[:] = u[:]
            for j in range(1, len(u) - 1):
                if np.isscalar(a):
                    lam = a * dt / (2.0 * dx)
                else:
                    lam = a[j] * dt / (2.0 * dx)

                u_new[j] = (u[j - 1] + u[j] + u[j + 1]) / 3.0 - lam * (u[j + 1] - u[j - 1])

        unnt[n + 1] = u_new

    return t, unnt


def evolv_Lax_uadv_burgers(
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
    Advance nt time-steps in time the burger eq for a being u using the Lax method.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dx = xx[1] - xx[0]

    t = np.zeros(nt + 1, dtype=float)
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        u = unnt[n]
        umax = np.max(np.abs(u))

        if umax < 1e-14:
            dt = cfl_cut * dx
        else:
            dt = cfl_cut * dx / umax

        u_new = np.empty_like(u)

        if bnd_type == "wrap":
            for j in range(len(u)):
                jp = (j + 1) % len(u)
                jm = (j - 1) % len(u)

                lam = dt / (2.0 * dx)

                u_new[j] = (u[jm] + u[j] + u[jp]) / 3.0 - (u[j] * lam) * (u[jp] - u[jm])
        else:
            u_new[:] = u[:]
            for j in range(1, len(u) - 1):
                lam = dt / (2.0 * dx)
                u_new[j] = (u[j - 1] + u[j] + u[j + 1]) / 3.0 - (u[j] * lam) * (u[j + 1] - u[j - 1])

        unnt[n + 1] = u_new
        t[n + 1] = t[n] + dt

    return t, unnt


def Rie_flux(
    hh: np.ndarray,
):
    """
     Flux from the burgers eq.
    """
    hh = np.asarray(hh, dtype=float)
    flux = 0.5 * hh**2
    return flux

def Rie_va(
    uL: np.ndarray,
    uR: np.ndarray,
):
    """
     absolute propagating speed (va), uses Rie_flux
    """
    uL = np.asarray(uL, dtype=float)
    uR = np.asarray(uR, dtype=float)

    va = np.maximum(np.abs(uL), np.abs(uR))
    return va


def Rie_interface_flux(
    uL: np.ndarray,
    uR: np.ndarray,
    va: np.ndarray,
):
    """
     Interface Rusanov flux
    """
    uL = np.asarray(uL, dtype=float)
    uR = np.asarray(uR, dtype=float)
    va = np.asarray(va, dtype=float)

    fL = Rie_flux(uL)
    fR = Rie_flux(uR)

    Fstar = 0.5 * (fL + fR) - 0.5 * va * (uR - uL)
    return Fstar


def evolv_Rie_uadv_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Riemann (Rusanov) method.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dx = xx[1] - xx[0]

    t = np.zeros(nt + 1, dtype=float)
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        u = unnt[n]
        umax = np.max(np.abs(u))

        if umax < 1e-14:
            dt = cfl_cut * dx
        else:
            dt = cfl_cut * dx / umax

        if bnd_type == "wrap":
            uL = u
            uR = np.roll(u, -1)

            va = Rie_va(uL, uR)
            F = Rie_interface_flux(uL, uR, va)

            # finite-volume update
            unnt[n + 1] = u - (dt / dx) * (F - np.roll(F, 1))
        else:
            unew = u.copy()

            uL = u[:-1]
            uR = u[1:]
            va = Rie_va(uL, uR)
            F = Rie_interface_flux(uL, uR, va)

            unew[1:-1] = u[1:-1] - (dt / dx) * (F[1:] - F[:-1])
            unnt[n + 1] = unew

        t[n + 1] = t[n] + dt

    return t, unnt
