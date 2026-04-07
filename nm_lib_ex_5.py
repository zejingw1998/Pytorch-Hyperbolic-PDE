"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

import numpy as np

from nm_lib.nm_ex.nm_lib_ex_1 import deriv_fwd


def cfl_diff_burger(a: float, x: np.ndarray) -> float:
    r"""
    Computes the dt_fact, i.e., Courant, Fredrich, and
    Lewy condition for the diffusive term in the Burger's eq.
    """
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]
     # explicit diffusion stability: dt <= 0.5 * dx^2 / |a|
    if np.isscalar(a):
        aa = abs(a)
        if aa < 1e-14:
            return 0.5 * dx**2
        return 0.5 * dx**2 / aa
    else:
        a = np.asarray(a, dtype=float)
        aa = np.max(np.abs(a))
        if aa < 1e-14:
            return 0.5 * dx**2
        return 0.5 * dx**2 / aa


def deriv2_cent(xx: np.ndarray, hh: np.ndarray, **kwargs) -> np.ndarray:
    r"""
    Returns the centered 2nd derivative of hh with respect to xx.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dx = xx[1] - xx[0]
     # centered second derivative on interior points
    d2h = np.full_like(hh, np.nan, dtype=float)
    d2h[1:-1] = (hh[2:] - 2.0 * hh[1:-1] + hh[:-2]) / dx**2

    return d2h


def step_diff_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    a: float,
    **kwargs,
) -> np.ndarray:
    r"""
    Right hand side of the diffusive term of Burger's eq.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dx = xx[1] - xx[0]
    bnd_type = kwargs.get("bnd_type", "wrap")

    if bnd_type == "wrap":
        # periodic Laplacian
        lap = (np.roll(hh, -1) - 2.0 * hh + np.roll(hh, 1)) / dx**2
    else:
        lap = deriv2_cent(xx, hh)

    return a * lap


def evolv_diff_burgers(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: float,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv2_cent(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a fix constant or array.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)
# choose dt from the explicit diffusion CFL condition
    dt = cfl_cut * cfl_diff_burger(a, xx)

    t = np.arange(nt + 1, dtype=float) * dt
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        rhs = step_diff_burgers(xx, unnt[n], a, bnd_type=bnd_type)
        unnt[n + 1] = unnt[n] + dt * rhs

    return t, unnt


def step_diff_variable(
    xx: np.ndarray,
    hh: np.ndarray,
    mu=lambda x, y: 1.0,
) -> np.ndarray:
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a constant or a function that
    depends on xx.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dx = xx[1] - xx[0]
    lap = (np.roll(hh, -1) - 2.0 * hh + np.roll(hh, 1)) / dx**2
     # variable diffusion coefficient
    coeff = mu(xx, hh)
    coeff = np.asarray(coeff, dtype=float)

    return coeff * lap


def evolv_diff_variable(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    cfl_cut: float = 0.98,
    mu=lambda x, y: 1.0,
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    **kwargs,
):
    r"""
    Advance nt time-steps in time the burger eq for a being a fix constant or array.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    # estimate a safe explicit dt using the largest diffusion coefficient
    coeff0 = mu(xx, hh)
    coeff0 = np.asarray(coeff0, dtype=float)
    amax = np.max(np.abs(coeff0))

    if amax < 1e-14:
        dt = cfl_cut * 0.5 * (xx[1] - xx[0])**2
    else:
        dt = cfl_cut * cfl_diff_burger(amax, xx)

    t = np.arange(nt + 1, dtype=float) * dt
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    for n in range(nt):
        rhs = step_diff_variable(xx, unnt[n], mu=mu)
        unnt[n + 1] = unnt[n] + dt * rhs

    return t, unnt


def NR_f(
    xx: np.ndarray,
    un: np.ndarray,
    uo: np.ndarray,
    a: float,
    dt: float,
    **kwargs,
) -> np.ndarray:
    r"""
    NR F function.
    """
    xx = np.asarray(xx, dtype=float)
    un = np.asarray(un, dtype=float)
    uo = np.asarray(uo, dtype=float)

    dx = xx[1] - xx[0]
    r = a * dt / dx**2
    # residual for implicit diffusion step
    F = un - uo - r * (np.roll(un, -1) - 2.0 * un + np.roll(un, 1))
    return F


def jacobian(xx: np.ndarray, un: np.ndarray, a: float, dt: float, **kwargs) -> np.ndarray:
    r"""
    Jacobian of the F function.
    """
    xx = np.asarray(xx, dtype=float)
    un = np.asarray(un, dtype=float)

    dx = xx[1] - xx[0]
    r = a * dt / dx**2
    N = len(un)
     # Jacobian matrix for the implicit periodic diffusion system
    J = np.zeros((N, N), dtype=float)

    for j in range(N):
        J[j, j] = 1.0 + 2.0 * r
        J[j, (j + 1) % N] = -r
        J[j, (j - 1) % N] = -r

    return J


def Newton_Raphson(
    xx: np.ndarray,
    hh: np.ndarray,
    a: np.ndarray,
    dt: float,
    nt: int,
    toll: float = 1e-5,
    ncount: int = 2,
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
):
    r"""
    NR scheme for the burgers equation.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    a : `float` or `array`
        Either constant or array multiplies the right-hand side of the Burger's eq.
    dt : `float`
        Time interval
    nt : `int`
        Number of time iterations.
    toll : `float`
        Error limit.
        By default 1e-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default, 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Array of time.
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    errt : `array`
        Error for each timestep
    countt : `list(int)`
        number iterations for each timestep
    """
    if bnd_limits is None:
        bnd_limits = [0, 1]
    err = 1.0
    unnt = np.zeros((np.size(xx), nt))
    errt = np.zeros(nt)
    countt = np.zeros(nt)
    unnt[:, 0] = hh
    t = np.zeros(nt)

    # Looping over time
    for it in range(1, nt):
        uo = unnt[:, it - 1]
        ug = unnt[:, it - 1]
        count = 0
        # iteration to reduce the error.
        while (err >= toll) and (count < ncount):
            jac = jacobian(xx, ug, a, dt)  # Jacobian
            ff1 = NR_f(xx, ug, uo, a, dt)  # F
            # Inversion:
            un = ug - np.matmul(np.linalg.inv(jac), ff1)

            # error:
            err = np.max(np.abs(un - ug) / (np.abs(un) + toll))
            errt[it] = err

            # Number of iterations
            count += 1
            countt[it] = count

            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = un[bnd_limits[0] : -bnd_limits[1]]
            else:
                u1_c = un[bnd_limits[0] :]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un
        err = 1.0
        t[it] = t[it - 1] + dt
        unnt[:, it] = un

    return t, unnt, errt, countt


def NR_f_u(
    xx: np.ndarray,
    un: np.ndarray,
    uo: np.ndarray,
    dt: float,
    **kwargs,
) -> np.ndarray:
    r"""
    NR F function.
    """
    xx = np.asarray(xx, dtype=float)
    un = np.asarray(un, dtype=float)
    uo = np.asarray(uo, dtype=float)

    dx = xx[1] - xx[0]
    lap = np.roll(un, -1) - 2.0 * un + np.roll(un, 1)
    # residual for implicit diffusion step
    F = un - uo - (dt / dx**2) * un * lap
    return F


def jacobian_u(
    xx: np.ndarray,
    un: np.ndarray,
    dt: float,
    **kwargs,
) -> np.ndarray:
    """
    Jacobian of the F function.
    """
    xx = np.asarray(xx, dtype=float)
    un = np.asarray(un, dtype=float)

    dx = xx[1] - xx[0]
    r = dt / dx**2
    N = len(un)

    J = np.zeros((N, N), dtype=float)

    for j in range(N):
        jp = (j + 1) % N
        jm = (j - 1) % N

        # dF_j / du_j
        J[j, j] = 1.0 - r * (un[jp] - 4.0 * un[j] + un[jm])

        # dF_j / du_{j+1}
        J[j, jp] = -r * un[j]

        # dF_j / du_{j-1}
        J[j, jm] = -r * un[j]

    return J


def Newton_Raphson_u(
    xx: np.ndarray,
    hh: np.ndarray,
    dt: float,
    nt: int,
    toll: float = 1e-5,
    ncount: int = 2,
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
):
    """
    NR scheme for the burgers equation.

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        A function that depends on xx.
    dt : `float`
        Time interval
    nt : `int`
        Number of time iterations.
    toll : `float`
        Error limit.
        By default 1-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string`
        Allows to select the type of boundaries.
        By default, 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Time.
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain.
    errt : `array`
        Error for each timestep
    countt : `array(int)`
        Number iterations for each timestep
    """
    if bnd_limits is None:
        bnd_limits = [0, 1]
    err = 1.0
    unnt = np.zeros((np.size(xx), nt))
    errt = np.zeros(nt)
    countt = np.zeros(nt)
    unnt[:, 0] = hh
    t = np.zeros(nt)

    # Looping over time
    for it in range(1, nt):
        uo = unnt[:, it - 1]
        ug = unnt[:, it - 1]
        count = 0
        # iteration to reduce the error.
        while (err >= toll) and (count < ncount):
            jac = jacobian_u(xx, ug, dt)  #  Jacobian
            ff1 = NR_f_u(xx, ug, uo, dt)  #  F
            # Inversion:
            un = ug - np.matmul(np.linalg.inv(jac), ff1)

            # error
            err = np.max(np.abs(un - ug) / (np.abs(un) + toll))
            errt[it] = err

            # Number of iterations
            count += 1
            countt[it] = count

            # Boundaries
            if bnd_limits[1] > 0:
                u1_c = un[bnd_limits[0] : -bnd_limits[1]]
            else:
                u1_c = un[bnd_limits[0] :]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un
        err = 1.0
        t[it] = t[it - 1] + dt
        unnt[:, it] = un

    return t, unnt, errt, countt


def taui_sts(nu: float, niter: int, iiter: int) -> float:
    """
    STS parabolic scheme.
    """
    # STS sub-step coefficient
    return 1.0 / ((nu - 1.0) * np.cos(np.pi * (2.0 * iiter - 1.0) / (2.0 * niter)) + nu + 1.0)


def evol_sts(
    xx: np.ndarray,
    hh: np.ndarray,
    nt: int,
    a: np.ndarray,
    cfl_cut: float = 0.98,
    ddx=lambda x, y: deriv_fwd(x, y),
    bnd_type: str = "wrap",
    bnd_limits: list | None = None,
    nu: float = 0.9,
    n_sts: float = 10,
):
    """
    Evolution of the STS method.
    """
    xx = np.asarray(xx, dtype=float)
    hh = np.asarray(hh, dtype=float)

    dt_exp = cfl_cut * cfl_diff_burger(a, xx)

    t = np.zeros(nt + 1, dtype=float)
    unnt = np.zeros((nt + 1, len(hh)), dtype=float)
    unnt[0] = hh.copy()

    n_sts = int(n_sts)

    for n in range(nt):
        u = unnt[n].copy()
        dt_tot = 0.0

        for i in range(1, n_sts + 1):
            tau = dt_exp * taui_sts(nu, n_sts, i)
            rhs = step_diff_burgers(xx, u, a, bnd_type=bnd_type)
            u = u + tau * rhs
            dt_tot += tau

        unnt[n + 1] = u
        t[n + 1] = t[n] + dt_tot

    return t, unnt