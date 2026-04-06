import numpy as np
import torch
import matplotlib.pyplot as plt
import importlib

import nm_lib.nm_ex.nm_lib_ex_3 as nm3
importlib.reload(nm3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

def lib_ddx_torch(ddx_fun, u, dx):
    xx = (torch.arange(u.numel(), device=device, dtype=dtype) * dx).detach().cpu().numpy()
    uu = u.detach().cpu().numpy()
    du = ddx_fun(xx, uu)
    du = np.nan_to_num(du, nan=0.0)
    return torch.tensor(du, device=device, dtype=dtype)



#Problem 3 The Numerical Instability and Stability Analysis

# Try to use Torch (GPU) if available 
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = torch.float64
torch.set_default_dtype(dtype)

USE_TORCH = TORCH_AVAILABLE  # True: torch and False: numpy


# In Task 1.1 we keep the same periodic grid style as in Exercise 2b,
# while in the later parts we use ghost cells for convenience.
#Part 1
# Task 1.1
# We repeat the numerical simulation from Exercise 2b, but now with a = +1 (instead of a = -1) 
# We recall the FTFS, if we have the constant a > 0. this method will be changed to Downwind, the error increases. 
# Define the intial value
x0= -2.6
xf = 2.6
a = +1.0
cfl =0.98
TH_FACTOR = 3.0
max_steps = 5000


# initial condition
def u0(x):
    
    return torch.cos(6*torch.pi*x/5)**2 / torch.cosh(5*x**2)

def the_periodic(u):
    u[0] = u[-2]
    u[-1] = u[1]
    return u
# Consider that the FTFS and periodic and count which step it blows up at
def run_and_count(N):
    dx = (xf - x0) / N
    dt = cfl * dx / abs(a)

    # grid, N intervals,we have N+1 points
    x = torch.linspace(x0, xf, N+1, device=device, dtype=dtype)

    # initial condition + periodic
    u = u0(x)
    u[-1] = u[0]

    M0 = torch.max(torch.abs(u)).item()
    TH = TH_FACTOR * M0

    max_hist = [M0]
    onset = None

    for n in range(1, max_steps + 1):
        u_new = u.clone()

        # FTFS update
        u_new[:-1] = u[:-1] - a * (u[1:] - u[:-1]) * (dt / dx)

        # periodic
        u_new[-1] = u_new[0]

        u = u_new

        Mn = torch.max(torch.abs(u)).item()
        max_hist.append(Mn)

        # How many steps we can get that 
        if (Mn > TH) or torch.isnan(u).any() or torch.isinf(u).any():
            onset = n
            break

    return onset, x, u, max_hist

# Define the intervals

for N in [128, 1024]:
    onset, x, u_end, max_hist = run_and_count(N)
    print(f"N={N} (points={N+1}), instability step = {onset}")

    plt.figure()
    plt.plot(max_hist)
    plt.yscale("log")
    plt.xlabel("timestep n")
    plt.ylabel("max|u| (log)")
    plt.title(f"max|u|, N={N}, onset={onset}")
    plt.grid(True)

    plt.figure()
    plt.plot(x.detach().cpu().numpy(), u_end.detach().cpu().numpy())
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"u(x) when stopped, N={N}")
    plt.grid(True)

plt.show()


# Part 1.2
# Use backward finite differences from nm_lib

def run_backward_lib(N, a=1.0, cfl_cut=0.98, TH=10.0, max_steps=5000):
    dx = (xf - x0) / N
    dt = cfl_cut * dx / abs(a)

    x_phys = torch.linspace(x0, xf, N+1, device=device, dtype=dtype)

    u = torch.empty(N+3, device=device, dtype=dtype)
    u[1:-1] = u0(x_phys)
    u = the_periodic(u)

    max_hist = [torch.max(torch.abs(u[1:-1])).item()]
    onset = None

    for n in range(1, max_steps + 1):
        dudx = lib_ddx_torch(nm3.deriv_bck, u, dx)

        u[1:-1] = u[1:-1] - a * dt * dudx[1:-1]
        u = the_periodic(u)

        Mn = torch.max(torch.abs(u[1:-1])).item()
        max_hist.append(Mn)

        if (Mn > TH) or torch.isnan(u).any() or torch.isinf(u).any():
            onset = n
            break

    return onset, x_phys, u[1:-1], max_hist
"""
# Task 1.2
# In this Task, we consider the a = 1, this means that  we need use the backward, because we need move to the right side.

# We consider the backward difference, this means that 
# u_x = (u_n-u_n-1)/delta(x)
#u_i^n+1 =u_i^n -a delta t / delta x ((u_n-u_n-1) )
# Define the function. 
def backward_derivative(u,dx):
    du = torch.zeros_like(u)
    N = u.numel()-1
    for i in range(1, N):
        du[i] = (u[i] - u[i-1]) / dx

    # periodic boundary，i=0 uses i-1 = N-1
    du[0] = (u[0] - u[N-1]) / dx
    return du

def evolv_adv_burgers_bck(u, dx, nsteps, a=1.0, cfl=0.98):
    dt = cfl * dx / abs(a)
    N = u.numel()-1
    u_now = u.clone()

    for _ in range(nsteps):
        dudx = backward_derivative(u_now, dx)
        u_now = u_now - a * dt * dudx
           
    return u_now
"""
# a = +1 : should be stable for CFL <= 1
onset, x_phys, u_end, max_hist = run_backward_lib(N=128, a=1.0, cfl_cut=0.98)
print(f"Backward difference, a=+1, onset = {onset}")

plt.figure()
plt.plot(max_hist)
plt.yscale("log")
plt.xlabel("timestep n")
plt.ylabel("max|u| (log)")
plt.title("Backward difference, a=+1")
plt.grid(True)

plt.figure()
plt.plot(x_phys.detach().cpu().numpy(), u_end.detach().cpu().numpy())
plt.xlabel("x")
plt.ylabel("u")
plt.title("Backward difference, a=+1")
plt.grid(True)

plt.show()


# a = -1 : backward becomes downwind, should be unstable
onset, x_phys, u_end, max_hist = run_backward_lib(N=128, a=-1.0, cfl_cut=0.98)
print(f"Backward difference, a=-1, onset = {onset}")

plt.figure()
plt.plot(max_hist)
plt.yscale("log")
plt.xlabel("timestep n")
plt.ylabel("max|u| (log)")
plt.title("Backward difference, a=-1")
plt.grid(True)

plt.figure()
plt.plot(x_phys.detach().cpu().numpy(), u_end.detach().cpu().numpy())
plt.xlabel("x")
plt.ylabel("u")
plt.title("Backward difference, a=-1")
plt.grid(True)

plt.show()


# Part 2
# Centered differences from nm_lib

def run_centered_lib(N, a=1.0, cfl_cut=0.3, TH=10.0, max_steps=2000):
    dx = (xf - x0) / N
    dt = cfl_cut * dx / abs(a)

    x_phys = torch.linspace(x0, xf, N+1, device=device, dtype=dtype)

    u = torch.empty(N+3, device=device, dtype=dtype)
    u[1:-1] = u0(x_phys)
    u = the_periodic(u)

    max_hist = [torch.max(torch.abs(u[1:-1])).item()]
    onset = None

    for n in range(1, max_steps + 1):
        dudx = lib_ddx_torch(nm3.deriv_cent, u, dx)

        u[1:-1] = u[1:-1] - a * dt * dudx[1:-1]
        u = the_periodic(u)
        # u[1:-1] are the physical points, while u[0] and u[-1] are ghost cells.
        Mn = torch.max(torch.abs(u[1:-1])).item()
        max_hist.append(Mn)

        if (Mn > TH) or torch.isnan(u).any() or torch.isinf(u).any():
            onset = n
            break

    return onset, x_phys, u[1:-1], max_hist



for a_val in [1.0, -1.0]:
    onset, x_phys, u_end, max_hist = run_centered_lib(N=128, a=a_val, cfl_cut=0.3)
    print(f"Centered difference, a={a_val:+.1f}, onset = {onset}")

    plt.figure()
    plt.plot(max_hist)
    plt.yscale("log")
    plt.xlabel("timestep n")
    plt.ylabel("max|u| (log)")
    plt.title(f"Centered difference, a={a_val:+.1f}")
    plt.grid(True)

    plt.figure()
    plt.plot(x_phys.detach().cpu().numpy(), u_end.detach().cpu().numpy())
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"Centered difference, a={a_val:+.1f}")
    plt.grid(True)

plt.show()
# Centered differences show oscillatory / unstable behavior.
# Changing the sign of a mainly changes the propagation direction,
# but does not remove the instability of the centered explicit scheme.

# Part 3  Stability of Non-Centered Schemes

# Part 3
# CFL threshold with backward (upwind) derivative from nm_lib

def run_upwind_and_count(N, cfl_cut, a=1.0, TH=10.0, max_steps=5000):
    dx = (xf - x0) / N
    dt = cfl_cut * dx / abs(a)

    x_phys = torch.linspace(x0, xf, N+1, device=device, dtype=dtype)

    u = torch.empty(N+3, device=device, dtype=dtype)
    u[1:-1] = u0(x_phys)
    u = the_periodic(u)

    max_hist = [torch.max(torch.abs(u[1:-1])).item()]
    onset = None

    for n in range(1, max_steps + 1):
        dudx = lib_ddx_torch(nm3.deriv_bck, u, dx)

        u[1:-1] = u[1:-1] - a * dt * dudx[1:-1]
        u = the_periodic(u)

        Mn = torch.max(torch.abs(u[1:-1])).item()
        max_hist.append(Mn)

        if (Mn > TH) or torch.isnan(u).any() or torch.isinf(u).any():
            onset = n
            break

    return onset, x_phys, u[1:-1], max_hist

"""
# Task 3.1
# Investigate the CFL Threshold

# In this problem, the aim is to test different dt and run this
# to see when it blows up.
# Since when a > 0, this is an explicit upwind scheme,
# the stability condition is the classical CFL.
# Hence cfl_cut = 1.

def the_periodic(u):
    u[0]  = u[-2]
    u[-1] = u[1]
    return u

def backward_derivative(u, dx):
    du = torch.zeros_like(u)
    du[1:-1] = (u[1:-1] - u[:-2]) / dx
    return du

def run_upwind_and_count(N, cfl_cut, a=1.0, TH=10.0, max_steps=5000):
    dx = (xf - x0) / N
    dt = cfl_cut * dx / abs(a)

    x_phys = torch.linspace(x0, xf, N+1, device=device, dtype=dtype)

    # u with  points
    u = torch.empty(N+3, device=device, dtype=dtype)
    u[1:-1] = u0(x_phys)
    u = the_periodic(u)

    max_hist = [torch.max(torch.abs(u[1:-1])).item()]
    onset = None

    for n in range(1, max_steps + 1):
        dudx = backward_derivative(u, dx)

        # update physical points
        u[1:-1] = u[1:-1] - a * dt * dudx[1:-1]
        u = the_periodic(u)

        Mn = torch.max(torch.abs(u[1:-1])).item()
        max_hist.append(Mn)

        # fixed blow-up rule
        if (Mn > TH) or torch.isnan(u).any() or torch.isinf(u).any():
            onset = n
            break

    return onset, x_phys, u[1:-1], max_hist
#Plot
for cfl_cut in [0.5, 0.99, 1.01, 2.0,3.0,4.0]: #Choose the value

    onset, x_phys, u_end, max_hist = run_upwind_and_count(N=128, cfl_cut=cfl_cut, a=1.0)
    print(f"cfl_cut={cfl_cut:>4}  onset(step)={onset}")

    plt.figure()
    plt.plot(max_hist)
    plt.yscale("log")
    plt.xlabel("timestep n")
    plt.ylabel("max|u| (log)")
    plt.title(f"Upwind/backward, cfl_cut={cfl_cut}, onset={onset}")
    plt.grid(True)

    plt.figure()
    plt.plot(x_phys.detach().cpu().numpy(), u_end.detach().cpu().numpy())
    plt.xlabel("x"); plt.ylabel("u")
    plt.title(f"u(x) when stopped, cfl_cut={cfl_cut}")
    plt.grid(True)

plt.show()
"""

for cfl_cut in [0.5, 0.99, 1.01, 2.0]:
    onset, x_phys, u_end, max_hist = run_upwind_and_count(N=128, cfl_cut=cfl_cut, a=1.0)
    print(f"cfl_cut={cfl_cut:>4}, onset(step)={onset}")

    plt.figure()
    plt.plot(max_hist)
    plt.yscale("log")
    plt.xlabel("timestep n")
    plt.ylabel("max|u| (log)")
    plt.title(f"Upwind/backward, cfl_cut={cfl_cut}, onset={onset}")
    plt.grid(True)

    plt.figure()
    plt.plot(x_phys.detach().cpu().numpy(), u_end.detach().cpu().numpy())
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"u(x) when stopped, cfl_cut={cfl_cut}")
    plt.grid(True)

plt.show()



#Comment

# Task 1.1

#When we change the a = -1 to a =1 , but still keep the FTFS scheme. The solution becomes unstable.
#The maximum amplitude max |u| grows rapidly. And the solution develops strong oscillations and spikes.
#For N = 128 the instability becomes clear after about 13 time steps, while for about 1024 points, 
#It appears after about 36 steps.
#Increasing N delays the onset, but the method is still unstable.


# Task 1.2

#Using the backward finite differences for the spatial derivative gives an upwind scheme when  a=+1
#And the simulation becomes stable for CFL <= 1
#If we use backward differences with a = -1
#The scheme becomes downwind and is unstable.



# The conclusion 

#We we confirmed the CFL stability limit for the explicit upwind scheme.
#It stay stable and becomes unstable for CFL > 1

#In the stable case the solution damped by numerical diffusion,
#While the downwind choice causes rapid blow up.

#About these plots.

#These plots show that when the CFL number is too large.

#The solution becomes unstable.

#max|u| grows rapidly and u(x) develops non-physical oscillations and spikes, Blows up.


#Part 4 Burgers' Equation (Optional)

# Part 4b  Forward Derivative and Lax Method
# In this problem we consider the Inviscid Burgers' Equation

# Task 4A Implement Burgers' Equation
#Define the intial condition 

def initial_burger(x,A =1.0): 
    #Parameter
    xc = 0.7
    W  = 0.1
    B  = 0.3
    return A*(torch.tanh((x+xc)/W) - torch.tanh((x-xc)/W)) + B
def the_periodic(u):
    u[0]  = u[-2]
    u[-1] = u[1]
    return u
#Consider the Lax-Method 
def step_uadv_burgers(u, dt, dx):
    u_next = torch.zeros_like(u)
    # flux f(u) = u^2/2 
    f = 0.5 * u**2
    # Lax-Friedrichs / Lax scheme
    u_next[1:-1] = 0.5*(u[2:] + u[:-2]) - (dt/(2*dx))*(f[2:] - f[:-2])
    u_next = the_periodic(u_next)
    return u_next
def evolv_uadv_burgers(N, tf=1.0, cfl_cut=0.5,A=1.0):
    # [-1.4, 2.0]
    x0, xf = -1.4, 2.0
    dx = (xf - x0) / N
    x_phys = torch.linspace(x0, xf, N+1, device=device, dtype=dtype)                                  
    u = torch.empty(N+3, device=device, dtype=dtype)
    u[1:-1] = initial_burger(x_phys,A=A)
    u = the_periodic(u)
    t = 0.0
    while t < tf:
        umax = torch.max(torch.abs(u[1:-1])).item()
        if umax < 1e-14:
            break
        dt = cfl_cut * dx / umax
        if t + dt > tf:
            dt = tf - t
        u = step_uadv_burgers(u, dt, dx)
        t += dt
    return x_phys, u[1:-1]
x, u_final = evolv_uadv_burgers(N=400, tf=1.0, cfl_cut=0.5)
plt.figure()
plt.plot(x.detach().cpu().numpy(), initial_burger(x).detach().cpu().numpy(), label="t=0")
plt.plot(x.detach().cpu().numpy(), u_final.detach().cpu().numpy(), label="t=1")
plt.grid(True); plt.legend()
plt.show()



# Task 4B Compare Methods for Burgers' Equation
# Forward Derivative Method
def step_uadv_burgers_forward(u, dt, dx):
    u_next = u.clone()
    # forward derivative version from the handout
    # u_j^{n+1} = u_j^n - (dt/dx) u_j^n (u_j^n - u_{j+1}^n)
    u_next[1:-1] = u[1:-1] - (dt / dx) * u[1:-1] * (u[1:-1] - u[2:])
    u_next = the_periodic(u_next)
    return u_next
def evolv_uadv_burgers_forward(N, tf=1.0, cfl_cut=0.5, A=1.0):
    # domain [-1.4, 2.0]
    x0, xf = -1.4, 2.0
    dx = (xf - x0) / N
    x_phys = torch.linspace(x0, xf, N+1, device=device, dtype=dtype)
    u = torch.empty(N+3, device=device, dtype=dtype)
    u[1:-1] = initial_burger(x_phys, A=A)
    u = the_periodic(u)
    t = 0.0
    while t < tf:
        umax = torch.max(torch.abs(u[1:-1])).item()
        if umax < 1e-14:
            break
        dt = cfl_cut * dx / umax
        if t + dt > tf:
            dt = tf - t
        u = step_uadv_burgers_forward(u, dt, dx)
        t += dt
    return x_phys, u[1:-1]
x = torch.linspace(-1.4, 2.0, 401, device=device, dtype=dtype)
x_lax, u_lax = evolv_uadv_burgers(N=400, tf=1.0, cfl_cut=0.5, A=1.0)
x_fwd, u_fwd = evolv_uadv_burgers_forward(N=400, tf=1.0, cfl_cut=0.5, A=1.0)
plt.figure()
plt.plot(x.detach().cpu().numpy(), initial_burger(x, A=1.0).detach().cpu().numpy(), label="t=0")
plt.plot(x_lax.detach().cpu().numpy(), u_lax.detach().cpu().numpy(), label="Lax")
plt.plot(x_fwd.detach().cpu().numpy(), u_fwd.detach().cpu().numpy(), label="Forward")
plt.grid(True)
plt.legend()
plt.show()
x_lax2, u_lax2 = evolv_uadv_burgers(N=400, tf=1.0, cfl_cut=0.5, A=-0.02)

x_fwd2, u_fwd2 = evolv_uadv_burgers_forward(N=400, tf=1.0, cfl_cut=0.5, A=-0.02)

plt.figure()
plt.plot(x.detach().cpu().numpy(), initial_burger(x, A=-0.02).detach().cpu().numpy(), label="t=0")
plt.plot(x_lax2.detach().cpu().numpy(), u_lax2.detach().cpu().numpy(), label="Lax")
plt.plot(x_fwd2.detach().cpu().numpy(), u_fwd2.detach().cpu().numpy(), label="Forward")
plt.grid(True)
plt.legend()
plt.show()

#Comment


# For Burgers' equation, the local propagation speed is a = u.
# This means that larger values of u move faster than smaller values,
# so compressive parts of the profile steepen and may develop into shock-like fronts.
#
# The Lax method is based on the conservative form
# u_t + (u^2 / 2)_x = 0,
# while the forward method discretizes u*u_x directly.
# These two formulations are equivalent in the continuous equation,
# but they are not equivalent at the discrete level once steep gradients appear.
#
# Therefore, the forward method may fail to approximate the correct weak solution,
# whereas the Lax method is more appropriate as a conservative reference method.
# On the other hand, the Lax method is diffusive, so the front is smeared numerically.
#
# When A = -0.02, the variation in u is much smaller,
# so the nonlinear steepening is weaker and the solution changes more gently.

#Figur 1
# The Burgers solution moves to the right and steepens because the local speed is u.
# Larger values travel faster, so the right edge compresses and the left edge spreads out.
# The Lax profile is smeared because of numerical diffusion.


#Figur 2

# The Lax and forward solutions differ strongly for A = 1.
# This shows that discretizing u*u_x directly is not equivalent to using the conservative flux form at the discrete level.
# The forward method may therefore fail to approximate the correct weak solution
# or produce an incorrect shock profile.


#Figur 3

# For A = -0.02, the variation in u is much smaller, so the nonlinear steepening is weaker.
# Both solutions stay closer to the initial profile, although they still do not match exactly.
