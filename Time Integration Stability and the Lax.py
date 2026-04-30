# Time Integration Stability and the Lax Method
import numpy as np
import torch 
import matplotlib.pyplot as plt
import importlib
import nm_lib.nm_ex.nm_lib_ex_4 as nm4
importlib.reload(nm4)
def lax_burgers_one_step_from_lib(x_t, u_t, cfl_cut=0.8):
     # Convert torch tensors to numpy arrays,
    # because nm4 in the lib is written in numpy.
    x_np = x_t.detach().cpu().numpy()
    u_np = u_t.detach().cpu().numpy()
    # Use the lib function to evolve exactly one time step
    # for Burgers' equation with the Lax-type method.
    t_np, unnt_np = nm4.evolv_Lax_uadv_burgers(
        x_np, u_np, nt=1, cfl_cut=cfl_cut, bnd_type="wrap"
    )

    dt = t_np[1] - t_np[0]
     # Convert the updated numpy solution back to torch,
    # so the rest of the notebook can still use torch arrays.
    u_new = torch.tensor(unnt_np[-1], dtype=u_t.dtype, device=u_t.device)
    return u_new, float(dt)


def lax_adv_one_step_from_lib(x_t, u_t, a, cfl_cut=0.8):
    x_np = x_t.detach().cpu().numpy()
    u_np = u_t.detach().cpu().numpy()

    t_np, unnt_np = nm4.evolv_Lax_adv_burgers(
        x_np, u_np, nt=1, a=a, cfl_cut=cfl_cut, bnd_type="wrap"
    )

    dt = t_np[1] - t_np[0]
    u_new = torch.tensor(unnt_np[-1], dtype=u_t.dtype, device=u_t.device)
    return u_new, float(dt)


def ftbs_one_step(u, a, dx, dt):
    N_local = len(u)
    u_new = torch.empty_like(u)
    lam = a * dt / dx

    for j in range(N_local):
        jm = (j - 1) % N_local
        u_new[j] = u[j] - lam * (u[j] - u[jm])

    return u_new

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

# Problem 4

# In this problem we consider the Burger's equation

# We consider the intial condition

#Define the function

#Intial 
x0=-1.4
xf = 2.0
# Parameters
A = 1.0
xc = 0.70
W = 0.1
tf = 100.0
CFL = 0.8 #Must less 1
N = 400


def burgers_initial(x, A=1.0, xc=0.70, W=0.1):
    return A * (torch.tanh((x + xc)/W) - torch.tanh((x - xc)/W))

#Part 1


# The Lax Method

#Task 1.1
# In this part, we need to work with the von Neumann's Stability analysis.
#The PDF

#Task 1.2
# The code.
"""
def evolv_Lax_uadv_burgers(u, dx, dt):
   
   # One step of the Lax method for Burgers' equation:
   #u_t + u * u_x = 0
   #with periodic boundary conditions.
    
    N_local = len(u)                 # Length of the solution array
    u_new = torch.empty_like(u)      # Create a new array with the same shape
    lam = dt / (2.0 * dx)

    for j in range(N_local):
        # Periodic boundary conditions:
        # if index goes beyond the right end, wrap to the left end
        # if index goes beyond the left end, wrap to the right end
        jp = (j + 1) % N_local       # index j+1
        jm = (j - 1) % N_local       # index j-1

        # Lax update formula (directly from the scheme)
        u_new[j] = (u[jm] + u[j] + u[jp]) / 3.0 - (u[j] * lam) * (u[jp] - u[jm]) #Update the lax step including u[j]

    return u_new
"""

# 4) Build grid x and compute dx

x = torch.linspace(x0, xf, N)
dx = (xf - x0) / (N - 1)

# Intial data
u = burgers_initial(x)
u0 = u.clone()   # Save a copy of the initial condition for plotting


# Time stepping  CFL c

t = 0.0
step = 0

while t < tf:
    u_new, dt = lax_burgers_one_step_from_lib(x, u, cfl_cut=CFL)

    if t + dt > tf:
        break

    u = u_new
    t = t + dt
    step = step + 1

print("Finished!")
print("t =", t)
print("steps =", step)


# 6) Plot 
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), u0.numpy(), label="intial")
plt.plot(x.numpy(), u.numpy(), label="Lax result")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.show()    

#Part 2 Numerical Diffusion


#Task 2.1: Compare Methods

#  Parameters 

x0 = -2.6
xf = 2.6
a = 1.0          
tf = 0.4        
CFL = 0.8

# Initial condition 
def initial_condition(x):
    return torch.cos(6*torch.pi*x/5)**2 / torch.cosh(5*x**2)

# 
#Original method from Ex 2b (FTFS)

def evolv_FTFS_uadv(u, a, dx, dt):
    N_local = len(u)
    u_new = torch.empty_like(u)
    lam = a * dt / dx

    for j in range(N_local):
        jp = (j + 1) % N_local
        u_new[j] = u[j] - lam * (u[jp] - u[j])

    return u_new


# Lax method for linear advection
"""
def evolv_Lax_uadv(u, a, dx, dt):
    N_local = len(u)
    u_new = torch.empty_like(u)
    lam = a * dt / (2.0 * dx)

    for j in range(N_local):
        jp = (j + 1) % N_local
        jm = (j - 1) % N_local
        u_new[j] = (u[jm] + u[j] + u[jp]) / 3.0 - lam * (u[jp] - u[jm])

    return u_new
"""

# Grid and initial data

x = torch.linspace(x0, xf, N)
dx = (xf - x0) / (N - 1)

u0 = initial_condition(x)

#The FTBS
u_ftbs = u0.clone()
u_lax  = u0.clone()


# Use same dt for both methods 
dt = CFL * dx / abs(a)

# Time stepping

t = 0.0
step = 0
t = 0.0
step = 0
while t < tf:
    dt_use = dt
    if t + dt_use > tf:
        dt_use = tf - t

    u_ftbs = ftbs_one_step(u_ftbs, a, dx, dt_use)

    u_lax_new, dt_lax = lax_adv_one_step_from_lib(x, u_lax, a=a, cfl_cut=CFL)
    if t + dt_lax > tf:
        break

    u_lax = u_lax_new
    t += dt_lax
    step += 1

print(f"Finished at t = {t:.4f}, steps = {step}")


# Plot comparison

plt.figure(figsize=(9, 4))
plt.plot(x.numpy(), u0.numpy(), '--', label='intial')
plt.plot(x.numpy(), u_ftbs.numpy(), label='FTBS')
plt.plot(x.numpy(), u_lax.numpy(), label='Lax')
plt.xlabel('x')
plt.ylabel('u')
plt.title(f'Compare methods at t={t:.2f}')
plt.legend()
plt.grid(True)
plt.show()




#Conclusion. 

#Task 1.2
# The simulation reached the final time t = 100 in 13423 CFL-controlled time steps,
#  and the Lax solution has been strongly smeared buy nmerical diffusion
#Becoming nearly uniform and close to the spatial average of the initial profile

#Task 1.3 
# With N = 64, the solution is clearly under-resolved and significantly more smeared,
# especially near the steep front, due to the strong numerical diffusion of the Lax method.
# As the resolution increases to N = 128 and higher, the solution becomes sharper,
# and the curves begin to overlap, indicating convergence of the numerical solution
# under grid refinement.

# Part 2
# For a > 0, the FTBS scheme is the stable upwind method and is therefore
# a more meaningful method to compare with Lax.
# Both FTBS and Lax remain stable for CFL <= 1.
# In this simulation, the two solutions are very close, but the Lax solution appears
# slightly more diffusive because of its averaging term, so it is slightly more smeared
# than the FTBS solution.