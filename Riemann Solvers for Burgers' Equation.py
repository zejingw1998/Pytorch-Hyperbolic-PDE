import torch 
import matplotlib.pyplot as plt
import math

device = torch.device("cpu")
dtype = torch.float64

print("Using device:", device)

# Parameter.

x0=-1.4
xf = 2
A = 0.02
W = 0.1
xc = 0.7
v = 0.1
N = 512 #In this problem does not specify N, I just assumed that N = 512, which works well.    
tf = 100.0    
CFL = 0.8 
#Intial
def burgers_initial(x, A=0.02, xc=0.7, W=0.1):
    return A * (torch.tanh((x + xc)/W) - torch.tanh((x - xc)/W))

#Conservative Lax-Friedrichs and Rusanov
def F_burgers(u):
    return 0.5 * u * u  # F(u)=u^2/2

def evolv_Lax_uadv_burgers(u, dx, dt):
    N = u.numel()
    u_new = torch.zeros_like(u)
    lam = dt / (2.0 * dx)

    for j in range(N):
        jp = (j + 1) % N   # j+1 periodic
        jm = (j - 1) % N   # j-1 periodic

        u_new[j] = 0.5 * (u[jp] + u[jm]) - lam * (F_burgers(u[jp]) - F_burgers(u[jm]))

    return u_new

def rusanov_interface_flux(uL, uR):
    
    FL = 0.5 * uL * uL
    FR = 0.5 * uR * uR
    a  = torch.maximum(torch.abs(uL), torch.abs(uR))
    return 0.5 * (FL + FR) - 0.5 * a * (uR - uL)


def evolv_Rusanov_burgers(u, dx, dt):
    N = u.numel()
    u_new = torch.zeros_like(u)

    # F_half[j] means flux at interface (j+1/2) between j and j+1
    F_half = torch.zeros_like(u)

    #interface fluxes
    for j in range(N):
        jp = (j + 1) % N
        F_half[j] = rusanov_interface_flux(u[j], u[jp])

    # update by flux difference
    for j in range(N):
        jm = (j - 1) % N
        u_new[j] = u[j] - (dt / dx) * (F_half[j] - F_half[jm])

    return u_new


def total_variation_periodic(u):
    # TV = sum |u_{i}-u_{i-1}| with periodic wrap
    N = u.numel()
    tv = 0.0
    for j in range(N):
        jm = (j - 1) % N
        tv += abs((u[j] - u[jm]).item())
    return tv

def to_np(u):
    return u.cpu().numpy()


# Part 2 Compare Lax vs Rusanov for Burgers, same intial, and same CFL


# Use x0, xf, A, xc, W, tf, CFL, N


dx = (xf - x0) / N
x  = x0 + dx * torch.arange(N, device=device, dtype=dtype)

u0 = burgers_initial(x, A=A, xc=xc, W=W)
u_lax = u0.clone()
u_rie = u0.clone()


t = 0.0
step = 0
eps = 1e-12

while t < tf:
    # Use SAME dt for both, based on current max speed among them
    umax = max(torch.max(torch.abs(u_lax)).item(), torch.max(torch.abs(u_rie)).item())
    dt = CFL * dx / max(umax, eps)

    if t + dt > tf:
        dt = tf - t

    u_lax = evolv_Lax_uadv_burgers(u_lax, dx, dt)
    u_rie = evolv_Rusanov_burgers(u_rie, dx, dt)

    t += dt
    step += 1


print("\n Compare Lax vs Rusanov (Burgers)")
print(f"Finished at t = {t:.6f}, steps = {step}")
print(f"Peak:  initial={torch.max(u0).item():.6f} | Lax={torch.max(u_lax).item():.6f} | Rusanov={torch.max(u_rie).item():.6f}")
print(f"TV:    initial={total_variation_periodic(u0):.6f} | Lax={total_variation_periodic(u_lax):.6f} | Rusanov={total_variation_periodic(u_rie):.6f}")

plt.figure(figsize=(10,4))
plt.plot(to_np(x), to_np(u0), '--', label='initial')
plt.plot(to_np(x), to_np(u_lax), label='Lax ')
plt.plot(to_np(x), to_np(u_rie), label='Rusanov ')
plt.xlabel('x'); plt.ylabel('u')
plt.title(f' Lax vs Rusanov at t={t:.2f}, N={N}, CFL={CFL}')
plt.legend(); plt.grid(True)
plt.show()


#Rusanov is less diffusive than Lax with higher peak and higher TV at the same CFL.