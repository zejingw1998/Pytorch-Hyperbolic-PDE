#Ex1 Extended
import torch as torch
import math
import matplotlib.pylab as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

#We define the test function first.
#In this exercise I will choose f(x)=x^4


#Define the function
def f(x):
    return x**4


#I will use autograd to compute the exact derivative.
#Here I want to compare the numerical derivative
#with the exact one at one point.


#Autograd
def df_autograd(x0):
    x = torch.tensor(x0, device=device, dtype=dtype, requires_grad=True)
    y = f(x)
    y.backward()
    return x.grad


#Second-order centered difference
def deriv_cent(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


#Fourth-order difference
def deriv_4tho(f, x, h):
    return (-f(x + 2.0*h) + 8.0*f(x + h) - 8.0*f(x - h) + f(x - 2.0*h)) / (12.0 * h)


#Order of convergence
#I will use the formula from the exercise sheet.
def order_conv(fN, f2N, f4N):
    return torch.log(torch.abs((f2N - fN) / (f4N - f2N))) / torch.log(torch.tensor(2.0, device=device, dtype=dtype))


#Choose the point x0
#I do not use x0=0, because then the derivative is 0
#and it is not good for checking numerically.
x0 = torch.tensor(1.0, device=device, dtype=dtype)

#Use autograd to get the exact derivative
df_exact = df_autograd(1.0)

print("Exact derivative from autograd =", df_exact.item())


#The grids N, 2N, 4N
#Since h=L/N, here I simply take L=1.
L = 1.0
N = 20

hN  = torch.tensor(L / N, device=device, dtype=dtype)
h2N = torch.tensor(L / (2 * N), device=device, dtype=dtype)
h4N = torch.tensor(L / (4 * N), device=device, dtype=dtype)


#Test the central difference
fN_cent  = deriv_cent(f, x0, hN)
f2N_cent = deriv_cent(f, x0, h2N)
f4N_cent = deriv_cent(f, x0, h4N)

m_cent = order_conv(fN_cent, f2N_cent, f4N_cent)

print("\nCentral difference")
print("fN   =", fN_cent.item())
print("f2N  =", f2N_cent.item())
print("f4N  =", f4N_cent.item())
print("order =", m_cent.item())
print("error against autograd =", abs(f4N_cent - df_exact).item())


#Test the fourth-order difference
fN_4  = deriv_4tho(f, x0, hN)
f2N_4 = deriv_4tho(f, x0, h2N)
f4N_4 = deriv_4tho(f, x0, h4N)

m_4 = order_conv(fN_4, f2N_4, f4N_4)

print("\nFourth-order difference")
print("fN   =", fN_4.item())
print("f2N  =", f2N_4.item())
print("f4N  =", f4N_4.item())
print("order =", m_4.item())
print("error against autograd =", abs(f4N_4 - df_exact).item())


#Extra test for f(x)=x^5
#I will also test x^5, because for x^4 the fourth-order formula
#can be too accurate, so the order is not always so easy to see.


#Define the function
def f5(x):
    return x**5

def df_autograd_f5(x0):
    x = torch.tensor(x0, device=device, dtype=dtype, requires_grad=True)
    y = f5(x)
    y.backward()
    return x.grad

#Choose the same point x0=1
x0_5 = torch.tensor(1.0, device=device, dtype=dtype)

#Exact derivative from autograd
df_exact_5 = df_autograd_f5(1.0)

print("\n==================================================")
print("Now test f(x)=x^5")
print("Exact derivative from autograd =", df_exact_5.item())

#Use the same hN, h2N, h4N as before

#Test the central difference for x^5
fN_cent_5  = deriv_cent(f5, x0_5, hN)
f2N_cent_5 = deriv_cent(f5, x0_5, h2N)
f4N_cent_5 = deriv_cent(f5, x0_5, h4N)

m_cent_5 = order_conv(fN_cent_5, f2N_cent_5, f4N_cent_5)

print("\nCentral difference for x^5")
print("fN   =", fN_cent_5.item())
print("f2N  =", f2N_cent_5.item())
print("f4N  =", f4N_cent_5.item())
print("order =", m_cent_5.item())
print("error against autograd =", abs(f4N_cent_5 - df_exact_5).item())

#Test the fourth-order difference for x^5
fN_4_5  = deriv_4tho(f5, x0_5, hN)
f2N_4_5 = deriv_4tho(f5, x0_5, h2N)
f4N_4_5 = deriv_4tho(f5, x0_5, h4N)

m_4_5 = order_conv(fN_4_5, f2N_4_5, f4N_4_5)

print("\nFourth-order difference for x^5")
print("fN   =", fN_4_5.item())
print("f2N  =", f2N_4_5.item())
print("f4N  =", f4N_4_5.item())
print("order =", m_4_5.item())
print("error against autograd =", abs(f4N_4_5 - df_exact_5).item())




#Conclusion

#For f(x)=x^4, the fourth-order formula is very accurate,
#so the order here can be affected by machine precision.

#That is why I also test f(x)=x^5.

#For f(x)=x^5, the central difference is close to order 2.

#And the fourth-order formula is close to order 4.