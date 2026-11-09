# 1D Finite Element Method implementation in Python
# Solving for solution: phi_exact = -1/2*x^2 + x
# By David Flores

#TODO: Add input values using reflection problem

import numpy as np
import matplotlib.pyplot as plt
from tkinter import font
import tkinter as tk
import cmath
import math

from pyparsing import line
import poisson_test as poisson

# Physical constants
epsilon_r = 5.0
mu_r = 1.0
mu0 = 4 * np.pi * 1e-7
eps0 = 8.854187817e-12

# Telegrahic equation parameters
L_inductance = 1.0e-6  # per unit length
C_capacitance = (mu_r * mu0 * epsilon_r * eps0) / L_inductance 
f = 1e9 
omega = 2 * np.pi * f

# Initialize problem parameters
M = 500  # Number of elements
L = 1.0  # Length of the domain
h = L / M  # Element length
N = M + 1

# Boundary condition type (1 for Dirichlet, 2 for Neumann, 3 for 3rd Kind)
bc = 1
p = 0    # Starting value: phi(0) = p
q = 0    # Ending Flux: alpha * phi' + gamma * phi = q @ L


alpha = 1
beta = 0
gamma = 0

# Initialize element properties
# phi = np.zeros(M, dtype=complex)
# for i in range(M):
#     phi[i] = alpha + beta*(i + 1)

a = np.zeros(N, dtype=complex) # Main diagonal of the tridiagonal matrix
c = np.zeros(N - 1, dtype=complex) # Upper diagonal of the tridiagonal matrix
b = np.zeros(N, dtype=complex) # Right-hand side vector

# Assemble Forcing Function Coefficients
forcing_func = np.zeros(N, dtype=complex)
for i in range(N):
    # x = i * h
    forcing_func[i] = 1.0


# Prepare the system of equations
k11 = (alpha / h) + (beta * h / 3)
k12 = (-alpha / h) + (beta * h / 6)
print(f"k11: {k11}, k12: {k12}")
for i in range(M):
    
    a[i] += k11
    a[i + 1] += k11
    c[i] = k12
    
    if (i % 10 == 0):  # Print every 10th element
        print(f"Element {i}: a[{i}]={a[i]}, a[{i+1}]={a[i+1]}, c[{i}]={c[i]}")
    
    b[i] += forcing_func[i] * h / 2
    b[i + 1] += forcing_func[i] * h / 2


# Apply Boundry Conditions
# if bc == 1:
    # Left boundary (node 0)
    a[0] = 1.0
    c[0] = 0.0
    b[0] = p

    # Right boundary (node M)
    # a[-1] = 1.0
    # b[-1] = q

# elif bc == 2:
#     # Apply Neumann boundary condition
#     pass 
# elif bc == 3:
#     # Apply Robin (3rd Kind) boundary condition
    a[-1] += gamma
    b[-1] += q


final_phi = np.zeros(N, dtype=complex)

# # --- Thomas Algorithm Solver ---
# # Forward Sweep
for i in range(1, N):
    # w = c[-(i-1)] / a[i-1]
    w = c[-i] / a[i-1]
    a[i] = a[i] - w * c[i-1]
    b[i] = b[i] - w * b[i-1]

# Backward Substitution
final_phi = np.zeros(N, dtype=complex)
final_phi[-1] = b[-1] / a[-1]
for i in range(N - 2, -1, -1):
    final_phi[i] = (b[i] - c[i] * final_phi[i+1]) / a[i]

# Solve the system of equations (using Gaussian elimination for tridiagonal matrix)
# Find the modified coefficients for the tridiagonal system
# for i in range(1, M):
#     a[i] = a[i] - c[i-1]**2 / a[i-1]
#     b[i] = b[i] - (c[i-1] * b[i-1]) / a[i-1]
#     print(f"coefficients; i={i}: a[i]={a[i]}, b[i]={b[i]}")

# # Backwards Gaussian Elimination
# final_phi[-1] = b[-1] / a[-1]
# for i in range(N-1, 0, -1):
#     final_phi[i] = (b[i] - c[i] * final_phi[i]) / a[i]

# final_phi = np.linalg.solve(np.diag(a) + np.diag(c, 1) + np.diag(c, -1), b)

print(f"Phi at x=0: {final_phi[0]:.2f}")
print(f"Phi at x=L/2: {final_phi[M//2]:.2f}")
print(f"Phi at x=L: {final_phi[-1]:.2f}")


# Model the exact solution for comparison
exact_phi = np.zeros(N, dtype=complex)
for i in range(N):
    x = i * h
    exact_phi[i] = x + (-1/2) * x**2

# Calc Residual, Avg Error, and Error till element
residual = np.abs(final_phi - exact_phi)/np.abs(exact_phi)
error_till_e = np.zeros(N, dtype=float)
for i in range(N):
    error_till_e[i] = np.sum(residual[:i+1]) / N
error = np.sum(residual) / N
print(f"Average Error: {error:.6f}")



# --- Plotting ---
# FEM Solution vs Exact Solution
x = np.linspace(0, L, N) 
plt.figure(figsize=(10, 5))
plt.plot(x, np.abs(final_phi), 'b-', linewidth=1.5)
plt.plot(x, np.abs(exact_phi), 'g--', linewidth=1.5)
plt.legend(['FEM Solution', 'Exact Solution'])
plt.xlabel('Domain Length (element * e_length)')
plt.ylabel('Output')
plt.title('1D FEM Solution vs Exact Solution: -1/2*x^2 + x')
plt.grid(True)
plt.show()

# Error vs Element
e = np.linspace(0, M, N)
plt.figure(figsize=(10, 5))
plt.plot(e, error_till_e, 'r-', linewidth=1.5)
plt.xlabel('Element')
plt.ylabel('Error')
plt.title('Residual Error per Element')
plt.grid(True)
plt.show()

# Error vs h decreasing in log scale
h_decreasing = np.linspace(L, 0, N)
plt.figure(figsize=(10, 5))
plt.plot(e, error_till_e, 'm-', linewidth=1.5)
