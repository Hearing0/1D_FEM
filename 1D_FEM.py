# 1D Finite Element Method implementation in Python
# By David Flores

#TODO: Add input values using reflection problem

import numpy as np
import matplotlib.pyplot as plt
from tkinter import font
import tkinter as tk
import cmath
import math
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
num_nodes = M + 1

# Boundary condition type (1 for Dirichlet, 2 for Neumann, 3 for 3rd Kind)
bc = 1
p = 100  # Starting value
q = 0    # Ending value 


alpha = 1
beta_o = omega * np.sqrt(L_inductance * C_capacitance)
beta = -(beta_o**2)
gamma = 1j * beta
print(f"Beta (rad/m): {beta}")

# Initialize element properties
# phi = np.zeros(M, dtype=complex)
# for i in range(M):
#     phi[i] = alpha + beta*(i + 1)

a = np.zeros(num_nodes, dtype=complex) # Main diagonal of the tridiagonal matrix
c = np.zeros(num_nodes - 1, dtype=complex) # Upper diagonal of the tridiagonal matrix
b = np.zeros(num_nodes, dtype=complex) # Right-hand side vector

# # Assemble element matrices
# for element in range(M):
#     a[element] += 1 / h
#     a[element + 1] += 1 / h
#     c[element] -= 1 / h

#     b[element] += h / 2
#     b[element + 1] += h / 2

k11 = (alpha / h) + (beta * h / 3)
k12 = (-alpha / h) + (beta * h / 6)
print(f"k11: {k11}, k12: {k12}")
# Prepare the system of equations
for i in range(M):
    
    a[i] += k11
    a[i + 1] += k11
    c[i] = k12
    
    if (i % 10 == 0):  # Print every 10th element
        print(f"Element {i}: a[{i}]={a[i]}, a[{i+1}]={a[i+1]}, c[{i}]={c[i]}")
    
    # b[i] += phi[i] * h / 2
    # b[i + 1] += phi[i] * h / 2


# Apply Boundry Conditions
if bc == 1:
    p = 100.0  # Voltage at x = 0
    q = 0.0    # Voltage at x = L

    # Left boundary (node 0)
    a[0] = 1.0
    c[0] = 0.0
    b[0] = p

    # Right boundary (node M)
    a[-1] = 1.0
    d[-1] = 0.0
    b[-1] = q
# elif bc == 2:
#     # Apply Neumann boundary condition
#     pass 
# elif bc == 3:
#     # Apply 3rd Kind boundary condition
#     for i in range(M):
#         a[i] += gamma
#         b[i] += q

# --- Thomas Algorithm Solver ---
# Forward Sweep
for i in range(1, num_nodes):
    # w = c[-(i-1)] / a[i-1]
    w = c[-i] / a[i-1]
    a[i] = a[i] - w * c[i-1]
    b[i] = b[i] - w * b[i-1]

# Backward Substitution
final_phi = np.zeros(num_nodes, dtype=complex)
final_phi[-1] = b[-1] / a[-1]
for i in range(num_nodes - 2, -1, -1):
    final_phi[i] = (b[i] - c[i] * final_phi[i+1]) / a[i]

# # Solve the system of equations (using Gaussian elimination for tridiagonal matrix)
# # Find the modified coefficients for the tridiagonal system
# for i in range(1, M):
#     a[i] = a[i] - (c[i-1] * c[i-1]) / a[i-1]
#     b[i] = b[i] - (c[i-1] * b[i-1]) / a[i-1]
#     print(f"coefficients; i={i}: a[i]={a[i]}, b[i]={b[i]}")

# # Backwards Gaussian Elimination
# final_phi[M-1] = b[M-1] / a[M-1]
# for i in range(M-1, 0, -1):
#     final_phi[i] = (b[i] - c[i] * final_phi[i]) / a[i]

print(f"Voltage at x=0: {final_phi[0]:.2f}")
print(f"Voltage at x=L/2: {final_phi[M//2]:.2f}")
print(f"Voltage at x=L: {final_phi[-1]:.2f}")

# Graph the final solution
# plt.plot(range(num_nodes), abs(final_phi), marker='o')
# plt.xlabel('Element')
# plt.ylabel('Reflection')
# # plt.ylim(0,1)
# plt.title('1D FEM Solution')
# plt.grid(True)
# plt.show()

# --- Plotting ---
x = np.linspace(0, L, num_nodes)  # Create physical x-axis

plt.figure(figsize=(10, 5))
plt.plot(x, np.abs(final_phi), 'b-', linewidth=1.5)
plt.xlabel('Position x (m)')
plt.ylabel('|V(x)| (Volts)')
plt.title('Standing Wave Magnitude along Transmission Line (FEM)')
plt.grid(True)
plt.show()

# Optional: Plot real part to see the actual instant wave
plt.figure(figsize=(10, 5))
plt.plot(x, np.real(final_phi), 'r-', linewidth=1.5)
plt.xlabel('Position x (m)')
plt.ylabel('Re{V(x)} (Volts)')
plt.title('Instantaneous Voltage along Transmission Line (FEM)')
plt.grid(True)
plt.show()