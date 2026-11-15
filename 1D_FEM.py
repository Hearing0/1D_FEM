# 1D Finite Element Method implementation in Python
# Solving for solution: phi_exact = -1/2*x^2 + x
# By David Flores



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

# Initialize problem parameters
n_elements = 4  # Number of elements
L = 1.0  # Length of the domain
h = L / n_elements  # Element length
n_nodes = n_elements + 1
n_exact_phi_resolution = 1000

# Boundary condition type (1 for Dirichlet, 2 for Neumann, 3 for 3rd Kind)
bc = 1
p = 0    # Starting value: phi(0) = p

# Neumann boundary parameters
# neumann_at = 1
q = 0    # Ending Flux: alpha * phi' + gamma * phi = q @ L
gamma = 0

alpha = 1
beta = 0

# Initialize element properties
# phi = np.zeros(M, dtype=complex)
# for i in range(M):
#     phi[i] = alpha + beta*(i + 1)

a = np.zeros(n_nodes, dtype=complex) # Main diagonal of the tridiagonal matrix
c = np.zeros(n_nodes - 1, dtype=complex) # Upper diagonal of the tridiagonal matrix
b = np.zeros(n_nodes, dtype=complex) # Right-hand side vector

# Assemble Forcing Function Coefficients
forcing_func = np.zeros(n_nodes, dtype=complex)
for i in range(n_nodes):
    # x = i * h
    forcing_func[i] = 1.0


# Prepare the system of equations
k11 = (alpha / h) + (beta * h / 3)
k12 = (-alpha / h) + (beta * h / 6)
print(f"k11: {k11}, k12: {k12}")
for i in range(n_elements):
    
    a[i] += k11
    a[i + 1] += k11
    c[i] = k12
    
    if (i % 10 == 0):  # Print every 10th element
        print(f"Element {i}: a[{i}]={a[i]}, a[{i+1}]={a[i+1]}, c[{i}]={c[i]}")
    
    b[i] += forcing_func[i] * (h / 2)
    b[i + 1] += forcing_func[i] * (h / 2)

c[-1] = k12



# Apply Boundry Conditions
if bc == 1:
    # Left boundary (node 0)
    a[0] = 1.0
    b[0] = p
    b[1] -= c[0] * p
    c[0] = 0.0
    

    # Right boundary (node M)
    # a[-1] = 1.0
    # b[-1] = q

# elif bc == 2:
#     # Apply Neumann boundary condition
#     pass 
# elif bc == 3:
#     # Apply Robin (3rd Kind) boundary condition
    # a[-1] += gamma
    # b[-1] += q
    
# # Check coefficient arrays
# print(f"a: {a}")
# print(f"b: {b}")
# print(f"c: {c}")


# for i in range(n_nodes):
#     if i < n_nodes - 1: 
#         print(f"K[{i}]: {a[i]:.2f}, c[{i}]: {c[i]:.2f}, b[{i}]: {b[i]:.2f}")


# Solve the system of equations (using Gaussian elimination for tridiagonal matrix)
# Find the modified coefficients for the tridiagonal system
final_phi = np.zeros(n_nodes, dtype=complex)
for i in range(1, n_nodes):
    a[i] = a[i] - ( c[i-1]**2 / a[i-1] )
    b[i] = b[i] - ((c[i-1] * b[i-1]) / a[i-1])
    print(f"coefficients; i={i}: a[i]={a[i]}, b[i]={b[i]}")

# Backwards Gaussian Elimination
final_phi[-1] = b[-1] / a[-1]
for i in range(n_nodes-2, 0, -1):
    final_phi[i] = (b[i] - c[i] * final_phi[i+1]) / a[i]

# final_phi = np.linalg.solve(np.diag(a) + np.diag(c, 1) + np.diag(c, -1), b)

# Check coefficients and phi
for i in range(n_nodes):
    if i < n_nodes - 1: 
        print(f"K[{i}]: {a[i]:.2f}, c[{i}]: {c[i]:.2f}, b[{i}]: {b[i]:.2f}")
    print(f"Phi at node {i}: {final_phi[i]:.2f}")

# Model the exact solution for comparison and interpolate final_phi for graphing
exact_phi = np.zeros(n_exact_phi_resolution, dtype=complex)
for i in range(n_exact_phi_resolution):
    x = i * h / (n_exact_phi_resolution / n_elements)
    exact_phi[i] = x + (-1/2) * x**2
resized_final_phi = np.interp(
    np.linspace(0, len(final_phi) - 1, n_exact_phi_resolution),
    np.arange(len(final_phi)),
    final_phi,
)

# Calc Residual, Avg Error, and Error till element
residual = np.abs(exact_phi - resized_final_phi) / np.abs(exact_phi)
residual = np.nan_to_num(residual)
error_till_e = np.zeros(n_exact_phi_resolution, dtype=float)
# print(f"Sum of residuals: {residual}")
for i in range(n_exact_phi_resolution):
    # print(f"Residual at element {i}: {residual[i]:.6f}")
    error_till_e[i] = np.sum(residual[:(i+1)]) / n_nodes
    # print(f"Error till element {i}: {error_till_e[i]:.6f}")
error = np.sum(residual) / n_nodes
print(f"Average Error: {error:.6f}")



### Plotting
# FEM Solution vs Exact Solution
x = np.linspace(0, L, n_exact_phi_resolution) 
plt.figure(figsize=(10, 5))
plt.plot(x, np.abs(resized_final_phi), 'b-', linewidth=1.5)
plt.plot(x, np.abs(exact_phi), 'g--', linewidth=1.5)
plt.legend(['FEM Solution', 'Exact Solution'])
plt.xlabel('Domain Length (element * e_length)')
plt.ylabel('Output')
plt.title('1D FEM Solution vs Exact Solution: -1/2*x^2 + x')
plt.grid(True)
plt.show()

# Error vs Element
e = np.linspace(0, n_exact_phi_resolution, n_exact_phi_resolution)
plt.figure(figsize=(10, 5))
plt.plot(e, error_till_e, 'r-', linewidth=1.5)
plt.xlabel('Element')
plt.ylabel('Error')
plt.title('Residual Error per Element')
plt.grid(True)
plt.show()

# Error vs h decreasing in log scale
h_decreasing = np.linspace(L, 0, n_exact_phi_resolution)
plt.figure(figsize=(10, 5))
plt.plot(h_decreasing, error_till_e, 'm-', linewidth=1.5)
plt.yscale('log')
plt.gca().invert_xaxis()
plt.xlabel('Domain Length (decreasing)')
plt.ylabel('Error')
plt.title('Residual Error vs Domain Length (decreasing)')
plt.grid(True)
plt.show()
