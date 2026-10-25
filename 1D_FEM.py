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

# Initialize problem parameters
M = 50  # Number of elements
L = 1.0  # Length of the domain
h = L / M  # Element length

# Boundary condition type (1 for Dirichlet, 2 for Neumann, 3 for 3rd Kind)
bc = 1
p = 0
q = 0
gamma = 0

epsilon_r = complex(4.0, 0.1)
mu_r = complex(2.0, 0.0)  

num_nodes = M + 1

# Initialize element properties
alpha = np.zeros(M)
beta = np.zeros(M)
function = np.zeros(M)
length = np.zeros(M)

# Read in element properties
for i in range(M):
    length[i] = h*i
    alpha[i] = 1*i  # Example initialization
    beta[i] = 1*i   # Example initialization
    function[i] = 1
    # if i <= M/2:
    #     function[i] = .1*i  # Example initialization
    # else:
    #     function[i] = .1*(M-i)  # Example initialization


# # Print Initialization of element properties
# print("Element properties:")
# for i in range(M):
#     print(f"Element {i}: length = {length[i]}, alpha = {alpha[i]}, beta = {beta[i]}, function = {function[i]}")
# print()


# Initialize element matrices
a = np.zeros(num_nodes) # Main diagonal of the tridiagonal matrix
c = np.zeros(num_nodes - 1) # Upper diagonal of the tridiagonal matrix
b = np.zeros(num_nodes) # Right-hand side vector
fin_function = np.zeros(M)

# Assemble element matrices
for element in range(M):
    a[element] += 1 / h
    a[element + 1] += 1 / h
    c[element] -= 1 / h

    b[element] += h / 2
    b[element + 1] += h / 2

# Prepare the system of equations
for i in range(M):
    # print(f"Initializing element matrix for i={i}")
    if (bc != 1 and (i == 0 or i == M)):
        a[i] += alpha[i] * length[i] + beta[i] * length[i] / 3
        b[i] += function[i] * length[i] / 2
    else:
        a[i] += alpha[i] * length[i] + beta[i] * length[i] / 3 + a[i-1]
        b[i] += function[i] * length[i] / 2 + b[i-1]
    if i != M: c[i] += alpha[i] * length[i] + beta[i] * length[i] / 6


# Apply Boundry Conditions
# if bc == 1:
#     # Apply Dirichlet boundary condition
#     a[0] = 1.0
#     b[0] = p
#     # for i in range(1, M):
#     #     b[i] = b[i] - p*c[i]
#     #     c[i] = 0
# elif bc == 2:
#     # Apply Neumann boundary condition
#     pass 
# elif bc == 3:
#     # Apply 3rd Kind boundary condition
#     for i in range(M):
#         a[i] += gamma
#         b[i] += q

# Solve the system of equations (using Gaussian elimination for tridiagonal matrix)
# Find the modified coefficients for the tridiagonal system
# for i in range(1, M):
#     a[i] += a[i] - c[i-1]* c[i-1] / a[i-1]
#     b[i] += b[i] - c[i-1] * b[i-1] / a[i-1]
#     print(f"coefficients; i={i}: a[i]={a[i]}, b[i]={b[i]}")


# Test: Poisson equation enforcement
fin_function = np.linalg.solve(np.diag(a) + np.diag(c, k=1) + np.diag(c, k=-1), b)

# Backwards Gaussian Elimination
# for i in range(M-1, 0, -1):
#     print(f"Backwards elimination; i={i}: a[i]={a[i]}, b[i]={b[i]}, c[i]={c[i]}, fin_function[i+1]={fin_function[i+1] if i < M-1 else 'N/A'}")
#     if i == M-1: 
#         fin_function[i] = b[i] / a[i]
#     else:
#         fin_function[i] = (b[i] - c[i] * fin_function[i+1]) / a[i]


# Print the final solution
print("Final solution:")
for i in range(M):
    print(f"Element {i}: {fin_function[i]}")

# Test against poisson equation
num_elements = 50
x, computed, exact = poisson.solve_poisson(num_elements)
print(num_elements, np.max(np.abs(computed - exact)))
print(num_elements, np.max(np.abs(fin_function - exact)))
print(np.max(np.abs(exact)))


# Graph the final solution
plt.plot(range(M+1), fin_function, marker='o')
plt.plot(range(M+1), fin_function, marker='o')
plt.xlabel('Element')
plt.ylabel('Reflection')
# plt.ylim(0,1)
plt.title('1D FEM Solution')
plt.grid(True)
plt.show()