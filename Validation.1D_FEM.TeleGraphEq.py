import numpy as np
import cmath

# --- Physical Parameters ---
L_inductance = 1.0e-6  
epsilon_r = 5.0
mu_r = 1.0
mu0 = 4 * np.pi * 1e-7
eps0 = 8.854187817e-12
C_capacitance = (mu_r * mu0 * epsilon_r * eps0) / L_inductance 
f = 1e9 
omega = 2 * np.pi * f

# --- Ritz Parameters ---
M = 50  
L = 1.0  
h = L / M  
num_nodes = M + 1

# The wave equation is -V'' - beta_0^2 V = 0
# So in standard Ritz form: alpha = 1, beta = -beta_0^2
beta_0 = omega * np.sqrt(L_inductance * C_capacitance)
alpha = 1.0
beta = -(beta_0**2) 

# --- Initialize Tridiagonal Matrices ---
a = np.zeros(num_nodes, dtype=complex)       # Main diagonal
c = np.zeros(num_nodes - 1, dtype=complex)   # Upper diagonal
d = np.zeros(num_nodes - 1, dtype=complex)   # Lower diagonal
b = np.zeros(num_nodes, dtype=complex)       # RHS vector

# --- Ritz Assembly ---
# Calculate local element coefficients
print(f"Alpha: {alpha}, Beta: {beta}")
K11 = (alpha / h) + (beta * h / 3)
K12 = (-alpha / h) + (beta * h / 6)

print(f"K11: {K11}, K12: {K12}")

for e in range(M):
    # Add local K11 to global node e
    a[e] += K11
    # Add local K22 (same as K11) to global node e+1
    a[e+1] += K11
    
    # Add local K12 to off-diagonals
    c[e] += K12
    d[e] += K12
    if (e % 10 == 0):  # Print every 10th element
        print(f"Element {e}: a[{e}]={a[e]}, a[{e+1}]={a[e+1]}, c[{e}]={c[e]}, d[{e}]={d[e]}")

# --- Apply Boundary Conditions ---
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

# --- Thomas Algorithm Solver ---
# Forward Sweep
for i in range(1, num_nodes):
    w = d[i-1] / a[i-1]
    a[i] = a[i] - w * c[i-1]
    b[i] = b[i] - w * b[i-1]

# Backward Substitution
final_phi = np.zeros(num_nodes, dtype=complex)
final_phi[-1] = b[-1] / a[-1]
for i in range(num_nodes - 2, -1, -1):
    final_phi[i] = (b[i] - c[i] * final_phi[i+1]) / a[i]

print(f"Voltage at x=0: {final_phi[0]:.2f}")
print(f"Voltage at x=L/2: {final_phi[M//2]:.2f}")
print(f"Voltage at x=L: {final_phi[-1]:.2f}")

# # Graph the final solution
import matplotlib.pyplot as plt

plt.plot(range(num_nodes), abs(final_phi), marker='o')
plt.xlabel('Node')
plt.ylabel('Voltage')
plt.title('1D FEM Solution')
plt.grid(True)
plt.show()