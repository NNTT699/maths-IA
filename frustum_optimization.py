import numpy as np
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = 100

def volume_frustum(r1, k, A):
 
    term1 = np.pi * r1**2 * (1 + k + k**2) / 3
    
    # Check if discriminant is positive
    inner = ((A - np.pi*r1**2) / (np.pi*r1*(1+k)))**2 - r1**2*(k-1)**2
    
    if inner < 0:
        return 0 
    
    term2 = np.sqrt(inner)
    
    return term1 * term2

def height_frustum(r1, k, A):

    inner = ((A - np.pi*r1**2) / (np.pi*r1*(1+k)))**2 - r1**2*(k-1)**2
    if inner < 0:
        return 0
    return np.sqrt(inner)

# Partial derivatives
def partial_l_r1(r1, k, A):
    """Partial derivative of l with respect to r1"""
    numerator = (-np.pi**2 * k**4 * r1**4 + 2*np.pi**2 * k**2 * r1**4 - 
                 A**2 * np.pi * r1**2 * (k+1)**2)
    denominator_sq = (np.pi**2 * k**2 * r1**4 - np.pi**2 * k**4 * r1**4 - 
                      2*np.pi*A*r1**2 + A**2)
    
    if denominator_sq <= 0:
        return 0
    
    return numerator / (2 * np.sqrt(denominator_sq))

def partial_l_k(r1, k, A):
    #Partial derivative of l with respect to k
    term1 = -np.pi**2 * r1**4 * (k-1) * (k+1)**3 + (A - np.pi*r1**2)**2
    denominator_sq = (2*np.pi**2*r1**4*k**2 - np.pi**2*r1**4*k**4 - 
                      2*np.pi*A*r1**2 + A**2)
    
    if denominator_sq <= 0:
        return 0
    
    term2 = np.pi * r1 * (k+1)**2
    
    return term1 / (2 * np.sqrt(denominator_sq) * term2)

def equations(vars):
    #system of equations
    r1, k = vars
    
    if r1 <= 0 or k <= 1:
        return [1e10, 1e10]
    
    l = height_frustum(r1, k, A)
    
    if l == 0:
        return [1e10, 1e10]
    
    dl_dr1 = partial_l_r1(r1, k, A)
    dl_dk = partial_l_k(r1, k, A)
    
    eq1 = 2*l + r1 * dl_dr1
    eq2 = (2*k + 1)*l + (1 + k + k**2) * dl_dk
    
    return [eq1, eq2]

# Method 1: Solve using partial derivatives
print("="*60)
print("METHOD 1: PARTIAL DIFFERENTIATION")
print("="*60)
print(f"Fixed Surface Area A = {A} cm²\n")

# Try multiple initial guesses
initial_guesses = [
    [2.0, 1.5],
    [3.0, 1.3],
    [2.5, 1.7],
    [1.5, 2.0],
    [4.0, 1.2]
]

solutions = []
for guess in initial_guesses:
    try:
        sol = fsolve(equations, guess, full_output=True)
        r1_opt, k_opt = sol[0]
        info = sol[1]
        
        # Check if solution is valid
        if r1_opt > 0 and k_opt > 1:
            # Check if equations are satisfied
            residual = np.linalg.norm(equations([r1_opt, k_opt]))
            if residual < 1e-6:
                V_opt = volume_frustum(r1_opt, k_opt, A)
                if V_opt > 0:
                    solutions.append((r1_opt, k_opt, V_opt, residual))
    except:
        continue

# Find best solution
if solutions:
    best_sol = max(solutions, key=lambda x: x[2])
    r1_opt, k_opt, V_opt, residual = best_sol
    
    r2_opt = k_opt * r1_opt
    h_opt = height_frustum(r1_opt, k_opt, A)
    
    print(f"Optimal base radius r₁ = {r1_opt:.4f} cm")
    print(f"Optimal top radius r₂ = {r2_opt:.4f} cm")
    print(f"Optimal ratio k = {k_opt:.4f}")
    print(f"Optimal height h = {h_opt:.4f} cm")
    print(f"Maximum Volume V = {V_opt:.4f} cm³")
    print(f"Residual error = {residual:.2e}")
    
    # Verify surface area
    s = np.sqrt(h_opt**2 + (r2_opt - r1_opt)**2)
    A_check = np.pi*r1_opt**2 + np.pi*s*(r1_opt + r2_opt)
    print(f"Surface area check: {A_check:.4f} cm² (should be {A})")
else:
    print("No valid solution found with partial derivatives")

print("\n" + "="*60)
print("METHOD 2: OPTIMIZATION (Numerical)")
print("="*60)

# Method 2: Direct optimization using minimize
def objective(vars):
    """Negative volume (to maximize, we minimize the negative)"""
    r1, k = vars
    if r1 <= 0 or k <= 1:
        return 1e10
    V = volume_frustum(r1, k, A)
    return -V  #Negative because we want to maximize

from scipy.optimize import minimize

result = minimize(objective, x0=[2.5, 1.5], 
                  method='L-BFGS-B',
                  bounds=[(0.1, 10), (1.01, 5)])

if result.success:
    r1_opt2, k_opt2 = result.x
    V_opt2 = -result.fun
    r2_opt2 = k_opt2 * r1_opt2
    h_opt2 = height_frustum(r1_opt2, k_opt2, A)
    
    print(f"Optimal base radius r₁ = {r1_opt2:.4f} cm")
    print(f"Optimal top radius r₂ = {r2_opt2:.4f} cm")
    print(f"Optimal ratio k = {k_opt2:.4f}")
    print(f"Optimal height h = {h_opt2:.4f} cm")
    print(f"Maximum Volume V = {V_opt2:.4f} cm³")
else:
    print("Optimization failed")

print("\n" + "="*60)
print("COMPARISON WITH CYLINDER")
print("="*60)

r_cyl = np.sqrt(A / (6*np.pi))
h_cyl = 2 * r_cyl
V_cyl = np.pi * r_cyl**2 * h_cyl

print(f"Cylinder optimal radius r = {r_cyl:.4f} cm")
print(f"Cylinder optimal height h = {h_cyl:.4f} cm")
print(f"Cylinder maximum volume V = {V_cyl:.4f} cm³")

if 'V_opt' in locals():
    print(f"\nFrustum volume / Cylinder volume = {V_opt/V_cyl:.4f}")
    print(f"Volume difference = {V_opt - V_cyl:.4f} cm³")

