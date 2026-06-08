import moreau
import cvxpy as cp
import numpy as np
from scipy import sparse


# ==================================================================
# 0. Problem data (3-food, 3-nutrient illustrative example)
# ==================================================================
food_names     = ['Bread', 'Milk', 'Eggs']
nutrient_names = ['Calories', 'Protein', 'Fat']

c = np.array([0.10, 0.20, 0.15])        # unit cost
A = np.array([
    [3.0, 1.5, 2.0],                    # Calories
    [1.0, 3.0, 2.5],                    # Protein
    [0.5, 2.0, 3.5],                    # Fat
])
b = np.array([6.0, 5.0, 4.0])           # minimum requirements
m, n = A.shape

print("=" * 60)
print("Stigler Diet Problem: Moreau vs CVXPY")
print("=" * 60)
print(f"Dims : m={m} nutrients, n={n} foods")
print(f"Cost : c = {c}")
print(f"Req. : b = {b}")


# ==================================================================
# 1. CVXPY reference solution
# ==================================================================
print("\n" + "=" * 60)
print("Part 1: CVXPY reference solution")
print("=" * 60)

x_cp   = cp.Variable(n, nonneg=True)
con_ax = (A @ x_cp >= b)
prob   = cp.Problem(cp.Minimize(c @ x_cp), [con_ax])
prob.solve(solver=cp.CLARABEL)

x_cvxpy   = x_cp.value
lam_cvxpy = con_ax.dual_value          # dual for Ax>=b, dim=m
nu_cvxpy  = c - A.T @ lam_cvxpy       # dual for x>=0 from stationarity

print(f"\nStatus : {prob.status}")
print(f"Primal obj c^T x* = {prob.value:.6f}")
print(f"\nPrimal solution x*:")
for i, name in enumerate(food_names):
    print(f"  x*[{name}] = {x_cvxpy[i]:.6f}")
print(f"\nDual lambda* (for Ax>=b):")
for i, name in enumerate(nutrient_names):
    print(f"  lambda*[{name}] = {lam_cvxpy[i]:.6f}")
print(f"\nDual obj b^T lambda* = {b @ lam_cvxpy:.6f}")


# ==================================================================
# 2. Moreau solution
# ==================================================================
print("\n" + "=" * 60)
print("Part 2: Moreau solution")
print("=" * 60)

P_qp   = sparse.csr_matrix((n, n))     # P=0 for LP
q_qp   = c.copy()


# Constraint matrix: stack [-I; -A], rhs: [0; -b]

A_cone = sparse.vstack(
    [-sparse.eye(n, format='csr'),
     -sparse.csr_matrix(A)], format='csr')
b_cone = np.concatenate([np.zeros(n), -b])
cones  = moreau.Cones(num_nonneg_cones=n + m)

solver  = moreau.Solver(P_qp, q_qp, A_cone, b_cone,
                        cones=cones,
                        settings=moreau.Settings(enable_grad=True))
sol     = solver.solve()

x_mor   = sol.x
mu_mor  = sol.z[:n]                    # dual for x>=0
lam_mor = sol.z[n:]                    # dual for Ax>=b
nu_mor  = c - A.T @ lam_mor

print(f"\nStatus : {solver.info.status}  (1 = optimal)")
print(f"Primal obj c^T x* = {c @ x_mor:.6f}")
print(f"\nPrimal solution x*:")
for i, name in enumerate(food_names):
    print(f"  x*[{name}] = {x_mor[i]:.6f}")
print(f"\nDual lambda* (for Ax>=b):")
for i, name in enumerate(nutrient_names):
    print(f"  lambda*[{name}] = {lam_mor[i]:.6f}")
print(f"\nDual obj b^T lambda* = {b @ lam_mor:.6f}")


# ==================================================================
# 3. KKT verification (Moreau solution)
# ==================================================================
print("\n" + "=" * 60)
print("KKT verification (Moreau)")
print("=" * 60)

ax_b   = A @ x_mor - b
cs_x   = np.abs(x_mor * mu_mor)
cs_lam = np.abs(lam_mor * ax_b)
dual_feas = np.abs(A.T @ lam_mor + nu_mor - c)

print(f"\nPrimal feasibility Ax*-b >= 0 : {ax_b.round(6)}")
print(f"x* >= 0                       : {x_mor.round(6)}")
print(f"Dual feasibility A^T lam+nu-c : {dual_feas.round(8)}")
print(f"lambda* >= 0                  : {lam_mor.round(6)}")
print(f"nu*     >= 0                  : {np.abs(nu_mor).round(6)}")
print(f"CS  x* o mu*                  : {cs_x.round(8)}")
print(f"CS  lambda* o (Ax*-b)         : {cs_lam.round(8)}")
print(f"Duality gap                   : {abs(c@x_mor - b@lam_mor):.2e}")


# ==================================================================
# 4. Backward: sensitivity dCost/db = lambda*
# ==================================================================
print("\n" + "=" * 60)
print("Backward: sensitivity dCost/db")
print("=" * 60)

# dL/dx for L(x)=c^T x
dl_dx = c.copy()

# Implicit differentiation through solver:
# compute gradients of optimal value w.r.t. problem data
grads = solver.backward(dl_dx, np.zeros(n+m), np.zeros(n+m))

# Recover dCost/db:
# b_cone = [0; -b], so dCost/db = -dCost/db_cone[n:]
dCost_db = -grads['db'][n:]            # chain rule: b_cone[n:] = -b

print(f"\ndCost/db (Moreau backward):")
for i, name in enumerate(nutrient_names):
    print(f"  [{name}]: {dCost_db[i]:.6f}  (lambda* = {lam_mor[i]:.6f})")
print(f"\nNote: dCost/db should equal lambda*.")


# ==================================================================
# 5. Side-by-side comparison
# ==================================================================
print("\n" + "=" * 60)
print("Comparison: Moreau vs CVXPY")
print("=" * 60)

W = 14
print(f"\n  {'Metric':<32} {'CVXPY':>{W}} {'Moreau':>{W}}")
print("  " + "-" * (32 + 2*W + 2))
print(f"  {'Primal obj c^T x*':<32} {prob.value:>{W}.6f} {c@x_mor:>{W}.6f}")
print(f"  {'Dual   obj b^T lam*':<32} {b@lam_cvxpy:>{W}.6f} {b@lam_mor:>{W}.6f}")
print(f"  {'Duality gap':<32} {abs(prob.value-b@lam_cvxpy):>{W}.2e} {abs(c@x_mor-b@lam_mor):>{W}.2e}")

print(f"\n  {'x* (primal)':<32} {'CVXPY':>{W}} {'Moreau':>{W}}")
print("  " + "-" * (32 + 2*W + 2))
for i, name in enumerate(food_names):
    print(f"  x*[{name}]{'':24} {x_cvxpy[i]:>{W}.6f} {x_mor[i]:>{W}.6f}")

print(f"\n  {'lambda* (dual for Ax>=b)':<32} {'CVXPY':>{W}} {'Moreau':>{W}}")
print("  " + "-" * (32 + 2*W + 2))
for i, name in enumerate(nutrient_names):
    print(f"  lam*[{name}]{'':20} {lam_cvxpy[i]:>{W}.6f} {lam_mor[i]:>{W}.6f}")

print(f"\n  {'Max |x* diff|':<32} {np.abs(x_cvxpy-x_mor).max():>{W}.2e}")
print(f"  {'Max |lam* diff|':<32} {np.abs(lam_cvxpy-lam_mor).max():>{W}.2e}")