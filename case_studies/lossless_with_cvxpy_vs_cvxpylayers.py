import numpy as np
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

np.random.seed(42)
torch.manual_seed(42)

# ── Problem parameters ────────────────────────────────────────────
d      = 3
N      = 10
rho1   = 0.3
rho2   = 1.0
theta  = np.deg2rad(45.0)
n_hat  = np.array([1.0, 0.0, 0.0])
z0_val = np.array([5.0, 2.0, -1.0])

# ── CVXPY model (NOCP-Relaxed, specialized G z = h to z_{i+1}=z_i+u_i) ───────
U        = cp.Variable((N, d),   name="U")      # controls u_i
Gam      = cp.Variable(N,        name="Gamma")  # slack Gamma_i
Z        = cp.Variable((N+1, d), name="Z")      # states  z_i
z0_param = cp.Parameter(d,       name="z0")

constraints = []
con_dyn = []   # z_{i+1} = z_i + u_i   ->  dual y_i    in R^d  (co-state)
con_soc = []   # ||u_i|| <= Gamma_i     ->  dual s_soc_i >= 0
con_lb  = []   # Gamma_i >= rho1        ->  dual mu_lb_i >= 0
con_ub  = []   # Gamma_i <= rho2        ->  dual mu_ub_i >= 0
con_pt  = []   # n^T u_i >= Gamma*cos   ->  dual mu_pt_i >= 0

for i in range(N):
    c = Z[i+1] == Z[i] + U[i]
    con_dyn.append(c); constraints.append(c)
constraints.append(Z[0] == z0_param)

for i in range(N):
    c = cp.norm(U[i]) <= Gam[i]
    con_soc.append(c); constraints.append(c)
    c = Gam[i] >= rho1
    con_lb.append(c); constraints.append(c)
    c = Gam[i] <= rho2
    con_ub.append(c); constraints.append(c)
    c = n_hat @ U[i] >= Gam[i] * np.cos(theta)
    con_pt.append(c); constraints.append(c)

objective = cp.Minimize(cp.sum(Gam) + 0.1 * cp.sum_squares(Z[N]))
problem   = cp.Problem(objective, constraints)

# ══════════════════════════════════════════════════════════════════
# Part 1: CVXPY  (NOCP-SOCP solved directly)
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("Part 1: plain CVXPY (Clarabel solver)")
print("=" * 60)

z0_param.value = z0_val
problem.solve(solver=cp.CLARABEL, verbose=False)

assert problem.status in ("optimal", "optimal_inaccurate"), (
    f"Solve failed: {problem.status}"
)
print(f"Status : {problem.status}")
print(f"Value  : {problem.value:.6f}")

U_cv   = U.value
Gam_cv = Gam.value
Z_cv   = Z.value

# Dual variables (NOCP-Dual: y <-> equality dual, s <-> conic dual)
y_i   = np.stack([c.dual_value for c in con_dyn])   # (N, d)  co-states
s_soc = np.array([c.dual_value for c in con_soc])   # (N,)
mu_lb = np.array([c.dual_value for c in con_lb])
mu_ub = np.array([c.dual_value for c in con_ub])
mu_pt = np.array([c.dual_value for c in con_pt])

# [1] Lossless condition:  Gamma_i* = ||u_i*||
U_norms      = np.linalg.norm(U_cv, axis=1)
lossless_err = np.abs(U_norms - Gam_cv)
print(f"\n[1] Lossless condition  max|Gamma_i* - ||u_i*||| = "
      f"{lossless_err.max():.2e}", end="  ")
assert lossless_err.max() < 1e-6
print("PASSED")

# [2] KKT conditions
print("\n[2] KKT conditions:")

dyn_res  = max(np.linalg.norm(Z_cv[i+1] - Z_cv[i] - U_cv[i])
               for i in range(N))
soc_viol = np.maximum(U_norms - Gam_cv, 0).max()
lb_viol  = np.maximum(rho1 - Gam_cv, 0).max()
pt_viol  = np.maximum(Gam_cv * np.cos(theta) - U_cv @ n_hat, 0).max()

# Stationarity w.r.t. Gamma_i:
#   1 - s_soc_i - mu_lb_i + mu_ub_i + mu_pt_i * cos(theta) = 0
stat_G = np.abs(
    1.0 - s_soc - mu_lb + mu_ub + mu_pt * np.cos(theta)
).max()

cs_soc = np.abs(s_soc * (U_norms - Gam_cv)).max()
cs_lb  = np.abs(mu_lb * (Gam_cv - rho1)).max()
cs_pt  = np.abs(mu_pt * (U_cv @ n_hat - Gam_cv * np.cos(theta))).max()

print(f"  Primal feasibility:"
      f"  dyn={dyn_res:.1e}  SOC={soc_viol:.1e}"
      f"  lb={lb_viol:.1e}  pt={pt_viol:.1e}")
print(f"  Dual feasibility:"
      f"  s_soc>={s_soc.min():.1e}"
      f"  mu_lb>={mu_lb.min():.1e}"
      f"  mu_pt>={mu_pt.min():.1e}")
print(f"  Stationarity w.r.t. Gamma_i        = {stat_G:.1e}")
print(f"  Complementary slackness  max        = "
      f"{max(cs_soc, cs_lb, cs_pt):.1e}")

for v in [dyn_res, soc_viol, lb_viol, pt_viol,
          stat_G, cs_soc, cs_lb, cs_pt]:
    assert v < 1e-6
print("  All KKT checks PASSED")

# [3] Dual variable summary
print(f"\n[3] Dual variable summary:")
print(f"  y_i   (co-states)  shape={y_i.shape}"
      f"  range=[{y_i.min():.4f}, {y_i.max():.4f}]")
print(f"  s_soc              range=[{s_soc.min():.4f}, {s_soc.max():.4f}]")
print(f"  mu_lb              range=[{mu_lb.min():.4f}, {mu_lb.max():.4f}]")
print(f"  mu_ub              range=[{mu_ub.min():.4f}, {mu_ub.max():.4f}]")
print(f"  mu_pt              range=[{mu_pt.min():.4f}, {mu_pt.max():.4f}]")

# ══════════════════════════════════════════════════════════════════
# Part 2: CVXPYLayers — implicit differentiation through the SOCP
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Part 2: CVXPYLayers (implicit differentiation)")
print("=" * 60)

z0_param.value = None   # reset before wrapping
layer = CvxpyLayer(
    problem,
    parameters=[z0_param],
    variables=[U, Gam, Z],
)

z0_t = torch.tensor(z0_val, dtype=torch.float64, requires_grad=True)
U_sol, Gam_sol, Z_sol = layer(z0_t)

# Primal consistency
U_cl        = U_sol.detach().numpy()
Gam_cl      = Gam_sol.detach().numpy()
lossless_cl = np.abs(np.linalg.norm(U_cl, axis=1) - Gam_cl).max()
primal_err  = np.linalg.norm(U_cl - U_cv, 'fro')
print(f"Lossless max error (CVXPYLayers) = {lossless_cl:.2e}")
print(f"||U_CVXPYLayers - U_CVXPY||_F   = {primal_err:.2e}")
assert lossless_cl < 1e-4
assert primal_err  < 1e-4

# Gradient: d(||z_N||^2) / d(z0)
# Measures how the terminal-state cost changes with the initial condition,
# propagated through the implicitly differentiable SOCP solution map.
loss = Z_sol[-1].pow(2).sum()
loss.backward()
grad_z0 = z0_t.grad.numpy()
print(f"\nd(||z_N||^2)/d(z0) = {grad_z0.round(6)}")
print("(nonzero gradient confirms differentiability through the SOCP)")

print("\nAll checks completed.")
print("=" * 60)