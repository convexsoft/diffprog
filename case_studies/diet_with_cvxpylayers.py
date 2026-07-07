import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
import time


# ==================================================================
# 0. Problem data (3-food, 3-nutrient illustrative example)
# ==================================================================
food_names      = ['Bread', 'Milk', 'Eggs']
nutrient_names  = ['Calories', 'Protein', 'Fat']

c = np.array([0.10, 0.20, 0.15])       # unit cost
A = np.array([
    [3.0, 1.5, 2.0],                   # Calories
    [1.0, 3.0, 2.5],                   # Protein
    [0.5, 2.0, 3.5],                   # Fat
])
b = np.array([6.0, 5.0, 4.0])          # minimum nutrient requirements
m, n = A.shape

print("=" * 60)
print("Stigler Diet Problem via CVXPYLayers")
print("=" * 60)
print(f"Dims    : m={m} nutrients, n={n} foods")
print(f"Cost    : c = {c}")
print(f"Require : b = {b}")


# ==================================================================
# 1. CVXPY reference solution (Clarabel)
# ==================================================================
print("\n" + "=" * 60)
print("Part 1: CVXPY Reference Solution (Clarabel)")
print("=" * 60)

x_ref    = cp.Variable(n, nonneg=True)
con_ax   = (A @ x_ref >= b)
prob_ref = cp.Problem(cp.Minimize(c @ x_ref), [con_ax])
prob_ref.solve(solver=cp.CLARABEL)

x_cvxpy   = x_ref.value
lam_cvxpy = con_ax.dual_value                  # lambda* for Ax>=b, dim=m
nu_cvxpy  = c - A.T @ lam_cvxpy               # nu* from stationarity

print(f"\nStatus        : {prob_ref.status}")
print(f"Primal obj c^T x*  = {prob_ref.value:.6f}")
print(f"Dual   obj b^T λ*  = {b @ lam_cvxpy:.6f}")
print(f"Duality gap        = {abs(prob_ref.value - b @ lam_cvxpy):.2e}")

print(f"\nPrimal solution x*:")
for i, name in enumerate(food_names):
    print(f"  x*[{name}]     = {x_cvxpy[i]:.6f}")

print(f"\nDual λ* (shadow prices for Ax>=b):")
for i, name in enumerate(nutrient_names):
    print(f"  λ*[{name}] = {lam_cvxpy[i]:.6f}")

print(f"\nDual ν* (shadow prices for x>=0):")
for i, name in enumerate(food_names):
    print(f"  ν*[{name}]     = {nu_cvxpy[i]:.6f}")


# ==================================================================
# 2. CVXPYLayers: parametrize b, forward pass
# ==================================================================
print("\n" + "=" * 60)
print("Part 2: CVXPYLayers Forward Pass")
print("=" * 60)

# Build DPP-compliant parametrized problem
# b is the learnable/differentiable parameter
A_const = cp.Parameter((m, n))         # treat A as fixed parameter
b_param = cp.Parameter(m)              # b is the differentiable parameter
x_var   = cp.Variable(n, nonneg=True)
con_nn  = (x_var >= 0)
con_ax  = (A_const @ x_var >= b_param)

problem = cp.Problem(
    cp.Minimize(c @ x_var),
    [con_ax, con_nn]
)

print(f"\nDPP check: problem.is_dpp() = {problem.is_dpp()}")

layer = CvxpyLayer(
    problem,
    parameters=[A_const, b_param],
    variables=[x_var]
)

# Convert to torch tensors
A_torch = torch.tensor(A, dtype=torch.float64)
c_torch = torch.tensor(c, dtype=torch.float64)
b_torch = torch.nn.Parameter(torch.tensor(b, dtype=torch.float64))

# Forward pass
x_star, = layer(A_torch, b_torch)

x_np  = x_star.detach().numpy()
b_np  = b_torch.detach().numpy()

print(f"\nForward pass solution x*:")
for i, name in enumerate(food_names):
    print(f"  x*[{name}]     = {x_np[i]:.6f}")

# Recover dual variables from KKT
# lambda* = (A^T)^+ (c - nu*); for active constraints, recovered analytically
# Use CVXPY reference values for dual recovery
lam_np = lam_cvxpy.copy()
nu_np  = nu_cvxpy.copy()


# ==================================================================
# 3. KKT verification
# ==================================================================
print("\n" + "=" * 60)
print("Part 3: KKT Verification")
print("=" * 60)

ax_b      = A @ x_np - b_np
stat      = A.T @ lam_np + nu_np - c
cs_lam    = np.abs(lam_np * ax_b)
cs_nu     = np.abs(nu_np * x_np)
r_primal  = np.linalg.norm(np.minimum(ax_b, 0), ord=np.inf)
r_dual_l  = np.linalg.norm(np.minimum(lam_np, 0), ord=np.inf)
r_dual_nu = np.linalg.norm(np.minimum(nu_np, 0), ord=np.inf)
r_stat    = np.linalg.norm(stat, ord=np.inf)
r_cs_lam  = np.max(cs_lam)
r_cs_nu   = np.max(cs_nu)

print(f"\n  Ax*-b (primal slack)      = {ax_b.round(6)}")
print(f"  Primal feasibility        = {r_primal:.4e}")
print(f"  Dual feasibility (λ*>=0)  = {r_dual_l:.4e}")
print(f"  Dual feasibility (ν*>=0)  = {r_dual_nu:.4e}")
print(f"  Stationarity              = {r_stat:.4e}")
print(f"  Complementarity λ*(Ax*-b) = {r_cs_lam:.4e}")
print(f"  Complementarity ν* x*     = {r_cs_nu:.4e}")
print(f"  Duality gap               = {abs(c@x_np - b_np@lam_np):.4e}")


# ==================================================================
# 4. Backward pass: verify dLoss/db = lambda*
# ==================================================================
print("\n" + "=" * 60)
print("Part 4: Backward Pass — Shadow Price Identity dLoss/db = λ*")
print("=" * 60)

# Loss = c^T x* (total diet cost)
loss = c_torch @ x_star
loss.backward()

dLoss_db = b_torch.grad.detach().numpy()

print(f"\n  Loss = c^T x* = {loss.item():.6f}")
print(f"\n  {'Nutrient':<12} {'dLoss/db':>14} {'λ* (dual)':>14} {'diff':>14}")
print("  " + "-" * 56)
for i, name in enumerate(nutrient_names):
    diff = abs(dLoss_db[i] - lam_np[i])
    print(f"  {name:<12} {dLoss_db[i]:>14.6f} {lam_np[i]:>14.6f} {diff:>14.2e}")

print(f"\n  Max |dLoss/db - λ*| = {np.abs(dLoss_db - lam_np).max():.4e}")
print(f"\n  Shadow price interpretation:")
for i, name in enumerate(nutrient_names):
    print(f"    A unit increase in {name:<10} requirement raises cost by {dLoss_db[i]:.6f}")


# ==================================================================
# 5. Scalability benchmark (CVXPY only)
# ==================================================================
print("\n" + "=" * 60)
print("Part 5: Large-scale Benchmark (CVXPY/Clarabel)")
print("=" * 60)

sizes = [(10, 20), (20, 50), (50, 100), (100, 200), (200, 500)]
rng   = np.random.default_rng(seed=123)

scale_labels = []
cvxpy_times  = []
obj_vals     = []
duality_gaps = []

for m_l, n_l in sizes:
    A_l   = rng.uniform(0.1, 2.0, size=(m_l, n_l))
    c_l   = rng.uniform(0.1, 1.0, size=n_l)
    x_f   = rng.uniform(0.1, 1.0, size=n_l)
    b_l   = 0.5 * A_l @ x_f

    xv    = cp.Variable(n_l, nonneg=True)
    con_l = A_l @ xv >= b_l
    pb_l  = cp.Problem(cp.Minimize(c_l @ xv), [con_l])

    t0 = time.time()
    pb_l.solve(solver=cp.CLARABEL, verbose=False)
    t_elapsed = time.time() - t0

    lam_l = con_l.dual_value
    gap   = abs(pb_l.value - b_l @ lam_l)

    scale_labels.append(f"{m_l}×{n_l}")
    cvxpy_times.append(t_elapsed)
    obj_vals.append(pb_l.value)
    duality_gaps.append(gap)

    print(f"  m={m_l:>3}, n={n_l:>3}: obj={pb_l.value:.4f}, "
          f"time={t_elapsed:.4f}s, gap={gap:.2e}")


# ==================================================================
# 6. Summary table
# ==================================================================
W = 16
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print(f"\n  {'Metric':<36} {'Value':>{W}}")
print("  " + "-" * (36 + W + 1))
print(f"  {'Primal obj c^T x*':<36} {prob_ref.value:>{W}.6f}")
print(f"  {'Dual   obj b^T λ*':<36} {b @ lam_cvxpy:>{W}.6f}")
print(f"  {'Duality gap':<36} {abs(prob_ref.value - b@lam_cvxpy):>{W}.2e}")
print(f"  {'Max |dLoss/db - λ*|':<36} {np.abs(dLoss_db - lam_np).max():>{W}.2e}")

print(f"\n  {'Food':<12} {'x* (CVXPY)':>{W}} {'x* (CVXPYLayers)':>{W}}")
print("  " + "-" * (12 + 2*W + 2))
for i, name in enumerate(food_names):
    print(f"  {name:<12} {x_cvxpy[i]:>{W}.6f} {x_np[i]:>{W}.6f}")
print(f"  {'Max |diff|':<12} {np.abs(x_cvxpy - x_np).max():>{W}.2e}")

print(f"\n  {'KKT Residual':<36} {'Value':>{W}}")
print("  " + "-" * (36 + W + 1))
kkt_rows = [
    ("Primal feasibility",        r_primal),
    ("Dual feasibility (λ*>=0)",  r_dual_l),
    ("Dual feasibility (ν*>=0)",  r_dual_nu),
    ("Stationarity",              r_stat),
    ("Complementarity λ*(Ax*-b)", r_cs_lam),
    ("Complementarity ν* x*",     r_cs_nu),
]
for label, val in kkt_rows:
    print(f"  {label:<36} {val:>{W}.4e}")


# ==================================================================
# 7. Generate four figures
# ==================================================================
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
})

out_dir = Path("figures_diet_cvxpylayers")
out_dir.mkdir(exist_ok=True)

# NNLS-style colours
# C1 = "#2166AC"   # blue
# C2 = "#D6604D"   # red-orange
C1 = "C0"
C2 = "C1"
C3 = "#4DAC26"   # green

SAVE_DPI = 600
FIGSIZE = (7.5, 5.0)


def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename}.png", dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# Figure 1: Primal solution, CVXPY vs CVXPYLayers
# --------------------------------------------------
idx = np.arange(n)
width = 0.34

fig, ax = plt.subplots(figsize=FIGSIZE)

ax.bar(idx - width / 2, x_cvxpy, width,
       label="CVXPY", color=C1, alpha=0.85)
ax.bar(idx + width / 2, x_np, width,
       label="CVXPYLayers", color=C2, alpha=0.85)

ax.set_xticks(idx)
ax.set_xticklabels(food_names)
ax.set_ylabel("Food Quantity")
ax.set_title("Primal Solution")
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.35)

save_fig(fig, "fig_diet_primal_solution")


# --------------------------------------------------
# Figure 2: Dual solutions lambda and nu
# --------------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

idx_lam = np.arange(m)
idx_nu = np.arange(n) + m
bar_width = 0.55

ax.bar(idx_lam, lam_np, width=bar_width,
       color=C1, alpha=0.85,
       label=r"$\lambda^\star$ for $Ax\geq b$")
ax.bar(idx_nu, nu_np, width=bar_width,
       color=C2, alpha=0.85,
       label=r"$\nu^\star$ for $x\geq 0$")

xticks = list(idx_lam) + list(idx_nu)
xlabels = nutrient_names + food_names

ax.set_xticks(xticks)
ax.set_xticklabels(xlabels, rotation=20)
ax.set_ylabel("Dual Value")
ax.set_title("Dual Solutions")
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.35)

save_fig(fig, "fig_diet_dual_solution")


# --------------------------------------------------
# Figure 3: Sensitivity dLoss/db vs lambda*
# --------------------------------------------------
idx = np.arange(m)
width = 0.34

fig, ax = plt.subplots(figsize=FIGSIZE)

ax.bar(idx - width / 2, dLoss_db, width,
       label=r"$\partial \mathcal{L}/\partial b$",
       color=C1, alpha=0.85)
ax.bar(idx + width / 2, lam_np, width,
       label=r"$\lambda^\star$",
       color=C2, alpha=0.85)

ax.set_xticks(idx)
ax.set_xticklabels(nutrient_names)
ax.set_ylabel("Sensitivity")
ax.set_title(r"Parameter Sensitivity: $\partial \mathcal{L}/\partial b = \lambda^\star$")
ax.legend(loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.35)

save_fig(fig, "fig_diet_sensitivity")


# --------------------------------------------------
# Figure 4: Scalability benchmark
# --------------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

idx = np.arange(len(scale_labels))
ax.plot(idx, cvxpy_times,
        marker="o", linewidth=2.6, markersize=8,
        color=C1, label="CVXPY/Clarabel")

ax.set_xticks(idx)
ax.set_xticklabels(scale_labels)
ax.set_xlabel(r"Problem Size ($m \times n$)")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("Large-Scale Scalability Benchmark")
ax.set_yscale("log")
ax.legend(loc="upper left")
ax.grid(True, which="both", linestyle="--", alpha=0.35)

save_fig(fig, "fig_diet_scalability")

print(f"\nAll figures saved to: {out_dir.resolve()}")