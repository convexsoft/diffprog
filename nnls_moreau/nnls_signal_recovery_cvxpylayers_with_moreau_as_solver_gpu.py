import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
from pathlib import Path


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "mathtext.default": "regular",
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13
})


# ==================================================================
# 0. Shared problem data
# ==================================================================
m, n = 6, 4
x_true = np.array([0.5, 0.3, 1.2, 0.8])

rng_data = np.random.default_rng(seed=7)
P_true = rng_data.standard_normal((m, n))
b = P_true @ x_true

rng_init = np.random.default_rng(seed=42)
P_init = P_true + 0.05 * rng_init.standard_normal((m, n))

b_torch = torch.tensor(b, dtype=torch.float64)
x_true_t = torch.tensor(x_true, dtype=torch.float64)

print("=" * 65)
print("NNLS Signal Recovery: CVXPYLayers vs CVXPYLayers+Moreau")
print("=" * 65)
print(f"Problem : min_{{x>=0}} (1/2)||Px-b||^2")
print(f"Loss    : (1/2)||x* - x_true||^2")
print(f"Dims    : m={m} (observations), n={n} (signal length)")
print(f"x_true  = {x_true}")
print(f"b       = {b.round(4)}")


# ==================================================================
# 1. Build parametrized CVXPY problem
# ==================================================================
A_param = cp.Parameter((m, n))
b_param = cp.Parameter(m)
x_var = cp.Variable(n)
con_nn = (x_var >= 0)

problem = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(A_param @ x_var - b_param)),
    [con_nn]
)

print(f"\nDPP check: problem.is_dpp() = {problem.is_dpp()}")


def recover_lambda(P_np, x_np, b_np):
    return b_np - P_np @ x_np


def kkt_residuals(P_np, x_np, lam_np, b_np):
    dual_slack = P_np.T @ lam_np

    stat = P_np.T @ (P_np @ x_np - b_np) + dual_slack

    r_primal = np.linalg.norm(np.minimum(x_np, 0.0), ord=np.inf)
    r_dual = np.linalg.norm(np.minimum(dual_slack, 0.0), ord=np.inf)
    r_stat = np.linalg.norm(stat, ord=np.inf)
    r_comp = np.linalg.norm(x_np * dual_slack, ord=np.inf)

    return np.array([r_primal, r_dual, r_stat, r_comp])


# ==================================================================
# 2. Helper: run one full experiment
# ==================================================================
def run_experiment(solver_name, device_str):
    layer_kwargs = {}
    if solver_name == "MOREAU":
        layer_kwargs["solver"] = "MOREAU"

    # Only expose x* as layer output, lambda* is recovered as b - P x*.
    layer = CvxpyLayer(
        problem,
        parameters=[A_param, b_param],
        variables=[x_var],
        **layer_kwargs
    )

    b_t = b_torch.to(device_str)
    xt_t = x_true_t.to(device_str)

    # ---- Forward pass ----
    P = torch.nn.Parameter(
        torch.tensor(P_init, dtype=torch.float64).to(device_str)
    )

    x_s, = layer(P, b_t)

    P_np = P.detach().cpu().numpy()
    x_np = x_s.detach().cpu().numpy()
    lam_np = recover_lambda(P_np, x_np, b)

    primal = 0.5 * np.sum((P_np @ x_np - b) ** 2)
    #dual = -0.5 * np.dot(lam_np, lam_np) - b @ lam_np
    dual = -0.5 * np.dot(lam_np, lam_np) + b @ lam_np

    # ---- Backpropagation ----
    loss = 0.5 * (x_s - xt_t).pow(2).sum()
    loss.backward()
    dLdP = P.grad.detach().cpu().numpy().copy()

    # ---- Gradient descent on P ----
    P2 = torch.nn.Parameter(
        torch.tensor(P_init, dtype=torch.float64).to(device_str)
    )

    opt = torch.optim.Adam([P2], lr=0.02)
    loss_history = []

    for step in range(300):
        opt.zero_grad()
        xs, = layer(P2, b_t)
        ls = 0.5 * (xs - xt_t).pow(2).sum()
        loss_history.append(ls.item())
        ls.backward()
        opt.step()

    xf, = layer(P2, b_t)
    P2_np = P2.detach().cpu().numpy()
    xf_np = xf.detach().cpu().numpy()
    lamf = recover_lambda(P2_np, xf_np, b)

    lossf = 0.5 * np.sum((xf_np - x_true) ** 2)
    steps_f = step + 1

    kkt0 = kkt_residuals(P_np, x_np, lam_np, b)

    return dict(
        x0=x_np,
        lam0=lam_np,
        primal=primal,
        dual=dual,
        loss0=loss.item(),
        dLdP=dLdP,
        xf=xf_np,
        lamf=lamf,
        kkt0=kkt0,
        lossf=lossf,
        steps=steps_f,
        loss_history=np.array(loss_history)
    )


# ==================================================================
# 3. Run both solvers
# ==================================================================
print("\n" + "=" * 65)
print("Running CVXPYLayers (CPU, diffcp backend) ...")
print("=" * 65)
res_cpu = run_experiment(solver_name=None, device_str="cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'=' * 65}")
print(f"Running CVXPYLayers + Moreau backend ({device}) ...")
print(f"{'=' * 65}")
res_mor = run_experiment(solver_name="MOREAU", device_str=device)


# ==================================================================
# 4. Print individual results
# ==================================================================
for label, res in [
    ("CVXPYLayers (CPU)", res_cpu),
    (f"CVXPYLayers+Moreau ({device})", res_mor)
]:
    print(f"\n{'=' * 65}")
    print(f"Results: {label}")
    print(f"{'=' * 65}")

    print(f"\nForward pass (initial P):")
    print(f"  Primal obj (1/2)||Px*-b||^2 = {res['primal']:.6f}")
    print(f"  Dual   obj                  = {res['dual']:.6f}")
    print(f"  Loss (1/2)||x*-x_true||^2   = {res['loss0']:.6f}")

    print(f"\n  x*      = {res['x0'].round(6)}")
    print(f"  lambda* = {res['lam0'].round(6)}  [residual dual, dim=m]")

    print(f"\n  dLoss/dP =\n{res['dLdP'].round(6)}")

    print(f"\nAfter gradient descent ({res['steps']} steps, lr=0.02):")
    print(f"  Final Loss = {res['lossf']:.8f}")
    print(f"  x* (final) = {res['xf'].round(6)}")
    print(f"  lambda* (final) = {res['lamf'].round(6)}")


# ==================================================================
# 5. Side-by-side comparison table
# ==================================================================
print(f"\n{'=' * 65}")
print("Side-by-side Comparison")
print(f"{'=' * 65}")

W = 22


def row(label, vc, vm):
    print(f"  {label:<36} {vc:>{W}.6f} {vm:>{W}.6f}")


print(f"\n  {'--- Forward pass ---'}")
print(f"  {'Metric':<36} {'CVXPYLayers(CPU)':>{W}} {'Moreau backend':>{W}}")
print("  " + "-" * (36 + 2 * W + 2))
row("Primal obj (1/2)||Px*-b||^2", res_cpu["primal"], res_mor["primal"])
row("Dual   obj", res_cpu["dual"], res_mor["dual"])
row("Loss (1/2)||x*-x_true||^2", res_cpu["loss0"], res_mor["loss0"])

print(f"\n  {'x* (primal solution)':<36} {'CVXPYLayers(CPU)':>{W}} {'Moreau backend':>{W}}")
print("  " + "-" * (36 + 2 * W + 2))
for i in range(n):
    row(f"x*[{i}]", res_cpu["x0"][i], res_mor["x0"][i])

print(f"\n  {'lambda* (residual dual, dim=m)':<36} {'CVXPYLayers(CPU)':>{W}} {'Moreau backend':>{W}}")
print("  " + "-" * (36 + 2 * W + 2))
for i in range(m):
    row(f"lambda*[{i}]", res_cpu["lam0"][i], res_mor["lam0"][i])

print(f"\n  {'dLoss/dP[i,j]':<36} {'CVXPYLayers(CPU)':>{W}} {'Moreau backend':>{W}}")
print("  " + "-" * (36 + 2 * W + 2))
for i in range(m):
    for j in range(n):
        row(f"dP[{i},{j}]", res_cpu["dLdP"][i, j], res_mor["dLdP"][i, j])

print(f"\n  {'--- After gradient descent ---'}")
print(f"  {'Metric':<36} {'CVXPYLayers(CPU)':>{W}} {'Moreau backend':>{W}}")
print("  " + "-" * (36 + 2 * W + 2))
row("Steps to convergence", float(res_cpu["steps"]), float(res_mor["steps"]))
row("Final Loss", res_cpu["lossf"], res_mor["lossf"])

print(f"\n  {'x* (final)':<36} {'CVXPYLayers(CPU)':>{W}} {'Moreau backend':>{W}}")
print("  " + "-" * (36 + 2 * W + 2))
for i in range(n):
    row(f"x*[{i}]", res_cpu["xf"][i], res_mor["xf"][i])

print(f"\n  {'lambda* (final, residual dual, dim=m)':<36} {'CVXPYLayers(CPU)':>{W}} {'Moreau backend':>{W}}")
print("  " + "-" * (36 + 2 * W + 2))
for i in range(m):
    row(f"lambda*[{i}]", res_cpu["lamf"][i], res_mor["lamf"][i])


# ==================================================================
# 6. Generate four figures
# ==================================================================
out_dir = Path("figures_nnls_moreau")
out_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


# --------------------------------------------------
# Figure 1: Solution difference
# --------------------------------------------------
x_err = np.abs(res_cpu["x0"] - res_mor["x0"])

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(n), x_err, marker="o", linewidth=2, markersize=7)

ax.set_xlabel("Signal Index")
ax.set_ylabel(r"$|x^*_{\rm CVXPYLayers}-x^*_{\rm Moreau}|$")
ax.set_title("Solution Difference")
ax.set_xticks(np.arange(n))
ax.set_xticklabels([rf"$x_{i+1}$" for i in range(n)])
ax.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(out_dir / "nnls_signal_recovery_solution_difference.png", dpi=300)
fig.savefig(out_dir / "nnls_signal_recovery_solution_difference.pdf")
plt.close(fig)


# --------------------------------------------------
# Figure 2: KKT residual comparison
# --------------------------------------------------
labels = ["Primal", "Dual", "Stationarity", "Complementarity"]
eps = 1e-16

cpu_vals = res_cpu["kkt0"] + eps
mor_vals = res_mor["kkt0"] + eps

fig, ax = plt.subplots(figsize=(6.5, 4))

ax.semilogy(
    np.arange(len(labels)),
    cpu_vals,
    marker="o",
    linewidth=2,
    markersize=7,
    label="CVXPYLayers"
)

ax.semilogy(
    np.arange(len(labels)),
    mor_vals,
    marker="s",
    linewidth=2,
    markersize=7,
    label="CVXPYLayers+Moreau"
)

ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel("KKT Condition")
ax.set_ylabel("Residual")
ax.set_title(r"KKT Residuals with $\lambda^*=b-Px^*$")
ax.legend()
ax.grid(True, which="both", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(out_dir / "nnls_signal_recovery_kkt_residual_comparison.png", dpi=300)
fig.savefig(out_dir / "nnls_signal_recovery_kkt_residual_comparison.pdf")
plt.close(fig)


# --------------------------------------------------
# Figure 3: Gradient consistency map
# --------------------------------------------------
grad_err = np.abs(res_cpu["dLdP"] - res_mor["dLdP"])

fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(grad_err, aspect="auto")

ax.set_xlabel("Column Index of P")
ax.set_ylabel("Row Index of P")
ax.set_title(
    r"Gradient Difference "
    r"$|\nabla_P L_{\rm CVXPYLayers}-"
    r"\nabla_P L_{\rm Moreau}|$"
)

ax.set_xticks(np.arange(n))
ax.set_xticklabels([rf"$j={j+1}$" for j in range(n)])
ax.set_yticks(np.arange(m))
ax.set_yticklabels([rf"$i={i+1}$" for i in range(m)])

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Absolute Gradient Difference")

fig.tight_layout()
fig.savefig(out_dir / "nnls_signal_recovery_gradient_difference_heatmap.png", dpi=300)
fig.savefig(out_dir / "nnls_signal_recovery_gradient_difference_heatmap.pdf")
plt.close(fig)


# --------------------------------------------------
# Figure 4: Loss descent curve
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(res_cpu["loss_history"], marker="o", markevery=10, label="CVXPYLayers")
ax.plot(res_mor["loss_history"], marker="s", markevery=10, label="CVXPYLayers+Moreau")

ax.set_xlabel("Gradient Step")
ax.set_ylabel(r"Reconstruction Loss $L(P)$")
ax.set_title("Loss Descent Curve")
ax.set_yscale("log")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(out_dir / "nnls_signal_recovery_loss_descent_curve.png", dpi=300)
fig.savefig(out_dir / "nnls_signal_recovery_loss_descent_curve.pdf")
plt.close(fig)

print(f"\nFigures saved to: {out_dir.resolve()}")