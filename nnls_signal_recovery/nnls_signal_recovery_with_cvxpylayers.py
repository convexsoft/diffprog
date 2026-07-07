import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ==================================================================
# Reproduce the experiment (same seeds, same problem)
# ==================================================================
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

m, n = 6, 4
x_true = np.array([0.5, 0.3, 1.2, 0.8])

rng_data = np.random.default_rng(seed=7)
P_true = rng_data.standard_normal((m, n))
b = P_true @ x_true

rng_init = np.random.default_rng(seed=42)
P_init = P_true + 0.05 * rng_init.standard_normal((m, n))

b_torch  = torch.tensor(b, dtype=torch.float64)
x_true_t = torch.tensor(x_true, dtype=torch.float64)

A_param = cp.Parameter((m, n))
b_param = cp.Parameter(m)
x_var   = cp.Variable(n)
problem = cp.Problem(
    cp.Minimize(0.5 * cp.sum_squares(A_param @ x_var - b_param)),
    [x_var >= 0]
)
layer = CvxpyLayer(problem, parameters=[A_param, b_param], variables=[x_var])

def recover_lambda(P_np, x_np, b_np):
    return b_np - P_np @ x_np

def kkt_residuals(P_np, x_np, lam_np, b_np):
    dual_constraint = P_np.T @ lam_np      # should be <= 0
    mu = -dual_constraint                  # should be >= 0
    stat = P_np.T @ (P_np @ x_np - b_np) - mu
    r_primal = np.linalg.norm(np.minimum(x_np, 0.0), ord=np.inf)
    r_dual   = np.linalg.norm(np.maximum(dual_constraint, 0.0), ord=np.inf)
    r_stat   = np.linalg.norm(stat, ord=np.inf)
    r_comp   = np.linalg.norm(x_np * mu, ord=np.inf)
    return np.array([r_primal, r_dual, r_stat, r_comp])


# --- forward pass ---
P = torch.nn.Parameter(torch.tensor(P_init, dtype=torch.float64))
x_s, = layer(P, b_torch)
P_np = P.detach().numpy()
x0 = x_s.detach().numpy()
lam0 = recover_lambda(P_np, x0, b)
kkt0 = kkt_residuals(P_np, x0, lam0, b)
loss = 0.5 * (x_s - x_true_t).pow(2).sum()
loss.backward()
dLdP = P.grad.detach().numpy().copy()

# --- gradient descent ---
P2 = torch.nn.Parameter(torch.tensor(P_init, dtype=torch.float64))
opt = torch.optim.Adam([P2], lr=0.02)
loss_history = []
for _ in range(300):
    opt.zero_grad()
    xs, = layer(P2, b_torch)
    ls  = 0.5 * (xs - x_true_t).pow(2).sum()
    loss_history.append(ls.item())
    ls.backward()
    opt.step()

xf,  = layer(P2, b_torch)
P2_np = P2.detach().numpy()
xf_np = xf.detach().numpy()
lamf  = recover_lambda(P2_np, xf_np, b)
kktf  = kkt_residuals(P2_np, xf_np, lamf, b)
loss_history = np.array(loss_history)


C1 = "#2166AC"   # blue  – initial
C2 = "#D6604D"   # red   – final / x_true
C3 = "#4DAC26"   # green – accent

out_dir = Path("figures_nnls_signal_recovery_with_cvxpylayers")
out_dir.mkdir(exist_ok=True)


# ==================================================================
# Figure 1 – Primal Solution: initial vs final vs x_true
# ==================================================================
idx   = np.arange(n)
width = 0.25

fig, ax = plt.subplots(figsize=(6.5, 4))
bars0 = ax.bar(idx - width, x0,    width, label=r"$x^*$ (initial $P$)", color=C1, alpha=0.85)
barsf = ax.bar(idx,         xf_np, width, label=r"$x^*$ (final $P$)",   color=C2, alpha=0.85)
barst = ax.bar(idx + width, x_true,width, label=r"$x_{\rm true}$",      color=C3, alpha=0.85,
               hatch="//", edgecolor="white", linewidth=0.5)

ax.set_xticks(idx)
ax.set_xticklabels([rf"$x_{i+1}$" for i in range(n)])
ax.set_xlabel("Signal Component")
ax.set_ylabel("Value")
ax.set_title("Primal Solution: Initial vs. Final vs. True")
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, 1.5)

fig.tight_layout()
fig.savefig(out_dir / "nnls_signal_recovery_primal_solution.png", dpi=300)
fig.savefig(out_dir / "nnls_signal_recovery_primal_solution.pdf")
plt.close(fig)
print("Saved nnls_signal_recovery_primal_solution")


# ==================================================================
# Figure 2 – Loss Descent Curve
# ==================================================================
steps = np.arange(len(loss_history))

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogy(steps, loss_history, color=C1, linewidth=2,
            marker="o", markevery=30, markersize=5)

ax.set_xlabel("Gradient Step")
ax.set_ylabel(r"Reconstruction Loss $\mathcal{L}(P)$")
ax.set_title("Loss Descent Curve (Adam, lr=0.02)")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

# annotate convergence
ax.axhline(y=loss_history[-1], color=C2, linestyle=":", linewidth=1.2,
           label=rf"Final loss = {loss_history[-1]:.2e}")
ax.legend()

fig.tight_layout()
fig.savefig(out_dir / "nnls_signal_recovery_loss_descent.png", dpi=300)
fig.savefig(out_dir / "nnls_signal_recovery_loss_descent.pdf")
plt.close(fig)
print("Saved nnls_signal_recovery_loss_descent")


# ==================================================================
# Figure 3 – KKT Residuals: initial vs final
# ==================================================================
kkt_labels_short = ["Primal\nFeasibility", "Dual\nFeasibility",
                    "Stationarity", "Complementarity"]
eps  = 1e-16
idx4 = np.arange(len(kkt_labels_short))
w    = 0.3

fig, ax = plt.subplots(figsize=(7, 4.5))
b0 = ax.bar(idx4 - w/2, kkt0 + eps, w, label="Initial $P$", color=C1, alpha=0.85)
bf = ax.bar(idx4 + w/2, kktf + eps, w, label="Final $P$",   color=C2, alpha=0.85)

ax.set_yscale("log")
ax.set_xticks(idx4)
ax.set_xticklabels(kkt_labels_short)
ax.set_ylabel("Residual (log scale)")
ax.set_title(r"KKT Residuals: Initial vs. Final $P$")
ax.legend()
ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)

# value labels on bars
for bar in list(b0) + list(bf):
    h = bar.get_height()
    if h > eps * 2:
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.5,
                f"{h:.1e}", ha="center", va="bottom", fontsize=8, rotation=0)

fig.tight_layout()
fig.savefig(out_dir / "nnls_signal_recovery_kkt_residuals.png", dpi=300)
fig.savefig(out_dir / "nnls_signal_recovery_kkt_residuals.pdf")
plt.close(fig)
print("Saved nnls_signal_recovery_kkt_residuals")


# ==================================================================
# Figure 4 – Gradient dL/dP heatmap
# ==================================================================
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(dLdP, cmap="RdBu_r", aspect="auto",
               vmin=-np.abs(dLdP).max(), vmax=np.abs(dLdP).max())

ax.set_xlabel("Column index $j$ of $P$")
ax.set_ylabel("Row index $i$ of $P$")
ax.set_title(r"Gradient $\partial\mathcal{L}/\partial P_{ij}$ at Initial $P$")

ax.set_xticks(np.arange(n))
ax.set_xticklabels([rf"$j={j+1}$" for j in range(n)])
ax.set_yticks(np.arange(m))
ax.set_yticklabels([rf"$i={i+1}$" for i in range(m)])

# annotate each cell
for i in range(m):
    for j in range(n):
        ax.text(j, i, f"{dLdP[i, j]:.3f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(dLdP[i, j]) > 0.07 else "black")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Gradient value")

fig.tight_layout()
fig.savefig(out_dir / "nnls_signal_recovery_gradient_heatmap.png", dpi=300)
fig.savefig(out_dir / "nnls_signal_recovery_gradient_heatmap.pdf")
plt.close(fig)
print("Saved nnls_signal_recovery_gradient_heatmap")

print(f"\nAll figures saved to: {out_dir.resolve()}")