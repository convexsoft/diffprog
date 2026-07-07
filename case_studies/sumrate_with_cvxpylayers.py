import time
import numpy as np
import cvxpy as cp
from scipy.linalg import eig
import torch
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt
import os


plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})


# ============================================================
# 0. Problem generation
# ============================================================
def generate_problem(L=4, K=2, seed=1):
    rng = np.random.default_rng(seed)

    G = rng.uniform(0.005, 0.03, size=(L, L))
    np.fill_diagonal(G, rng.uniform(1.5, 2.0, size=L))

    n = 0.1 * np.ones(L)
    w = rng.uniform(0.5, 1.5, size=L)
    w = w / np.sum(w)

    F = np.zeros((L, L))
    for l in range(L):
        for j in range(L):
            if l != j:
                F[l, j] = G[l, j] / G[l, l]

    v = n / np.diag(G)

    Aweights = [np.ones(L)]
    pbar     = [2.0]
    if K >= 2:
        Aweights.append(rng.uniform(0.5, 1.5, size=L))
        pbar.append(1.5)
    for _ in range(2, K):
        Aweights.append(rng.uniform(0.5, 1.5, size=L))
        pbar.append(rng.uniform(1.0, 2.0))

    Aweights = np.array(Aweights)
    pbar     = np.array(pbar)

    Btilde_list = []
    for k in range(K):
        Bk     = F + np.outer(v, Aweights[k]) / pbar[k]
        Btilde = np.linalg.solve(np.eye(L) + Bk, Bk)
        Btilde = np.maximum(Btilde, 1e-12)
        Btilde_list.append(Btilde)

    rbar_vec = np.zeros(L)
    for l in range(L):
        ub = max(pbar[k] / (Aweights[k, l] * v[l]) for k in range(K))
        rbar_vec[l] = np.log(1.0 + ub)

    return G, F, v, w, Aweights, pbar, Btilde_list, rbar_vec


# ============================================================
# 1. Utility functions
# ============================================================
def spectral_radius(M):
    vals = eig(M, left=False, right=False)
    return float(np.max(np.abs(vals)))


def recover_power(r, F, v):
    L  = len(r)
    Er = np.diag(np.exp(r))
    q  = np.linalg.solve(np.eye(L) + F - F @ Er, v)
    p  = np.diag(np.exp(r) - 1.0) @ q
    return p, q


def check_spectral(r, Btilde_list):
    return np.array([spectral_radius(B @ np.diag(np.exp(r)))
                     for B in Btilde_list])


def check_row_sums(r, s, Btilde_list):
    K, L = s.shape[0], len(r)
    row_sums = np.zeros((K, L))
    for k, B in enumerate(Btilde_list):
        for i in range(L):
            row_sums[k, i] = sum(
                B[i, j] * np.exp(r[j] + s[k, j] - s[k, i])
                for j in range(L) if B[i, j] > 0
            )
    return row_sums


# ============================================================
# 2. CVXPY reference solver
# ============================================================
def solve_cvxpy_pf(Btilde_list, w, rbar, theta=None, s_bound=20.0):
    """
    theta: (K, L) array of row-sum constraint RHS.
           Default is all-ones (standard Collatz-Wielandt).
    """
    L = len(w)
    K = len(Btilde_list)

    if theta is None:
        theta = np.ones((K, L))

    r = cp.Variable(L)
    s = cp.Variable((K, L))

    constraints  = [r >= 0, r <= rbar]
    for k in range(K):
        constraints += [s[k, 0] == 0]
    constraints += [s <= s_bound, s >= -s_bound]

    row_sum_cons = []
    for k, B in enumerate(Btilde_list):
        for i in range(L):
            t_vars = []
            for j in range(L):
                if B[i, j] > 0:
                    t  = cp.Variable(nonneg=True)
                    xe = np.log(B[i, j]) + r[j] + s[k, j] - s[k, i]
                    constraints.append(cp.ExpCone(xe, 1.0, t))
                    t_vars.append(t)
            c = cp.sum(cp.hstack(t_vars)) <= float(theta[k, i])
            constraints.append(c)
            row_sum_cons.append(c)

    prob = cp.Problem(cp.Maximize(w @ r), constraints)
    t0   = time.time()
    prob.solve(solver=cp.CLARABEL, verbose=False)
    elapsed = time.time() - t0

    dual_mu = np.array(
        [c.dual_value if c.dual_value is not None else np.nan
         for c in row_sum_cons]
    ).reshape(K, L)

    return {
        "r":          np.array(r.value).reshape(-1),
        "s":          np.array(s.value),
        "dual_mu":    dual_mu,
        "objective":  float(prob.value),
        "solve_time": elapsed,
        "status":     prob.status,
    }


# ============================================================
# 3. CVXPYLayers: parametrize w and theta
# ============================================================
def build_cvxpylayers(Btilde_list, L, K, s_bound=20.0):
    """
    Build a CvxpyLayer with two differentiable parameters:
        w_param   : (L,)   objective weight vector
        theta_param: (K*L,) row-sum constraint RHS (flattened)

    Variables: r (L,), s (K, L)
    """
    w_param     = cp.Parameter(L,   name="w")
    theta_param = cp.Parameter(K * L, name="theta")

    r = cp.Variable(L)
    s = cp.Variable((K, L))

    rbar_param = cp.Parameter(L, name="rbar")

    constraints = [r >= 0, r <= rbar_param]
    for k in range(K):
        constraints += [s[k, 0] == 0]
    constraints += [s <= s_bound, s >= -s_bound]

    for k, B in enumerate(Btilde_list):
        for i in range(L):
            t_vars = []
            for j in range(L):
                if B[i, j] > 0:
                    t  = cp.Variable(nonneg=True)
                    xe = np.log(B[i, j]) + r[j] + s[k, j] - s[k, i]
                    constraints.append(cp.ExpCone(xe, 1.0, t))
                    t_vars.append(t)
            # theta_param[k*L + i] is the RHS for row (k, i)
            c = cp.sum(cp.hstack(t_vars)) <= theta_param[k * L + i]
            constraints.append(c)

    prob  = cp.Problem(cp.Maximize(w_param @ r), constraints)
    layer = CvxpyLayer(
        prob,
        parameters=[w_param, theta_param, rbar_param],
        variables=[r, s]
    )
    return layer


# ============================================================
# 4. Main
# ============================================================
if __name__ == "__main__":
    L, K, seed = 4, 2, 3

    G, F, v, w, Aweights, pbar, Btilde_list, rbar_vec = generate_problem(
        L=L, K=K, seed=seed
    )

    print("=" * 65)
    print("Sum-Rate Maximization via CVXPYLayers")
    print("=" * 65)
    print(f"L={L} links, K={K} power constraints, seed={seed}")
    print(f"\nWeights  w    = {w.round(4)}")
    print(f"Budgets  pbar = {pbar}")
    print(f"Rate UB  rbar = {rbar_vec.round(4)}")
    for k, B in enumerate(Btilde_list):
        status = "OK" if B.min() >= 0 else "WARNING: negative entries"
        print(f"Btilde[{k}]: min={B.min():.2e}, max={B.max():.2e}  [{status}]")

    # ----------------------------------------------------------
    # Part 1: CVXPY reference
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("Part 1: CVXPY Reference Solution (Clarabel)")
    print("=" * 65)

    cvx = solve_cvxpy_pf(Btilde_list, w, rbar_vec)

    r_ref   = cvx["r"]
    s_ref   = cvx["s"]
    mu_ref  = cvx["dual_mu"]
    p_ref, q_ref = recover_power(r_ref, F, v)
    rho_ref = check_spectral(r_ref, Btilde_list)
    row_ref = check_row_sums(r_ref, s_ref, Btilde_list)

    print(f"\nStatus    : {cvx['status']}")
    print(f"Objective w^T r* = {cvx['objective']:.6f}")
    print(f"Solve time       = {cvx['solve_time']:.4f}s")
    print(f"\nOptimal rate vector r*:")
    for l in range(L):
        print(f"  r*[{l}] = {r_ref[l]:.6f}")
    print(f"\nRecovered power vector p*:")
    for l in range(L):
        print(f"  p*[{l}] = {p_ref[l]:.6f}")
    print(f"\nSpectral radius rho(Btilde_k diag(exp(r*))):")
    for k in range(K):
        print(f"  k={k}: rho = {rho_ref[k]:.8f}  (<=1: {'OK' if rho_ref[k]<=1+1e-6 else 'VIOLATED'})")
    print(f"\nRow-sum verification sum_j Btilde[i,j] exp(r_j+s_j-s_i):")
    print(f"  {'(k,i)':<8} {'row_sum':>12} {'theta':>8} {'status':>10}")
    for k in range(K):
        for i in range(L):
            status = "active" if abs(row_ref[k, i] - 1.0) < 1e-4 else "slack"
            print(f"  ({k},{i}){'':4} {row_ref[k,i]:>12.6f} {'1.0':>8} {status:>10}")
    print(f"\nDual variables mu*[k,l] (row-sum constraints):")
    print(f"  {'(k,l)':<8} {'mu*':>12}")
    for k in range(K):
        for l in range(L):
            print(f"  ({k},{l}){'':4} {mu_ref[k,l]:>12.6f}")

    # ----------------------------------------------------------
    # Part 2: CVXPYLayers forward pass
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("Part 2: CVXPYLayers Forward Pass")
    print("=" * 65)

    layer = build_cvxpylayers(Btilde_list, L, K)

    print(f"\nDPP check: passed (layer built successfully)")

    w_t     = torch.nn.Parameter(torch.tensor(w,                         dtype=torch.float64))
    theta_t = torch.nn.Parameter(torch.tensor(np.ones(K * L),            dtype=torch.float64))
    rbar_t  = torch.tensor(rbar_vec, dtype=torch.float64)

    r_t, s_t = layer(w_t, theta_t, rbar_t)

    r_layer = r_t.detach().numpy()
    s_layer = s_t.detach().numpy()

    print(f"\nCVXPYLayers solution r*:")
    for l in range(L):
        print(f"  r*[{l}] = {r_layer[l]:.6f}")

    print(f"\nConsistency with CVXPY reference:")
    print(f"  Max |r*(CVXPYLayers) - r*(CVXPY)| = {np.abs(r_layer - r_ref).max():.4e}")

    # ----------------------------------------------------------
    # Part 3a: Backward — d(w^T r*)/dw = r*
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("Part 3a: Backward Pass — Envelope Theorem d(w^T r*)/dw = r*")
    print("=" * 65)

    # Re-run forward to get fresh computation graph
    w_t2     = torch.nn.Parameter(torch.tensor(w,            dtype=torch.float64))
    theta_t2 = torch.tensor(np.ones(K * L),                  dtype=torch.float64)
    r_t2, _  = layer(w_t2, theta_t2, rbar_t)

    obj = w_t2 @ r_t2
    obj.backward()

    dobj_dw = w_t2.grad.detach().numpy()

    print(f"\n  Objective V* = w^T r* = {obj.item():.6f}")
    print(f"\n  {'l':<6} {'dV*/dw[l]':>14} {'r*[l]':>14} {'diff':>14}")
    print("  " + "-" * 50)
    for l in range(L):
        diff = abs(dobj_dw[l] - r_ref[l])
        print(f"  {l:<6} {dobj_dw[l]:>14.6f} {r_ref[l]:>14.6f} {diff:>14.2e}")
    print(f"\n  Max |dV*/dw - r*| = {np.abs(dobj_dw - r_ref).max():.4e}")

    # ----------------------------------------------------------
    # Part 3b: Backward — dV*/d(theta) vs mu* (CVXPY dual)
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("Part 3b: Backward Pass — Constraint Sensitivity dV*/d(theta) vs mu*")
    print("=" * 65)

    w_t3     = torch.tensor(w,            dtype=torch.float64)
    theta_t3 = torch.nn.Parameter(torch.tensor(np.ones(K * L), dtype=torch.float64))
    r_t3, _  = layer(w_t3, theta_t3, rbar_t)

    obj3 = w_t3 @ r_t3
    obj3.backward()

    dV_dtheta = theta_t3.grad.detach().numpy().reshape(K, L)

    print(f"\n  {'(k,l)':<8} {'dV*/d(theta)':>16} {'mu* (CVXPY)':>16} {'diff':>14}")
    print("  " + "-" * 58)
    for k in range(K):
        for l in range(L):
            diff = abs(dV_dtheta[k, l] - mu_ref[k, l])
            print(f"  ({k},{l}){'':4} {dV_dtheta[k,l]:>16.6f} {mu_ref[k,l]:>16.6f} {diff:>14.2e}")
    print(f"\n  Max |dV*/d(theta) - mu*| = {np.abs(dV_dtheta - mu_ref).max():.4e}")

    # ----------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------
    W = 18
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"\n  {'Metric':<36} {'Value':>{W}}")
    print("  " + "-" * (36 + W + 1))
    print(f"  {'Objective V* = w^T r*':<36} {cvx['objective']:>{W}.6f}")
    print(f"  {'Max |r*(layer) - r*(CVXPY)|':<36} {np.abs(r_layer-r_ref).max():>{W}.4e}")
    print(f"  {'Max |dV*/dw - r*|':<36} {np.abs(dobj_dw-r_ref).max():>{W}.4e}")
    print(f"  {'Max |dV*/d(theta) - mu*|':<36} {np.abs(dV_dtheta-mu_ref).max():>{W}.4e}")
    for k in range(K):
        print(f"  {'Spectral radius k='+str(k):<36} {rho_ref[k]:>{W}.8f}")

    print(f"\n  {'l':<6} {'r*[l]':>12} {'p*[l]':>12}")
    print("  " + "-" * 32)
    for l in range(L):
        print(f"  {l:<6} {r_ref[l]:>12.6f} {p_ref[l]:>12.6f}")

    # ----------------------------------------------------------
    # Figures
    # ----------------------------------------------------------
    out_dir = "pf_sumrate_figures"
    os.makedirs(out_dir, exist_ok=True)

    x   = np.arange(L)
    kk  = np.arange(K)
    w_  = 0.35

    C1, C2 = "C0", "C1"

    # ---- Figure 1: rate vector (CVXPY vs CVXPYLayers) ----
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x - w_/2, r_ref,   w_, label="CVXPY",       color=C1, alpha=0.85)
    ax.bar(x + w_/2, r_layer, w_, label="CVXPYLayers", color=C2, alpha=0.85)
    ax.set_xlabel("Link index $l$")
    ax.set_ylabel("Optimal rate $r_l^*$")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_sumrate_rate_vector.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_sumrate_rate_vector.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved fig_sumrate_rate_vector")

    # ---- Figure 2: recovered power vector ----
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(x, p_ref, color=C1, alpha=0.85, width=0.5)
    ax.set_xlabel("Link index $l$")
    ax.set_ylabel("Recovered power $p_l^*$")
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_sumrate_power_vector.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_sumrate_power_vector.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_sumrate_power_vector")

    # ---- Figure 3: spectral radius per constraint ----
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    bars = ax.bar(kk, rho_ref, color=C1, alpha=0.85, width=0.4)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="gray", label="Limit = 1")
    ax.set_xlabel("Constraint index $k$")
    ax.set_ylabel(r"$\rho(\widetilde{\mathbf{B}}_k\,\mathrm{diag}(e^{\mathbf{r}^*}))$")
    ax.set_xticks(kk)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_sumrate_spectral_radius.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_sumrate_spectral_radius.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_sumrate_spectral_radius")

    # ---- Figure 4: envelope theorem dV*/dw vs r* ----
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(x - w_/2, r_ref,   w_, label=r"$r^*_l$ (primal)",
           color=C1, alpha=0.85)
    ax.bar(x + w_/2, dobj_dw, w_, label=r"$\partial V^*/\partial w_l$ (backprop)",
           color=C2, alpha=0.85)
    ax.set_xlabel("Link index $l$")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    diff_max = np.abs(dobj_dw - r_ref).max()
    # ax.text(0.97, 0.97, f"Max diff = {diff_max:.2e}",
    #         transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray",
    #         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_sumrate_gradient_backpropa.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig_sumrate_gradient_backpropa.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_sumrate_gradient_backpropa")
    print(f"\nAll figures saved to: {out_dir}/")