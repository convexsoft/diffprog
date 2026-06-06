import time
import numpy as np
import torch
from scipy import sparse
import moreau


# ==================================================================
# 0. Utilities
# ==================================================================
def evaluate_all(v, data):
    G = data["G"]
    Gv = G @ v
    p = v * Gv
    f = torch.dot(v, Gv)

    V_lower, V_upper = data["V_lower"], data["V_upper"]
    v_viol_low = torch.clamp(V_lower - v, min=0.0)
    v_viol_up  = torch.clamp(v - V_upper, min=0.0)
    max_v_viol = torch.max(torch.cat([v_viol_low, v_viol_up])).item()

    load_idx, gen_idx = data["load_idx"], data["gen_idx"]
    d, p_cap = data["d"], data["p_cap"]
    load_viol = torch.clamp(d[load_idx] - p[load_idx], min=0.0)
    gen_viol  = torch.clamp(p[gen_idx] - p_cap[gen_idx], min=0.0)
    max_load_viol = load_viol.max().item() if load_viol.numel() else 0.0
    max_gen_viol  = gen_viol.max().item()  if gen_viol.numel()  else 0.0

    edges_i, edges_j = data["edges_i"], data["edges_j"]
    g_ij, I_max = data["g_ij"], data["I_max"]
    current   = g_ij * (v[edges_i] - v[edges_j])
    abs_cur   = torch.abs(current)
    line_viol = torch.clamp(abs_cur - I_max, min=0.0)
    max_line_viol = line_viol.max().item() if line_viol.numel() else 0.0

    return {
        "f": f.item(), "p": p.detach().clone(),
        "abs_current": abs_cur.detach().clone(),
        "max_v_viol": max_v_viol, "max_load_viol": max_load_viol,
        "max_gen_viol": max_gen_viol, "max_line_viol": max_line_viol,
    }


def print_stats(tag, stats, lam=None, gam=None, mu=None):
    print(f"[{tag}] f={stats['f']:.6f}, "
          f"V_viol={stats['max_v_viol']:.2e}, "
          f"load_viol={stats['max_load_viol']:.2e}, "
          f"gen_viol={stats['max_gen_viol']:.2e}, "
          f"line_viol={stats['max_line_viol']:.2e}")
    if lam is not None:
        lam_np = lam.detach().cpu().numpy() if torch.is_tensor(lam) else np.array(lam)
        gam_np = gam.detach().cpu().numpy() if torch.is_tensor(gam) else np.array(gam)
        mu_np  = mu.detach().cpu().numpy()  if torch.is_tensor(mu)  else np.array(mu)
        print(f"       lambda (load dual) = {lam_np.round(4)}")
        print(f"       gamma  (gen  dual) = {gam_np.round(4)}")
        print(f"       mu     (line dual) = {mu_np.round(4)}")


# ==================================================================
# 1. Test case
# ==================================================================
def build_test_case(N=5, V_min=0.9, V_max=1.2, seed=0):
    torch.manual_seed(seed)
    edges_i = torch.arange(N - 1, dtype=torch.long)
    edges_j = torch.arange(1, N,   dtype=torch.long)
    E   = N - 1
    g_ij = 0.5 + torch.rand(E)

    G = torch.zeros(N, N)
    for k in range(E):
        i, j, g = edges_i[k].item(), edges_j[k].item(), g_ij[k].item()
        G[i,i] += g; G[j,j] += g; G[i,j] -= g; G[j,i] -= g

    V_lower = torch.full((N,), V_min)
    V_upper = torch.full((N,), V_max)

    num_load = N // 2
    load_idx = torch.arange(num_load, dtype=torch.long)
    gen_idx  = torch.arange(num_load, N, dtype=torch.long)

    d     = torch.zeros(N)
    p_cap = torch.zeros(N)
    d[load_idx]    = 0.01 + 0.04 * torch.rand(len(load_idx))
    p_cap[gen_idx] = 1.0  + torch.rand(len(gen_idx))
    I_max = 1.0 + 0.5 * torch.rand(E)

    return {"N": N, "nodes": torch.arange(N),
            "edges_i": edges_i, "edges_j": edges_j, "E": E,
            "g_ij": g_ij, "G": G,
            "V_lower": V_lower, "V_upper": V_upper,
            "load_idx": load_idx, "gen_idx": gen_idx,
            "d": d, "p_cap": p_cap, "I_max": I_max}


# ==================================================================
# 2. Primal-Dual Gradient (PyTorch)
# ==================================================================
def primal_dual_solve(data, num_iters=500, tau=0.1, eta=0.1, verbose=True):
    N = data["N"]
    V_lower, V_upper = data["V_lower"], data["V_upper"]
    load_idx, gen_idx = data["load_idx"], data["gen_idx"]
    edges_i, edges_j = data["edges_i"], data["edges_j"]
    g_ij, I_max, G   = data["g_ij"], data["I_max"], data["G"]

    v   = (V_lower + V_upper) / 2.0
    lam = torch.zeros(N)
    gam = torch.zeros(N)
    mu  = torch.zeros(data["E"])

    def grad_L(v, lam, gam, mu):
        Gv     = G @ v
        grad_f = 2.0 * Gv
        a      = torch.zeros_like(v)
        a[load_idx] = -lam[load_idx]
        a[gen_idx]  =  gam[gen_idx]
        grad_p = (a * (G @ v)) + (G @ (a * v))
        vi, vj = v[edges_i], v[edges_j]
        sgn    = torch.sign(vi - vj)
        w      = mu * g_ij * sgn
        grad_line = torch.zeros_like(v)
        grad_line.scatter_add_(0, edges_i,  w)
        grad_line.scatter_add_(0, edges_j, -w)
        return grad_f + grad_p + grad_line

    history = {"f": [], "lambda": [], "gamma": [], "mu": []}
    for k in range(num_iters):
        f = torch.dot(v, G @ v)
        with torch.no_grad():
            v = torch.clamp(v - tau * grad_L(v, lam, gam, mu),
                            V_lower, V_upper)
            Gv  = G @ v
            p   = v * Gv
            lam_new = lam.clone()
            lam_new[load_idx] = torch.clamp(
                lam[load_idx] + eta * (data["d"][load_idx] - p[load_idx]), min=0.0)
            gam_new = gam.clone()
            gam_new[gen_idx] = torch.clamp(
                gam[gen_idx] + eta * (p[gen_idx] - data["p_cap"][gen_idx]), min=0.0)
            cur = g_ij * (v[edges_i] - v[edges_j])
            mu  = torch.clamp(mu + eta * (torch.abs(cur) - I_max), min=0.0)
            lam, gam = lam_new, gam_new

        history["f"].append(f.item())
        history["lambda"].append(lam.clone())
        history["gamma"].append(gam.clone())
        history["mu"].append(mu.clone())

        if verbose and (k % 100 == 0 or k == num_iters - 1):
            s = evaluate_all(v.detach(), data)
            print(f"  Iter {k:4d}: f={s['f']:.6f}, "
                  f"V_viol={s['max_v_viol']:.2e}, "
                  f"load_viol={s['max_load_viol']:.2e}, "
                  f"gen_viol={s['max_gen_viol']:.2e}, "
                  f"line_viol={s['max_line_viol']:.2e}")

    return v.detach(), lam, gam, mu, history


# ==================================================================
# 3. CVXPY-SCA
# ==================================================================
def _build_weighted_incidence(data):
    N = int(data["N"]); E = int(data["E"])
    ei = data["edges_i"].numpy().astype(int)
    ej = data["edges_j"].numpy().astype(int)
    g  = data["g_ij"].numpy()
    A  = np.zeros((E, N))
    w  = np.sqrt(g)
    for k in range(E):
        A[k, ei[k]] = +w[k]; A[k, ej[k]] = -w[k]
    return A

def _build_Pi_matrices(G_torch):
    G = G_torch.numpy(); N = G.shape[0]
    P = []
    for i in range(N):
        Pi = np.zeros((N, N)); Pi[i,i] = G[i,i]
        for j in range(N):
            if j != i and G[i,j] != 0.0:
                Pi[i,j] = 0.5*G[i,j]; Pi[j,i] = 0.5*G[i,j]
        P.append(Pi)
    return P

def cvxpy_baseline_sca(data, v0=None, num_sca_iters=40,
                        rho=1e-2, solver="OSQP", verbose=False):
    import cvxpy as cp
    N = int(data["N"])
    V_lower = data["V_lower"].numpy(); V_upper = data["V_upper"].numpy()
    load_idx = data["load_idx"].numpy().astype(int)
    gen_idx  = data["gen_idx"].numpy().astype(int)
    d = data["d"].numpy(); p_cap = data["p_cap"].numpy()
    ei = data["edges_i"].numpy().astype(int)
    ej = data["edges_j"].numpy().astype(int)
    g_ij = data["g_ij"].numpy(); I_max = data["I_max"].numpy()
    E = int(data["E"])
    A_inc  = _build_weighted_incidence(data)
    P_list = _build_Pi_matrices(data["G"])
    v_prev = ((V_lower + V_upper) / 2.0) if v0 is None else v0.numpy().copy()

    last_prob = None
    for t in range(num_sca_iters):
        v   = cp.Variable(N)
        obj = cp.sum_squares(A_inc @ v) + 0.5 * rho * cp.sum_squares(v - v_prev)
        cons = [v >= V_lower, v <= V_upper]
        for k in range(E):
            cons.append(cp.abs(g_ij[k]*(v[ei[k]] - v[ej[k]])) <= I_max[k])
        for i in load_idx:
            Pi = P_list[i]; p0 = float(v_prev @ Pi @ v_prev)
            grad = 2.0*(Pi @ v_prev)
            cons.append(p0 + grad @ (v - v_prev) >= d[i])
        for i in gen_idx:
            Pi = P_list[i]; p0 = float(v_prev @ Pi @ v_prev)
            grad = 2.0*(Pi @ v_prev)
            cons.append(p0 + grad @ (v - v_prev) <= p_cap[i])
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.OSQP, verbose=verbose)
        if v.value is None:
            raise RuntimeError(f"CVXPY-SCA failed at iter {t}")
        v_prev = v.value; last_prob = prob

    # Extract dual variables from last iteration
    # cons order: [v>=Vl(1), v<=Vu(1), abs_line(E), load(nL), gen(nG)]
    lam_out = np.zeros(N); gam_out = np.zeros(N)
    nL_ = len(load_idx); nG_ = len(gen_idx)
    for k, i in enumerate(load_idx):
        dv = last_prob.constraints[2 + E + k].dual_value
        lam_out[i] = float(abs(dv)) if dv is not None else 0.0
    for k, i in enumerate(gen_idx):
        dv = last_prob.constraints[2 + E + nL_ + k].dual_value
        gam_out[i] = float(abs(dv)) if dv is not None else 0.0
    mu_out = np.array([
        float(abs(last_prob.constraints[2 + k].dual_value or 0.0))
        for k in range(E)
    ])

    v_star = torch.tensor(v_prev, dtype=data["V_lower"].dtype)
    return (v_star, float(last_prob.value), last_prob.status,
            torch.tensor(lam_out), torch.tensor(gam_out), torch.tensor(mu_out))


# ==================================================================
# 4. Moreau-SCA
# ==================================================================
def moreau_sca(data, v0=None, num_sca_iters=40, rho=1e-2, verbose=False):
    """
    SCA with Moreau as the QP solver for each subproblem.

    Each SCA subproblem:
        min  (1/2) v^T Q v + q^T v
        s.t. A_cone v + s = b_cone,  s in nonneg cone

    where:
        Q = 2*(A_inc^T A_inc) + rho*I        (PSD, n x n)
        q = -rho * v_prev + linear load/gen terms
        Constraints stacked (all as Moreau nonneg convention A=-I style):
          v >= V_lower  ->  -v + s = -V_lower,  s >= 0
          v <= V_upper  ->   v + s = V_upper,   s >= 0
          g_ij(vi-vj) <= I_max   ->  g_ij(vi-vj) + s = I_max,  s>=0
         -g_ij(vi-vj) <= I_max   -> -g_ij(vi-vj) + s = I_max,  s>=0
          linearized load: grad_i^T v + s = d_i - p0_i + grad_i^T v_prev
                           with -grad form for >= constraint
          linearized gen:  grad_i^T v + s = p_cap_i - p0_i + grad_i^T v_prev

    Dual variables are extracted from sol.z at the final iteration:
        z[:N]         -> dual for v >= V_lower  (mu_v_low)
        z[N:2N]       -> dual for v <= V_upper  (mu_v_up)
        z[2N:2N+E]    -> dual for line upper
        z[2N+E:2N+2E] -> dual for line lower
        z[2N+2E:2N+2E+|L|] -> lambda (load dual)
        z[2N+2E+|L|:] -> gamma  (gen  dual)
    """
    N  = int(data["N"]); E = int(data["E"])
    V_lower = data["V_lower"].numpy(); V_upper = data["V_upper"].numpy()
    load_idx = data["load_idx"].numpy().astype(int)
    gen_idx  = data["gen_idx"].numpy().astype(int)
    d = data["d"].numpy(); p_cap = data["p_cap"].numpy()
    ei = data["edges_i"].numpy().astype(int)
    ej = data["edges_j"].numpy().astype(int)
    g_ij = data["g_ij"].numpy(); I_max = data["I_max"].numpy()
    nL = len(load_idx); nG = len(gen_idx)

    A_inc  = _build_weighted_incidence(data)   # (E x N)
    P_list = _build_Pi_matrices(data["G"])

    # QP quadratic term: Q = 2*(A_inc^T A_inc + rho/2 * I)
    # (Moreau uses (1/2)v^T Q v, so set Q = 2*A_inc^T A_inc + rho*I)
    Q_dense = 2.0 * (A_inc.T @ A_inc) + rho * np.eye(N)
    Q_dense = (Q_dense + Q_dense.T) / 2.0          # ensure symmetry
    Q_sp    = sparse.csr_matrix(Q_dense)

    v_prev = ((V_lower + V_upper) / 2.0) if v0 is None else v0.numpy().copy()

    # Fixed constraint rows (voltage box + line limits): do not change across iters
    # v >= V_lower:  -I v + s = -V_lower  (N rows)
    # v <= V_upper:  +I v + s =  V_upper  (N rows)
    # g_ij(vi-vj) <=  I_max: B v + s =  I_max  (E rows)
    # g_ij(vi-vj) >= -I_max: -B v + s =  I_max  (E rows)
    # where B[k, ei[k]] = g_ij[k], B[k, ej[k]] = -g_ij[k]

    B = np.zeros((E, N))
    for k in range(E):
        B[k, ei[k]] = +g_ij[k]; B[k, ej[k]] = -g_ij[k]

    A_fixed = np.vstack([
        -np.eye(N),          # v >= V_lower
        +np.eye(N),          # v <= V_upper
        +B,                  # line upper
        -B,                  # line lower
    ])
    b_fixed = np.concatenate([
        -V_lower,            # -v + s = -V_lower  -> s = v - V_lower >= 0
        +V_upper,            # +v + s =  V_upper  -> s = V_upper - v >= 0
        +I_max,              # B v + s = I_max    -> s = I_max - Bv >= 0
        +I_max,              # -Bv + s = I_max    -> s = I_max + Bv >= 0
    ])
    n_fixed = 2*N + 2*E
    n_load_gen = nL + nG
    n_total = n_fixed + n_load_gen

    sol_final = None
    for t in range(num_sca_iters):
        # Build linearized load/gen rows (change each iteration)
        A_lg = np.zeros((n_load_gen, N))
        b_lg = np.zeros(n_load_gen)
        q    = -rho * v_prev   # linear term from proximal

        for k, i in enumerate(load_idx):
            Pi   = P_list[i]
            p0   = float(v_prev @ Pi @ v_prev)
            grad = 2.0 * (Pi @ v_prev)
            # constraint: grad^T v >= d_i - p0 + grad^T v_prev
            # -> -grad^T v + s = -(d_i - p0 + grad^T v_prev)
            # -> s = grad^T v - d_i + p0 - grad^T v_prev >= 0
            A_lg[k, :] = -grad
            b_lg[k]    = -(d[i] - p0 + grad @ v_prev)
            # q contribution: none (linear constraint only)

        for k, i in enumerate(gen_idx):
            Pi   = P_list[i]
            p0   = float(v_prev @ Pi @ v_prev)
            grad = 2.0 * (Pi @ v_prev)
            # constraint: grad^T v <= p_cap_i - p0 + grad^T v_prev
            # -> grad^T v + s = p_cap_i - p0 + grad^T v_prev
            # -> s = p_cap_i - p0 + grad^T v_prev - grad^T v >= 0
            A_lg[nL + k, :] = +grad
            b_lg[nL + k]    = p_cap[i] - p0 + grad @ v_prev

        # Stack all constraints
        A_cone = sparse.csr_matrix(np.vstack([A_fixed, A_lg]))
        b_cone = np.concatenate([b_fixed, b_lg])
        cones  = moreau.Cones(num_nonneg_cones=n_total)

        # Linear term q = -rho * v_prev (proximal gradient)
        q_vec = -rho * v_prev

        sett   = moreau.Settings(enable_grad=False)
        solver = moreau.Solver(Q_sp, q_vec, A_cone, b_cone,
                               cones=cones, settings=sett)
        sol    = solver.solve()

        if sol.x is None:
            raise RuntimeError(f"Moreau-SCA failed at iter {t}")

        v_prev    = sol.x.copy()
        sol_final = sol

        if verbose:
            v_t = torch.tensor(v_prev, dtype=data["V_lower"].dtype)
            s   = evaluate_all(v_t, data)
            print(f"  Moreau-SCA iter {t:3d}: f={s['f']:.6f}, "
                  f"load_viol={s['max_load_viol']:.2e}, "
                  f"gen_viol={s['max_gen_viol']:.2e}, "
                  f"line_viol={s['max_line_viol']:.2e}")

    # Extract dual variables from final sol.z
    z = sol_final.z                              # shape (n_total)
    # z[:N]               -> dual for v >= V_lower (not directly lambda/gamma)
    # z[N:2N]             -> dual for v <= V_upper
    # z[2N:2N+E]          -> dual for line  B v <= I_max
    # z[2N+E:2N+2E]       -> dual for line -B v <= I_max
    # z[2N+2E:2N+2E+nL]   -> lambda* (load, for -grad v <= -(d-p0-grad*v_prev))
    # z[2N+2E+nL:]        -> gamma*  (gen,  for  grad v <=  p_cap-p0+grad*v_prev)

    lam_out = np.zeros(N)
    gam_out = np.zeros(N)
    for k, i in enumerate(load_idx):
        lam_out[i] = z[n_fixed + k]
    for k, i in enumerate(gen_idx):
        gam_out[i] = z[n_fixed + nL + k]

    # Line dual: take max of upper/lower cone duals
    mu_out = np.maximum(z[2*N : 2*N+E], z[2*N+E : 2*N+2*E])

    v_star = torch.tensor(v_prev, dtype=data["V_lower"].dtype)
    return (v_star, float(np.dot(Q_dense @ v_prev, v_prev) / 2.0
                          + np.dot(q_vec, v_prev)),
            torch.tensor(lam_out), torch.tensor(gam_out), torch.tensor(mu_out))


# ==================================================================
# 5. Main
# ==================================================================
if __name__ == "__main__":
    data = build_test_case(N=5, V_min=0.9, V_max=1.2, seed=0)

    # ── Method 1: Primal-Dual Gradient (PyTorch) ──────────────────
    print("=" * 60)
    print("Method 1: Primal-Dual Gradient (PyTorch)")
    print("=" * 60)
    t0 = time.time()
    v_pd, lam_pd, gam_pd, mu_pd, history = primal_dual_solve(
        data, num_iters=8000, tau=0.06, eta=0.06, verbose=True)
    t_pd = time.time() - t0
    stats_pd = evaluate_all(v_pd, data)
    print(f"\n[PDG] Final:")
    print_stats("PDG", stats_pd, lam_pd, gam_pd, mu_pd)
    print(f"      Time: {t_pd:.2f}s")

    # ── Method 2: CVXPY-SCA ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Method 2: CVXPY-SCA")
    print("=" * 60)
    t0 = time.time()
    v_csca, obj_csca, status_csca, lam_csca, gam_csca, mu_csca = cvxpy_baseline_sca(
        data, v0=v_pd, num_sca_iters=30, rho=1e-2, verbose=False)
    t_csca = time.time() - t0
    stats_csca = evaluate_all(v_csca, data)
    print(f"[CVXPY-SCA] status: {status_csca}")
    print_stats("CVXPY-SCA", stats_csca, lam_csca, gam_csca, mu_csca)
    print(f"      Time: {t_csca:.2f}s")

    # ── Method 3: Moreau-SCA ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Method 3: Moreau-SCA")
    print("=" * 60)
    t0 = time.time()
    v_msca, obj_msca, lam_msca, gam_msca, mu_msca = moreau_sca(
        data, v0=v_pd, num_sca_iters=30, rho=1e-2, verbose=True)
    t_msca = time.time() - t0
    stats_msca = evaluate_all(v_msca, data)
    print(f"\n[Moreau-SCA] Final:")
    print_stats("Moreau-SCA", stats_msca, lam_msca, gam_msca, mu_msca)
    print(f"      Time: {t_msca:.2f}s")

    # ── Comparison table ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Comparison: PDG vs CVXPY-SCA vs Moreau-SCA")
    print("=" * 60)
    W = 14
    print(f"\n  {'Metric':<24} {'PDG':>{W}} {'CVXPY-SCA':>{W}} {'Moreau-SCA':>{W}}")
    print("  " + "-" * (24 + 3*W + 4))

    def r(label, a, b, c, fmt=".6f"):
        print(f"  {label:<24} {a:>{W}{fmt}} {b:>{W}{fmt}} {c:>{W}{fmt}}")

    r("Objective f(v)",
      stats_pd['f'], stats_csca['f'], stats_msca['f'])
    r("Max V violation",
      stats_pd['max_v_viol'], stats_csca['max_v_viol'], stats_msca['max_v_viol'],
      fmt=".2e")
    r("Max load violation",
      stats_pd['max_load_viol'], stats_csca['max_load_viol'], stats_msca['max_load_viol'],
      fmt=".2e")
    r("Max gen violation",
      stats_pd['max_gen_viol'], stats_csca['max_gen_viol'], stats_msca['max_gen_viol'],
      fmt=".2e")
    r("Max line violation",
      stats_pd['max_line_viol'], stats_csca['max_line_viol'], stats_msca['max_line_viol'],
      fmt=".2e")
    r("Time (s)",
      t_pd, t_csca, t_msca, fmt=".2f")

    print(f"\n  {'Dual variable':<24} {'PDG':>{W}} {'CVXPY-SCA':>{W}} {'Moreau-SCA':>{W}}")
    print("  " + "-" * (24 + 3*W + 4))
    N_ = int(data["N"])
    for i in range(N_):
        tag = "L" if i in data["load_idx"].tolist() else "G"
        if tag == "L":
            r(f"  lambda[{i}] ({tag})",
              lam_pd[i].item(), lam_csca[i].item(), lam_msca[i].item())
        else:
            r(f"  gamma [{i}] ({tag})",
              gam_pd[i].item(), gam_csca[i].item(), gam_msca[i].item())
    for k in range(int(data["E"])):
        r(f"  mu[{k}] (line {k}-{k+1})",
          mu_pd[k].item(), mu_csca[k].item(), mu_msca[k].item())

    print(f"\n  ||v_pd - v_csca|| = {torch.linalg.norm(v_pd - v_csca):.4e}")
    print(f"  ||v_pd - v_msca|| = {torch.linalg.norm(v_pd - v_msca):.4e}")
    print(f"  ||v_csca - v_msca|| = {torch.linalg.norm(v_csca - v_msca):.4e}")