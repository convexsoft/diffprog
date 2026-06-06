import time
import numpy as np
import cvxpy as cp
from scipy import sparse
from scipy.linalg import eig
import moreau


# ============================================================
# 1. Problem generation
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

    # Weighted power constraints
    # k=0: total power-like constraint
    # k=1: individual-weighted aggregate constraint
    Aweights = []
    pbar = []

    Aweights.append(np.ones(L))
    pbar.append(2.0)

    if K >= 2:
        Aweights.append(rng.uniform(0.5, 1.5, size=L))
        pbar.append(1.5)

    for _ in range(2, K):
        Aweights.append(rng.uniform(0.5, 1.5, size=L))
        pbar.append(rng.uniform(1.0, 2.0))

    Aweights = np.array(Aweights)
    pbar = np.array(pbar)

    B_list = []
    Btilde_list = []
    for k in range(K):
        Bk = F + np.outer(v, Aweights[k]) / pbar[k]
        Btilde = np.linalg.solve(np.eye(L) + Bk, Bk)

        if np.min(Btilde) < -1e-9:
            print(f"Warning: Btilde[{k}] has negative entries. min={np.min(Btilde):.2e}")

        Btilde = np.maximum(Btilde, 1e-12)
        B_list.append(Bk)
        Btilde_list.append(Btilde)

    # Loose rate upper bound
    rbar = np.zeros(L)
    for l in range(L):
        ub = max(pbar[k] / (Aweights[k, l] * v[l]) for k in range(K))
        rbar[l] = np.log(1.0 + ub)

    return G, F, v, w, Aweights, pbar, B_list, Btilde_list, rbar


# ============================================================
# 2. Index map for Moreau variables
# ============================================================
class PFIndex:
    def __init__(self, L, K, Btilde_list):
        self.L = L
        self.K = K

        self.r0 = 0

        # s_{k,0} is fixed to 0 and not included as a variable
        self.s0 = L
        self.s_size = K * (L - 1)

        self.t_terms = []
        idx = self.s0 + self.s_size

        for k in range(K):
            rows_k = []
            B = Btilde_list[k]
            for i in range(L):
                terms_i = []
                for j in range(L):
                    if B[i, j] > 0:
                        terms_i.append((i, j, idx))
                        idx += 1
                rows_k.append(terms_i)
            self.t_terms.append(rows_k)

        self.nvar = idx

    def r(self, j):
        return self.r0 + j

    def s(self, k, j):
        # s_{k,0}=0 is fixed, not a decision variable
        if j == 0:
            return None
        return self.s0 + k * (self.L - 1) + (j - 1)


# ============================================================
# 3. Moreau solver
# ============================================================
def solve_moreau_pf(Btilde_list, w, rbar, s_bound=20.0):
    L = len(w)
    K = len(Btilde_list)
    idx = PFIndex(L, K, Btilde_list)
    nvar = idx.nvar

    P = sparse.csr_array((nvar, nvar))
    q_vec = np.zeros(nvar)

    # Maximize w^T r = minimize -w^T r
    for j in range(L):
        q_vec[idx.r(j)] = -w[j]

    rows = []
    b_vals = []

    def add_row(coeff_dict, b_val):
        row = np.zeros(nvar)
        for col, val in coeff_dict.items():
            row[col] = val
        rows.append(row)
        b_vals.append(float(b_val))

    # r <= rbar
    for j in range(L):
        add_row({idx.r(j): 1.0}, rbar[j])

    # r >= 0  ->  -r <= 0
    for j in range(L):
        add_row({idx.r(j): -1.0}, 0.0)

    # Optional bounds for s variables, excluding fixed s_{k,0}=0
    for k in range(K):
        for j in range(1, L):
            sj = idx.s(k, j)
            add_row({sj: 1.0}, s_bound)
            add_row({sj: -1.0}, s_bound)

    # Row-sum constraints: sum_j t_{kij} <= 1
    row_sum_start = len(rows)
    for k in range(K):
        for i in range(L):
            coeff = {}
            for _, _, t_index in idx.t_terms[k][i]:
                coeff[t_index] = 1.0
            add_row(coeff, 1.0)
    row_sum_end = len(rows)

    num_nonneg = len(rows)

    # Exponential cone constraints:
    # exp(logB_ij + r_j + s_j - s_i) <= t_ij
    num_exp = 0
    for k in range(K):
        B = Btilde_list[k]
        for i in range(L):
            for _, j, t_index in idx.t_terms[k][i]:
                logB = np.log(B[i, j])

                # s1 = logB + r_j + s_{k,j} - s_{k,i}
                # Moreau uses cone variable = b - A x
                # Choose A x = -r_j - s_j + s_i
                coeff = {idx.r(j): -1.0}

                sj = idx.s(k, j)
                si = idx.s(k, i)

                if sj is not None:
                    coeff[sj] = coeff.get(sj, 0.0) - 1.0

                if si is not None:
                    coeff[si] = coeff.get(si, 0.0) + 1.0

                add_row(coeff, logB)

                # s2 = 1
                add_row({}, 1.0)

                # s3 = t
                add_row({t_index: -1.0}, 0.0)

                num_exp += 1

    A = sparse.csr_array(np.vstack(rows))
    b = np.array(b_vals)

    cones = moreau.Cones(
        num_nonneg_cones=num_nonneg,
        num_exp_cones=num_exp,
    )

    settings = moreau.Settings(
        enable_grad=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=20000,
    )
    solver = moreau.Solver(P, q_vec, A, b, cones=cones, settings=settings)

    t0 = time.time()
    sol = solver.solve()
    elapsed = time.time() - t0

    x = sol.x
    z = sol.z

    r_val = x[idx.r0:idx.r0 + L]
    s_val = np.zeros((K, L))
    s_raw = x[idx.s0:idx.s0 + K * (L - 1)].reshape(K, L - 1)
    s_val[:, 1:] = s_raw

    dual_row_sum = z[row_sum_start:row_sum_end].reshape(K, L)

    return {
        "r": r_val,
        "s": s_val,
        "dual_row_sum": dual_row_sum,
        "objective": float(w @ r_val),
        "solve_time": elapsed,
        "info": solver.info,
        "raw_solution": sol,
    }


def check_row_sum_residual(r, s, Btilde_list):
    K = len(Btilde_list)
    L = len(r)
    row_sums = np.zeros((K, L))

    for k in range(K):
        B = Btilde_list[k]
        for i in range(L):
            val = 0.0
            for j in range(L):
                if B[i, j] > 0:
                    val += B[i, j] * np.exp(r[j] + s[k, j] - s[k, i])
            row_sums[k, i] = val

    return row_sums


# ============================================================
# 4. CVXPY solver with explicit exponential cones
# ============================================================
def solve_cvxpy_pf(Btilde_list, w, rbar, s_bound=20.0):
    L = len(w)
    K = len(Btilde_list)

    r = cp.Variable(L)
    s = cp.Variable((K, L))

    constraints = []
    constraints += [r >= 0, r <= rbar]

    for k in range(K):
        constraints += [s[k, 0] == 0]
    constraints += [s <= s_bound, s >= -s_bound]

    row_sum_constraints = []

    for k in range(K):
        B = Btilde_list[k]
        for i in range(L):
            t_terms = []
            for j in range(L):
                if B[i, j] > 0:
                    t = cp.Variable(nonneg=True)
                    x_exp = np.log(B[i, j]) + r[j] + s[k, j] - s[k, i]
                    constraints.append(cp.ExpCone(x_exp, 1.0, t))
                    t_terms.append(t)

            c = cp.sum(cp.hstack(t_terms)) <= 1.0
            constraints.append(c)
            row_sum_constraints.append(c)

    prob = cp.Problem(cp.Maximize(w @ r), constraints)

    t0 = time.time()
    prob.solve(solver=cp.CLARABEL, verbose=False)
    elapsed = time.time() - t0

    dual_row_sum = np.array(
        [c.dual_value if c.dual_value is not None else np.nan
         for c in row_sum_constraints]
    ).reshape(K, L)

    return {
        "r": np.array(r.value).reshape(-1),
        "s": np.array(s.value),
        "dual_row_sum": dual_row_sum,
        "objective": float(prob.value),
        "solve_time": elapsed,
        "status": prob.status,
    }


# ============================================================
# 5. Recovery and diagnostics
# ============================================================
def spectral_radius(M):
    vals = eig(M, left=False, right=False)
    return float(np.max(np.abs(vals)))


def recover_power_from_rate(r, F, v):
    L = len(r)
    Er = np.diag(np.exp(r))
    q = np.linalg.solve(np.eye(L) + F - F @ Er, v)
    p = np.diag(np.exp(r) - 1.0) @ q
    return p, q


def compute_sum_rate(p, F, v, w):
    q = F @ p + v
    sinr = p / q
    return float(np.sum(w * np.log(1.0 + sinr)))


def check_constraints(r, Btilde_list):
    vals = []
    for B in Btilde_list:
        A = B @ np.diag(np.exp(r))
        vals.append(spectral_radius(A))
    return np.array(vals)


def print_comparison(moreau_res, cvx_res, Btilde_list, F, v, w, Aweights, pbar):
    r_m = moreau_res["r"]
    r_c = cvx_res["r"]

    p_m, q_m = recover_power_from_rate(r_m, F, v)
    p_c, q_c = recover_power_from_rate(r_c, F, v)

    rho_m = check_constraints(r_m, Btilde_list)
    rho_c = check_constraints(r_c, Btilde_list)

    R_m = compute_sum_rate(p_m, F, v, w)
    R_c = compute_sum_rate(p_c, F, v, w)

    row_m = check_row_sum_residual(r_m, moreau_res["s"], Btilde_list)
    row_c = check_row_sum_residual(r_c, cvx_res["s"],    Btilde_list)

    K  = len(Btilde_list)
    L  = len(w)
    W  = 18
    SEP  = "  " + "-" * (34 + 2 * W + 2)

    def row(label, a, b, fmt=".8f"):
        print(f"  {label:<34} {a:>{W}{fmt}} {b:>{W}{fmt}}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Summary: Moreau vs CVXPY")
    print("=" * 72)
    print(f"\n  {'Metric':<34} {'Moreau':>{W}} {'CVXPY':>{W}}")
    print(SEP)
    row("Objective  w^T r*",     moreau_res["objective"], cvx_res["objective"])
    row("Sum-rate   R(p*)",      R_m,                     R_c)
    row("Solve time (s)",        moreau_res["solve_time"],cvx_res["solve_time"], fmt=".4f")
    for k in range(K):
        row(f"Spectral radius k={k}", rho_m[k], rho_c[k])
    for k in range(K):
        row(f"Power constr k={k}  (<={pbar[k]:.1f})",
            Aweights[k] @ p_m, Aweights[k] @ p_c)

    t_ratio = cvx_res["solve_time"] / moreau_res["solve_time"]
    print(f"\n  |objective diff|       = {abs(moreau_res['objective']-cvx_res['objective']):.2e}")
    print(f"  max |r* diff|          = {np.max(np.abs(r_m-r_c)):.2e}")
    print(f"  max |p* diff|          = {np.max(np.abs(p_m-p_c)):.2e}")
    print(f"  max |dual diff|        = "
          f"{np.nanmax(np.abs(moreau_res['dual_row_sum']-cvx_res['dual_row_sum'])):.2e}")
    print(f"  Speedup (CVXPY/Moreau) = {t_ratio:.1f}x")

    # ── Primal solution ───────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Primal Solution")
    print("=" * 72)
    print(f"\n  {'Variable':<34} {'Moreau':>{W}} {'CVXPY':>{W}}")
    print(SEP)
    for l in range(L):
        row(f"r*[{l}]", r_m[l], r_c[l])
    print(SEP)
    for l in range(L):
        row(f"p*[{l}]", p_m[l], p_c[l])
    print(SEP)
    for l in range(L):
        row(f"q*[{l}]", q_m[l], q_c[l])

    # ── Constraints ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Constraint Verification")
    print("=" * 72)

    print(f"\n  Spectral radius  rho(Btilde_k diag(e^r*)) <= 1:")
    print(f"  {'k':<10} {'rho_Moreau':>16} {'rho_CVXPY':>16} {'Viol_M':>10} {'Viol_C':>10}")
    print("  " + "-" * 64)
    for k in range(K):
        print(f"  k={k}{'':7} {rho_m[k]:>16.8f} {rho_c[k]:>16.8f} "
              f"{max(rho_m[k]-1,0):>10.2e} {max(rho_c[k]-1,0):>10.2e}")

    print(f"\n  Power constraints  a_k^T p* <= pbar_k:")
    print(f"  {'k':<10} {'Moreau':>16} {'CVXPY':>16} {'Budget':>10}")
    print("  " + "-" * 54)
    for k in range(K):
        print(f"  k={k}{'':7} {Aweights[k]@p_m:>16.8f} "
              f"{Aweights[k]@p_c:>16.8f} {pbar[k]:>10.4f}")

    print(f"\n  Row-sum verification  (sum_j Btilde[i,j] exp(r_j+s_j-s_i) <= 1):")
    print(f"  {'(k,i)':<10} {'Moreau':>16} {'CVXPY':>16} {'Status':>10}")
    print("  " + "-" * 54)
    for k in range(K):
        for i in range(L):
            status = "active" if abs(row_m[k,i]-1) < 1e-4 else "slack"
            print(f"  ({k},{i}){'':6} {row_m[k,i]:>16.8f} "
                  f"{row_c[k,i]:>16.8f} {status:>10}")

    # ── Dual variables ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Dual Variables  mu*[k,l]  (shadow prices for row-sum constraints)")
    print("=" * 72)
    print(f"  mu*[k,l] > 0  =>  row (k,l) is active at optimum")
    print(f"\n  {'(k,l)':<34} {'Moreau':>{W}} {'CVXPY':>{W}}")
    print(SEP)
    for k in range(K):
        for l in range(L):
            row(f"mu*[k={k}, l={l}]",
                moreau_res["dual_row_sum"][k,l],
                cvx_res["dual_row_sum"][k,l])



if __name__ == "__main__":
    L = 4
    K = 2
    seed = 3

    G, F, v, w, Aweights, pbar, B_list, Btilde_list, rbar = generate_problem(
        L=L, K=K, seed=seed,
    )

    print("=" * 72)
    print("Perron-Frobenius Sum-Rate Convexification: Moreau vs CVXPY")
    print("=" * 72)
    print(f"L={L}, K={K}, seed={seed}")
    print(f"\nProblem parameters:")
    print(f"  Weights w      = {w.round(4)}")
    print(f"  Power budgets  = {pbar}")
    print(f"  Rate upper bound rbar = {rbar.round(4)}")
    print(f"\nChannel gain matrix G:")
    print(np.round(G, 6))
    print(f"\nNormalized interference F:")
    print(np.round(F, 6))
    print(f"\nNormalized noise v = {v.round(6)}")
    print(f"\nPower weight vectors A:")
    print(np.round(Aweights, 6))
    print(f"\nBtilde quasi-inverse nonnegativity check:")
    for k, Btilde in enumerate(Btilde_list):
        status = "OK" if Btilde.min() >= 0 else "WARNING: negative entries"
        print(f"  k={k}: min={Btilde.min():.3e}, max={Btilde.max():.3e}  [{status}]")

    moreau_res = solve_moreau_pf(Btilde_list, w, rbar)
    cvx_res    = solve_cvxpy_pf(Btilde_list, w, rbar)

    print_comparison(moreau_res, cvx_res, Btilde_list, F, v, w, Aweights, pbar)