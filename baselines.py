import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from pact import _sort_summary_for_lambda_selection


def storey_pi0(pvals, lam):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if not 0 <= lam < 1:
        raise ValueError("lambda must satisfy 0 <= lambda < 1.")
    return (1.0 + np.sum(pvals > lam)) / (n * (1.0 - lam))


def storey_bh_reject(pvals, lam, alpha):
    pi_0 = storey_pi0(pvals, lam)
    rej = multipletests(pvals, alpha=alpha / pi_0, method="fdr_bh")[0]
    rej = np.asarray(rej).reshape(-1).astype(int)
    return (pi_0, rej)


def storey_pi0_robust(pvals, lam):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    pi0 = storey_pi0(pvals, lam)
    var_hat = 1.0 / n * pi0 * (1.0 / (1.0 - lam) - pi0)
    var_hat = max(var_hat, 0.0)
    return pi0 + np.sqrt(var_hat)


def make_lambda_grid(q, delta, lam_max=1.0):
    if not 0 < q < 1:
        raise ValueError("q must satisfy 0 < q < 1.")
    if not q < lam_max <= 1:
        raise ValueError("lam_max must satisfy q < lam_max <= 1.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    lam_max_eff = min(lam_max, 1.0 - 1e-12)
    grid = [q]
    current = q + delta
    while current <= lam_max_eff + 1e-15:
        grid.append(current)
        current += delta
    if len(grid) == 1:
        grid.append(lam_max_eff)
    return np.asarray(grid, dtype=float)


def choose_lambda_step_down(pvals, q, delta=None, lam_max=1.0, robust=False):
    pvals = np.asarray(pvals, dtype=float)
    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("All p-values must be in [0, 1].")
    if delta is None:
        tail_count = np.sum(pvals > q)
        if tail_count == 0:
            delta = lam_max - q
        else:
            delta = 50.0 / tail_count
        if delta <= 0 or delta > lam_max - q:
            delta = lam_max - q
    grid = make_lambda_grid(q=q, delta=delta, lam_max=lam_max)
    estimator = storey_pi0_robust if robust else storey_pi0
    prev_val = estimator(pvals, grid[0])
    for j in range(1, len(grid)):
        cur_val = estimator(pvals, grid[j])
        if prev_val <= cur_val:
            return grid[j]
        prev_val = cur_val
    return grid[-1]


def adaptive_storey_bh_gao(pvals, q=0.05, delta=None, lam_max=1.0, robust=False):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("All p-values must be in [0, 1].")
    lambda_hat = choose_lambda_step_down(
        pvals=pvals, q=q, delta=delta, lam_max=lam_max, robust=robust
    )
    pi0_hat = storey_pi0(pvals, lambda_hat)
    p_sorted = np.sort(pvals)
    p_candidates = p_sorted[p_sorted < q]
    if len(p_candidates) == 0:
        reject = np.zeros(n, dtype=bool)
        return (pi0_hat, reject)
    k = np.arange(1, len(p_candidates) + 1)
    estimated_fdp = pi0_hat * n * p_candidates / k
    ok = estimated_fdp <= q
    if not np.any(ok):
        reject = np.zeros(n, dtype=bool)
    else:
        tau_hat = p_candidates[np.where(ok)[0][-1]]
        reject = pvals <= tau_hat
    return (pi0_hat, reject)


def by_2006_adaptive_bh(pvals, q=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("All p-values must be in [0, 1].")
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1).")
    q_reduced = q / (1.0 + q)
    reject_stage1 = multipletests(pvals, alpha=q_reduced, method="fdr_bh")[0]
    R1 = int(np.sum(reject_stage1))
    pi0_hat = (1.0 + q) * (1.0 - R1 / n)
    if R1 == n:
        reject = np.ones(n, dtype=bool)
        pi0_hat = 0.0
        return (pi0_hat, reject)
    alpha_eff = q / pi0_hat
    alpha_eff = min(alpha_eff, 1.0 - 1e-12)
    reject = multipletests(pvals, alpha=alpha_eff, method="fdr_bh")[0]
    return (pi0_hat, reject)


def run_greedy_ams(
    *, p_test, candi_lam, alpha, tie_break="larger_lambda", tie_break_random_state=None
):
    records = []
    for idx, lam in enumerate(candi_lam):
        pi_0_cur, rej_cur = storey_bh_reject(pvals=p_test, lam=lam, alpha=alpha)
        records.append(
            {
                "idx": idx,
                "lambda": float(lam),
                "pi_0": pi_0_cur,
                "Num_Rej": int(np.sum(rej_cur)),
            }
        )
    summary = pd.DataFrame(records)
    summary = _sort_summary_for_lambda_selection(
        summary=summary,
        primary_col="Num_Rej",
        primary_ascending=False,
        tie_break=tie_break,
        tie_break_random_state=tie_break_random_state,
    )
    lam_sel = float(summary.iloc[0]["lambda"])
    pi_0, rej = storey_bh_reject(pvals=p_test, lam=lam_sel, alpha=alpha)
    return {
        "pi_0": pi_0,
        "rej": rej,
        "Selected_lambda": lam_sel,
        "selection_summary": summary,
    }
