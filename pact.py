import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from statsmodels.stats.multitest import multipletests


def cp_vec(X_score_vec, cal_score_vec):
    cal_score_vec = np.asarray(cal_score_vec)
    X_score_vec = np.asarray(X_score_vec)
    cal_sorted = np.sort(cal_score_vec)
    n = len(cal_sorted)
    less_count = np.searchsorted(cal_sorted, X_score_vec, side="left")
    equal_count = np.searchsorted(cal_sorted, X_score_vec, side="right") - less_count
    x = (less_count + (1 + equal_count)) / (1 + n)
    return x


def evaluate_rejection(theta, rej):
    theta = np.asarray(theta).reshape(-1)
    rej = np.asarray(rej).reshape(-1).astype(int)
    num_rej = int(np.sum(rej))
    num_signal = int(np.sum(theta))
    fdp = np.sum((1 - theta) * rej) / max(num_rej, 1)
    power = np.sum(theta * rej) / max(num_signal, 1)
    return {"FDR": fdp, "Power": power, "Num_Rej": num_rej}


def _sort_summary_for_lambda_selection(
    summary,
    primary_col,
    primary_ascending,
    tie_break="larger_lambda",
    tie_break_random_state=None,
):
    allowed = {"larger_lambda", "smaller_lambda", "first", "random"}
    if tie_break not in allowed:
        raise ValueError(
            "tie_break must be 'larger_lambda', 'smaller_lambda', 'first', or 'random'."
        )
    if tie_break == "random":
        primary_values = summary[primary_col].to_numpy()
        if primary_ascending:
            best_value = np.min(primary_values)
        else:
            best_value = np.max(primary_values)
        if np.issubdtype(primary_values.dtype, np.floating):
            tied_mask = np.isclose(primary_values, best_value, rtol=1e-12, atol=1e-15)
        else:
            tied_mask = primary_values == best_value
        tied_indices = summary.index.to_numpy()[tied_mask]
        if len(tied_indices) == 0:
            raise RuntimeError("No tied candidate found. This should not happen.")
        if isinstance(tie_break_random_state, np.random.Generator):
            rng = tie_break_random_state
        else:
            rng = np.random.default_rng(tie_break_random_state)
        chosen_index = rng.choice(tied_indices)
        chosen_row = summary.loc[[chosen_index]]
        remaining = summary.drop(index=chosen_index).sort_values(
            by=[primary_col, "idx"],
            ascending=[primary_ascending, True],
            kind="mergesort",
        )
        summary_sorted = pd.concat([chosen_row, remaining], axis=0)
        return summary_sorted
    if tie_break == "larger_lambda":
        secondary_col = "lambda"
        secondary_ascending = False
    elif tie_break == "smaller_lambda":
        secondary_col = "lambda"
        secondary_ascending = True
    elif tie_break == "first":
        secondary_col = "idx"
        secondary_ascending = True
    summary_sorted = summary.sort_values(
        by=[primary_col, secondary_col],
        ascending=[primary_ascending, secondary_ascending],
        kind="mergesort",
    )
    return summary_sorted


def effective_lambda(lam, n_calibration):
    if (
        isinstance(n_calibration, (bool, np.bool_))
        or not np.isfinite(n_calibration)
        or int(n_calibration) != n_calibration
        or (n_calibration < 1)
    ):
        raise ValueError("n_calibration must be a positive integer")
    n = int(n_calibration)
    if not np.isfinite(lam) or not 0 < lam < n / (n + 1):
        raise ValueError(
            "lambda must satisfy 0 < lambda < n_calibration / (n_calibration + 1)"
        )
    return float((np.floor((n + 1) * float(lam)) + 1) / (n + 1))


def pact_pi0(pvals, lam, n_calibration):
    p = np.asarray(pvals, dtype=float).reshape(-1)
    if not p.size or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("pvals must be nonempty, finite, and in [0, 1]")
    lambda_plus = effective_lambda(lam, n_calibration)
    return float((1.0 + np.count_nonzero(p > lam)) / (p.size * (1.0 - lambda_plus)))


def pact_bh_reject(pvals, lam, alpha, n_calibration):
    pi0 = pact_pi0(pvals, lam, n_calibration)
    reject = multipletests(pvals, alpha=alpha / pi0, method="fdr_bh")[0]
    return (pi0, np.asarray(reject).reshape(-1).astype(int))


def run_pact_discrete(
    *,
    X_tr,
    X_cal,
    X_test,
    p_test,
    occ,
    candi_lam,
    alpha,
    base_seed,
    split_random_state,
    tie_break="larger_lambda",
    tie_break_random_state=None,
    selection_criterion="num_rej",
    gamma=1.0,
):
    allowed_criteria = {"num_rej", "pi0", "pi0_plus_var", "pi0_plus_se"}
    if selection_criterion not in allowed_criteria:
        raise ValueError(
            f"selection_criterion must be one of {allowed_criteria}, but got {selection_criterion!r}."
        )
    if gamma < 0:
        raise ValueError("gamma must be nonnegative.")
    X_tr_1, X_tr_2 = train_test_split(
        X_tr, test_size=0.5, random_state=split_random_state, shuffle=True
    )
    X_cal_modi = X_tr_2
    X_test_modi = np.vstack((X_cal, X_test))
    base_mod_modi = clone(occ)
    if "random_state" in base_mod_modi.get_params(deep=False):
        base_mod_modi.set_params(random_state=base_seed)
    base_mod_modi.fit(X_tr_1)
    sco_cal_modi = base_mod_modi.score_samples(X_cal_modi)
    sco_test_modi = base_mod_modi.score_samples(X_test_modi)
    p_test_modi = cp_vec(sco_test_modi, sco_cal_modi)
    records = []
    m_modi = len(p_test_modi)
    for idx, lam in enumerate(candi_lam):
        lam = float(lam)
        effective_lambda(lam, len(X_cal))
        pi_0_modi, rej_modi = pact_bh_reject(
            pvals=p_test_modi, n_calibration=len(X_cal_modi), lam=lam, alpha=alpha
        )
        num_rej_modi = int(np.sum(rej_modi))
        lam_aux = effective_lambda(lam, len(X_cal_modi))
        K_lam = int(np.sum(p_test_modi > lam))
        q_hat = K_lam / m_modi
        var_pi0_modi = q_hat * (1.0 - q_hat) / (m_modi * (1.0 - lam_aux) ** 2)
        se_pi0_modi = np.sqrt(var_pi0_modi)
        if selection_criterion == "num_rej":
            objective = num_rej_modi
        elif selection_criterion == "pi0":
            objective = pi_0_modi
        elif selection_criterion == "pi0_plus_var":
            objective = pi_0_modi + gamma * var_pi0_modi
        elif selection_criterion == "pi0_plus_se":
            objective = pi_0_modi + gamma * se_pi0_modi
        records.append(
            {
                "idx": idx,
                "lambda": lam,
                "lambda_effective": lam_aux,
                "n_calibration": len(X_cal_modi),
                "pi_0_modi": float(pi_0_modi),
                "Num_Rej_modi": num_rej_modi,
                "K_lam_modi": K_lam,
                "q_hat_modi": float(q_hat),
                "var_pi0_modi": float(var_pi0_modi),
                "se_pi0_modi": float(se_pi0_modi),
                "objective": float(objective),
            }
        )
    summary = pd.DataFrame(records)
    if selection_criterion == "num_rej":
        primary_col = "Num_Rej_modi"
        primary_ascending = False
    else:
        primary_col = "objective"
        primary_ascending = True
    summary = _sort_summary_for_lambda_selection(
        summary=summary,
        primary_col=primary_col,
        primary_ascending=primary_ascending,
        tie_break=tie_break,
        tie_break_random_state=tie_break_random_state,
    )
    lam_sel = float(summary.iloc[0]["lambda"])
    pi_0, rej = pact_bh_reject(
        pvals=p_test, n_calibration=len(X_cal), lam=lam_sel, alpha=alpha
    )
    return {
        "pi_0": pi_0,
        "rej": rej,
        "Selected_lambda": lam_sel,
        "Selected_lambda_effective": effective_lambda(lam_sel, len(X_cal)),
        "Selected_lambda_aux_effective": effective_lambda(lam_sel, len(X_cal_modi)),
        "n_calibration_aux": len(X_cal_modi),
        "aux_calibration_indices": train_test_split(
            np.arange(len(X_tr)),
            test_size=0.5,
            random_state=split_random_state,
            shuffle=True,
        )[1],
        "selection_summary": summary,
        "p_test_modi": p_test_modi,
        "selection_criterion": selection_criterion,
        "gamma": gamma,
    }
