from pathlib import Path
import json
import numpy as np
from sklearn.base import clone
from sklearn.svm import OneClassSVM
from statsmodels.stats.multitest import multipletests
from baselines import (
    storey_pi0,
    by_2006_adaptive_bh,
    adaptive_storey_bh_gao,
    choose_lambda_step_down,
)
from pact import cp_vec, evaluate_rejection, run_pact_discrete
from real_data.cifar_data import (
    build_cifar_multiclass_ood_split,
    preprocess_features_for_occ,
)


def load_config():
    config = json.loads(
        (Path(__file__).resolve().parent / "settings.json").read_text(encoding="utf-8")
    )
    config["occ"] = OneClassSVM(**config["occ"]["params"])
    config["candi_lam"] = np.asarray(config["candi_lam"], dtype=float)
    return config


def run_single_rep(
    scenario_id,
    rep_id,
    *,
    features_train,
    labels_train,
    features_test,
    labels_test,
    exp_config
):
    scenario = exp_config["scenarios"][scenario_id]
    base_seed = int(exp_config["random_state"] + 1000 * scenario_id + rep_id)
    X_tr, X_cal, X_test, theta, meta = build_cifar_multiclass_ood_split(
        features_train=features_train,
        labels_train=labels_train,
        features_test=features_test,
        labels_test=labels_test,
        scenario=scenario,
        random_state=base_seed,
    )
    theta = np.asarray(theta).reshape(-1).astype(int)
    X_tr, X_cal, X_test = preprocess_features_for_occ(
        X_tr,
        X_cal,
        X_test,
        use_pca=True,
        pca_dim=int(exp_config["pca_dim"]),
        random_state=base_seed,
    )
    alpha = exp_config["alpha"]
    candi_lam = exp_config["candi_lam"]
    occ = exp_config["occ"]
    tie_break_ori = exp_config["tie_break_ori"]
    m = len(X_test)
    theta = theta.reshape((m,))
    base_mod = clone(occ)
    if "random_state" in base_mod.get_params(deep=False):
        base_mod.set_params(random_state=base_seed)
    base_mod.fit(X_tr)
    sco_cal = base_mod.score_samples(X_cal)
    sco_test = base_mod.score_samples(X_test)
    p_test = cp_vec(sco_test, sco_cal)
    results = []
    candidate_rows = []
    common = dict(
        PCA_dim=int(exp_config["pca_dim"]),
        ScenarioIndex=scenario_id,
        ScenarioID=scenario["sid"],
        ScenarioTitle=scenario["title"],
        Rep=rep_id,
        Seed=base_seed,
        SeedBase=int(exp_config["random_state"]),
        m=len(theta),
        m1=int(theta.sum()),
        n_tr=int(scenario["n_tr"]),
        n_cal=int(scenario["n_cal"]),
        alpha=float(alpha),
        pi_value=float(scenario["pi_out"]),
        anomaly_fraction=float(scenario["pi_out"]),
        pi0_true=1 - float(scenario["pi_out"]),
        Candidate_lambda=json.dumps(np.asarray(candi_lam, dtype=float).tolist()),
    )
    for name in exp_config["METHODS"]:
        rej = None
        pi_0 = np.nan
        lam_sel = np.nan
        selected_aux = None
        pact_details = {}
        selection = None
        if name == "Storey-0.2":
            pi_0 = storey_pi0(p_test, 0.2)
            rej = multipletests(p_test, alpha=alpha / pi_0, method="fdr_bh")[0]
        if name == "Storey-0.5":
            pi_0 = storey_pi0(p_test, 0.5)
            rej = multipletests(p_test, alpha=alpha / pi_0, method="fdr_bh")[0]
        if name == "Storey-0.8":
            pi_0 = storey_pi0(p_test, 0.8)
            rej = multipletests(p_test, alpha=alpha / pi_0, method="fdr_bh")[0]
        if name == "BH":
            rej = multipletests(p_test, alpha=alpha, method="fdr_bh")[0]
            pi_0 = 1
        if name == "BKY":
            pi_0, rej = by_2006_adaptive_bh(pvals=p_test, q=alpha)
        if name == "PACT":
            out = run_pact_discrete(
                X_tr=X_tr,
                X_cal=X_cal,
                X_test=X_test,
                p_test=p_test,
                occ=occ,
                candi_lam=candi_lam,
                alpha=alpha,
                base_seed=base_seed,
                split_random_state=base_seed,
                tie_break=tie_break_ori,
                tie_break_random_state=base_seed,
                selection_criterion="num_rej",
                gamma=0.0,
            )
            pi_0 = out["pi_0"]
            rej = out["rej"]
            lam_sel = out["Selected_lambda"]
            selection = out["selection_summary"]
            pact_details = {
                key: out[key]
                for key in [
                    "Selected_lambda_effective",
                    "Selected_lambda_aux_effective",
                    "n_calibration_aux",
                ]
            }
            pact_details["n_calibration_final"] = len(X_cal)
        if name == "Gao":
            pi_0, rej = adaptive_storey_bh_gao(
                p_test, q=alpha, delta=None, lam_max=0.8, robust=True
            )
            lam_sel = choose_lambda_step_down(
                p_test, q=alpha, delta=None, lam_max=0.8, robust=True
            )
        if rej is None:
            raise ValueError("Unknown method: " + name)
        rej = np.asarray(rej).reshape(-1).astype(bool)
        eval_dict = evaluate_rejection(theta, rej)
        count, tp = (int(rej.sum()), int((theta * rej).sum()))
        fdp, tpr = ((count - tp) / max(count, 1), tp / max(common["m1"], 1))
        if selection is not None:
            candidate_rows.extend(
                (
                    dict(common, Method=name, **row)
                    for row in selection.to_dict(orient="records")
                )
            )
            if "Num_Rej_modi" in selection:
                selected_aux = float(selection.iloc[0]["Num_Rej_modi"])
        results.append(
            {
                **common,
                **pact_details,
                "ScenarioID": scenario["sid"],
                "ScenarioTitle": scenario["title"],
                "Method": name,
                "pi_0": pi_0,
                "FDR": eval_dict["FDR"],
                "Power": eval_dict["Power"],
                "Num_Rej": eval_dict["Num_Rej"],
                "Selected_lambda": lam_sel,
                "Oracle_lambda": np.nan,
                "Rep": rep_id,
                "FDP": fdp,
                "TPR": tpr,
                "TP": tp,
                "FP": count - tp,
                "R": count,
                "pi0_hat": float(pi_0),
                "Aux_Num_Rej": selected_aux,
            }
        )
    return {"rows": results, "candidates": candidate_rows}
