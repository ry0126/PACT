exp_config = {
    "data_name": "cifar10_resnet18",

    "scenarios": [
        {
            "sid": "S1",
            "title": "Land vehicles vs airplanes",
            "inlier_classes": [1, 9],
            "outlier_classes": [0],
            "n_tr": ntr,
            "n_cal": ncal,
            "m": m,
            "pi_out": pi_out,
        },
        {
            "sid": "S2",
            "title": "Land vehicles vs air/sea vehicles",
            "inlier_classes": [1, 9],
            "outlier_classes": [0, 8],
            "n_tr": ntr,
            "n_cal": ncal,
            "m": m,
            "pi_out": pi_out,
        },
        {
            "sid": "S3",
            "title": "Land vehicles vs hoofed mammals",
            "inlier_classes": [1, 9],
            "outlier_classes": [4, 7],
            "n_tr": ntr,
            "n_cal": ncal,
            "m": m,
            "pi_out": pi_out,
        },
    ],

    "METHODS": [
        "BH",

        "BKY",
        "Storey-0.2",
        "Storey-0.5",
        "Storey-0.8",

        "Gao",

        # "greedy-AMS",
        # "AMS-rob",
        "AMS-ori",
        # "oracle",
    ],

    "styles": {
        "colors": {
            "BH": "gray",
            "BKY": "purple",
            "Storey-0.2": "orange",
            "Storey-0.5": "brown",
            "Storey-0.8": "goldenrod",
            "Gao": "green",
            "greedy-AMS": "tomato",
            "oracle": "blue",
            "AMS-rob": "pink",
            "AMS-ori": "red",
        },

        "markers": {
            "BH": "X",

            "BKY": "P",
            "Storey-0.2": "s",
            "Storey-0.5": "^",
            "Storey-0.8": "v",

            "Gao": "D",

            "greedy-AMS": "*",
            "oracle": "o",
            "AMS-rob": "h",
            "AMS-ori": "h",

        },

        "line_styles": {
            "BH": "-.",
            "BKY": "-.",
            "Storey-0.2": "--",
            "Storey-0.5": "--",
            "Storey-0.8": "--",
            "Gao": "-",
            "greedy-AMS": "-",
            "oracle": ":",
            "AMS-rob": "-",
            "AMS-ori": "-",

        },
    },

    "occ": default_config["occ_family"]["OneClassSVM"],
    "alpha": 0.1,
    "random_state": 2026,

    "candi_lam": np.linspace(0.1, 0.9, 9),

    "selection_cri_rob": "pi0_plus_se",
    "gamma_rob": 1.0,
    "tie_break_rob": "random",
    "tie_break_ori": "random",

    "x_label_name": r"$\lambda$",
}

occ = exp_config["occ"]
alpha = exp_config["alpha"]
scenario_list = exp_config["scenarios"]
candi_lam = exp_config["candi_lam"]
styles = exp_config["styles"]
selection_cri_rob = exp_config["selection_cri_rob"]
tie_break_rob = exp_config['tie_break_rob']
tie_break_ori = exp_config['tie_break_ori']
gamma_rob = exp_config.get("gamma_rob")
plot_methods = exp_config["METHODS"]

nrp = 500


run_pvalue_hist = True
run_storey_path = True
nrp_pvalue_hist = nrp
nrp_storey_path = nrp
storey_path_our_method = "AMS-ori"  # options: "AMS-ori" or "AMS-rob"
lam_plot_grid = np.linspace(0.1, 0.9, 20)
diagnostic_n_jobs = -1
diagnostic_show = True
diagnostic_save = True

def finalize_oracle_by_mean(df):

    df = df.copy()

    oracle_cand = df[df["Method"] == "oracle_candidate"].copy()
    df_other = df[df["Method"] != "oracle_candidate"].copy()

    if oracle_cand.empty:
        return df, pd.DataFrame()

    group_cols = ["ScenarioID", "ScenarioTitle", "Oracle_lambda"]

    oracle_summary = (
        oracle_cand
        .groupby(group_cols, as_index=False)
        .agg(
            pi_0=("pi_0", "mean"),
            FDR=("FDR", "mean"),
            Power=("Power", "mean"),
            Num_Rej=("Num_Rej", "mean"),
            n_rep=("Rep", "nunique"),
        )
    )

    oracle_summary = oracle_summary.sort_values(
        by=["ScenarioID", "Num_Rej", "Oracle_lambda"],
        ascending=[True, False, False],
    )

    oracle_final = (
        oracle_summary
        .drop_duplicates(subset=["ScenarioID"], keep="first")
        .copy()
    )

    oracle_final["Method"] = "oracle"
    oracle_final["Selected_lambda"] = oracle_final["Oracle_lambda"]
    oracle_final["Rep"] = -1

    oracle_final = oracle_final[
        [
            "ScenarioID",
            "ScenarioTitle",
            "Method",
            "pi_0",
            "FDR",
            "Power",
            "Num_Rej",
            "Selected_lambda",
            "Oracle_lambda",
            "Rep",
            "n_rep",
        ]
    ]

    df_plot = pd.concat([df_other, oracle_final], ignore_index=True)

    return df_plot, oracle_summary



def run_single_rep(
        scenario_id,
        rep_id,
        *,
        features_train,
        labels_train,
        features_test,
        labels_test,
        exp_config,
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
        random_state=base_seed,
    )

    alpha = exp_config["alpha"]
    candi_lam = exp_config["candi_lam"]

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
    for name in exp_config['METHODS']:

        rej = None
        pi_0 = np.nan
        lam_sel = np.nan
        if name == "Storey-0.2":

            pi_0 = storey_pi0(p_test,0.2)

            rej = multipletests(p_test, alpha=alpha/pi_0, method='fdr_bh')[0]

        if name == "Storey-0.5":

            pi_0 = storey_pi0(p_test,0.5)

            rej = multipletests(p_test, alpha=alpha/pi_0, method='fdr_bh')[0]

        if name == "Storey-0.8":

            pi_0 = storey_pi0(p_test,0.8)

            rej = multipletests(p_test, alpha=alpha/pi_0, method='fdr_bh')[0]

        if name == "BH":

            rej = multipletests(p_test, alpha=alpha, method='fdr_bh')[0]
            pi_0 = 1

        if name == "BKY":
            pi_0, rej = by_2006_adaptive_bh(
                pvals=p_test,
                q=alpha,
            )

        if name == "oracle":
            oracle_results = run_oracle_candidates_box(
                p_test=p_test,
                theta=theta,
                candi_lam=candi_lam,
                alpha=alpha,
                scenario_id=scenario["sid"],
                scenario_title=scenario["title"],
                rep_id=rep_id,
            )

            results.extend(oracle_results)
            continue

        if name == "greedy-AMS":
            out = run_greedy_ams(
                p_test=p_test,
                candi_lam=candi_lam,
                alpha=alpha,
                tie_break="larger_lambda",
            )

            pi_0 = out["pi_0"]
            rej = out["rej"]
            lam_sel = out["Selected_lambda"]

        if name == "AMS-rob":
            out = run_ams_new(
                X_tr=X_tr,
                X_cal=X_cal,
                X_test=X_test,
                p_test=p_test,
                occ=occ,
                candi_lam=candi_lam,
                alpha=alpha,
                base_seed=base_seed,
                split_random_state=base_seed,
                tie_break=tie_break_rob,
                selection_criterion=selection_cri_rob,
                gamma=gamma_rob,
            )

            pi_0 = out["pi_0"]
            rej = out["rej"]
            lam_sel = out["Selected_lambda"]

        if name == "AMS-ori":
            out = run_ams_new(
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


        if name == "Gao":
            pi_0, rej = adaptive_storey_bh_gao(
                p_test,
                q=alpha,
                delta=None,  # Section 4.1 default: 50 / #{P_i > q}
                lam_max=0.8,  # Section 4.1 constraint
                robust=True,
            )

        eval_dict = evaluate_rejection(theta, rej)

        results.append(
            {
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
            }
        )


    return results




