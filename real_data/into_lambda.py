
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
    ],

    "METHODS": [
        "Storey path",
    ],

    "styles": {
        "colors": {
            "Storey path": "blue",
            "PACT": "red",
            "AMS-rob": "orange",
            "Gao": "green",
        },
        "markers": {
            "Storey path": "o",
            "PACT": "",
            "AMS-rob": "",
            "Gao": "",
        },
        "line_styles": {
            "Storey path": "-",
            "PACT": "--",
            "AMS-rob": "--",
            "Gao": "-.",
        },
    },

    "occ": default_config["occ_family"]["OneClassSVM"],
    "alpha": 0.1,

    "random_state": 2026,

    "candi_lam": np.linspace(0.1, 0.9, 9),

    "lambda_grid": np.linspace(0.1, 0.9, 20),

    "selection_cri": "pi0_plus_se",
    "selection_gamma": 1.0,

    "tie_break": "random",
    "tie_break_ori": "random",

    "x_label_name": r"$\lambda$",
}


panel_settings = [
    {
        "PanelID": "P1",
        "PanelTitle": "Average performance",
        "PanelOrder": 0,
        "random_state": 2026,
        "nrp": 100,
    },
    {
        "PanelID": "P2",
        "PanelTitle": "Single performance-1",
        "PanelOrder": 1,
        "random_state": 100,
        "nrp": 1,
    },
    {
        "PanelID": "P3",
        "PanelTitle": "Single performance-2",
        "PanelOrder": 2,
        "random_state": 20,
        "nrp": 1,
    },
    {
        "PanelID": "P4",
        "PanelTitle": "Single performance-3",
        "PanelOrder": 3,
        "random_state": 2026,
        "nrp": 1,
    },
]

occ = exp_config["occ"]
alpha = exp_config["alpha"]
lambda_grid = exp_config["lambda_grid"]
scenario_list = exp_config["scenarios"]
candi_lam = exp_config["candi_lam"]
styles = exp_config["styles"]
selection_cri = exp_config["selection_cri"]
tie_break = exp_config["tie_break"]
tie_break_ori = exp_config["tie_break_ori"]
selection_gamma = exp_config.get("selection_gamma", 1.0)
plot_methods = exp_config["METHODS"]


def extract_ams_result(out, theta):
    rej = np.asarray(out["rej"]).reshape(-1).astype(int)

    num_rej = int(np.sum(rej))
    num_true_rej = int(np.sum(theta * rej))
    fdr = np.sum((1 - theta) * rej) / max(num_rej, 1)
    power = np.sum(theta * rej) / max(np.sum(theta), 1)

    selection_summary = out["selection_summary"].copy()
    best = selection_summary.iloc[0]

    return {
        "lambda": float(out["Selected_lambda"]),
        "pi_0": float(out["pi_0"]),
        "rej": rej,
        "Num_Rej": num_rej,
        "Num_True_Rej": num_true_rej,
        "FDR": float(fdr),
        "Power": float(power),
        "Selection_pi_0": float(best["pi_0_modi"]),
        "Selection_Num_Rej": int(best["Num_Rej_modi"]),
        "Selection_Objective": float(best["objective"]),
    }


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
    alpha = exp_config["alpha"]
    lambda_grid = exp_config["lambda_grid"]
    candi_lam = exp_config["candi_lam"]
    selection_cri = exp_config["selection_cri"]
    selection_gamma = exp_config.get("selection_gamma", 1.0)
    tie_break = exp_config["tie_break"]
    tie_break_ori = exp_config["tie_break_ori"]

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
    m = len(theta)


    X_tr, X_cal, X_test = preprocess_features_for_occ(
        X_tr,
        X_cal,
        X_test,
        use_pca=True,
        random_state=base_seed,
    )


    base_mod = clone(exp_config["occ"])

    if "random_state" in base_mod.get_params(deep=False):
        base_mod.set_params(random_state=base_seed)

    base_mod.fit(X_tr)

    sco_cal = base_mod.score_samples(X_cal)
    sco_test = base_mod.score_samples(X_test)

    p_test = cp_vec(sco_test, sco_cal)

    results = []


    ams_rob_out = run_ams_new(
        X_tr=X_tr,
        X_cal=X_cal,
        X_test=X_test,
        p_test=p_test,
        occ=exp_config["occ"],
        candi_lam=candi_lam,
        alpha=alpha,
        base_seed=base_seed,
        split_random_state=base_seed,
        tie_break=tie_break,
        tie_break_random_state=base_seed + 17,
        selection_criterion=selection_cri,
        gamma=selection_gamma,
    )

    ams_rob = extract_ams_result(ams_rob_out, theta)

    ams_ori_out = run_ams_new(
        X_tr=X_tr,
        X_cal=X_cal,
        X_test=X_test,
        p_test=p_test,
        occ=exp_config["occ"],
        candi_lam=candi_lam,
        alpha=alpha,
        base_seed=base_seed,
        split_random_state=base_seed,
        tie_break=tie_break_ori,
        tie_break_random_state=base_seed + 29,
        selection_criterion="num_rej",
        gamma=0.0,
    )

    ams_ori = extract_ams_result(ams_ori_out, theta)

    gao_pi_0, gao_rej = adaptive_storey_bh_gao(
        p_test,
        q=alpha,
        delta=None,
        lam_max=0.8,
        robust=True,
    )

    gao_rej = np.asarray(gao_rej).reshape(-1).astype(int)

    gao_num_rej = int(np.sum(gao_rej))
    gao_num_true_rej = int(np.sum(theta * gao_rej))
    gao_fdr = np.sum((1 - theta) * gao_rej) / max(gao_num_rej, 1)
    gao_power = np.sum(theta * gao_rej) / max(np.sum(theta), 1)

    for lam in lambda_grid:
        lam = float(lam)

        pi_0, rej = storey_bh_reject(
            pvals=p_test,
            lam=lam,
            alpha=alpha,
        )

        rej = np.asarray(rej).reshape(-1).astype(int)

        num_rej = int(np.sum(rej))
        num_true_rej = int(np.sum(theta * rej))
        fdp = np.sum((1 - theta) * rej) / max(num_rej, 1)
        power = np.sum(theta * rej) / max(np.sum(theta), 1)

        results.append(
            {
                "ScenarioID": scenario["sid"],
                "ScenarioTitle": scenario["title"],
                "ScenarioOrder": scenario_id,
                "Lambda": lam,

                "pi_0": float(pi_0),
                "Num_Rej": num_rej,
                "Num_True_Rej": num_true_rej,
                "FDR": float(fdp),
                "Power": float(power),

                "Gao_pi_0": float(gao_pi_0),
                "Gao_Num_Rej": gao_num_rej,
                "Gao_Num_True_Rej": gao_num_true_rej,
                "Gao_FDR": float(gao_fdr),
                "Gao_Power": float(gao_power),

                "AMS_rob_lambda": ams_rob["lambda"],
                "AMS_rob_pi_0": ams_rob["pi_0"],
                "AMS_rob_Num_Rej": ams_rob["Num_Rej"],
                "AMS_rob_Num_True_Rej": ams_rob["Num_True_Rej"],
                "AMS_rob_FDR": ams_rob["FDR"],
                "AMS_rob_Power": ams_rob["Power"],

                "AMS_rob_Selection_pi_0": ams_rob["Selection_pi_0"],
                "AMS_rob_Selection_Num_Rej": ams_rob["Selection_Num_Rej"],
                "AMS_rob_Selection_Objective": ams_rob["Selection_Objective"],
                "AMS_rob_Selection_Criterion": selection_cri,
                "AMS_rob_Selection_Gamma": selection_gamma,

                "AMS_ori_lambda": ams_ori["lambda"],
                "AMS_ori_pi_0": ams_ori["pi_0"],
                "AMS_ori_Num_Rej": ams_ori["Num_Rej"],
                "AMS_ori_Num_True_Rej": ams_ori["Num_True_Rej"],
                "AMS_ori_FDR": ams_ori["FDR"],
                "AMS_ori_Power": ams_ori["Power"],

                "AMS_ori_Selection_pi_0": ams_ori["Selection_pi_0"],
                "AMS_ori_Selection_Num_Rej": ams_ori["Selection_Num_Rej"],
                "AMS_ori_Selection_Objective": ams_ori["Selection_Objective"],
                "AMS_ori_Selection_Criterion": "num_rej",
                "AMS_ori_Selection_Gamma": 0.0,

                "Rep": rep_id,
                "alpha": alpha,
                "m": m,
                "pi_out": scenario["pi_out"],
            }
        )

    return results


def run_single_panel_rep(
    panel,
    scenario_id,
    rep_id,
    *,
    features_train,
    labels_train,
    features_test,
    labels_test,
    exp_config,
):

    exp_config_local = copy.deepcopy(exp_config)
    exp_config_local["random_state"] = int(panel["random_state"])

    out = run_single_rep(
        scenario_id=scenario_id,
        rep_id=rep_id,
        features_train=features_train,
        labels_train=labels_train,
        features_test=features_test,
        labels_test=labels_test,
        exp_config=exp_config_local,
    )

    for row in out:
        row["PanelID"] = panel["PanelID"]
        row["PanelTitle"] = panel["PanelTitle"]
        row["PanelOrder"] = panel["PanelOrder"]
        row["PanelRandomState"] = int(panel["random_state"])
        row["PanelNRP"] = int(panel["nrp"])

    return out



def build_lambda_path_plot_df(df):


    base_cols = [
        "ScenarioID",
        "ScenarioTitle",
        "ScenarioOrder",
        "PanelID",
        "PanelTitle",
        "PanelOrder",
        "PanelRandomState",
        "PanelNRP",
        "Lambda",
        "Rep",
    ]

    # Storey path
    storey_df = df[
        base_cols + ["pi_0", "Num_Rej", "FDR", "Power"]
    ].copy()
    storey_df["Method"] = "Storey path"

    # Gao horizontal path
    gao_df = df[
        base_cols
        + [
            "Gao_pi_0",
            "Gao_Num_Rej",
            "Gao_FDR",
            "Gao_Power",
        ]
    ].copy()

    gao_df = gao_df.rename(
        columns={
            "Gao_pi_0": "pi_0",
            "Gao_Num_Rej": "Num_Rej",
            "Gao_FDR": "FDR",
            "Gao_Power": "Power",
        }
    )
    gao_df["Method"] = "Gao"

    # PACT horizontal path
    ams_ori_df = df[
        base_cols
        + [
            "AMS_ori_pi_0",
            "AMS_ori_Num_Rej",
            "AMS_ori_FDR",
            "AMS_ori_Power",
            "AMS_ori_lambda",
            "AMS_ori_Selection_pi_0",
            "AMS_ori_Selection_Num_Rej",
            "AMS_ori_Selection_Objective",
            "AMS_ori_Selection_Criterion",
            "AMS_ori_Selection_Gamma",
        ]
    ].copy()

    ams_ori_df = ams_ori_df.rename(
        columns={
            "AMS_ori_pi_0": "pi_0",
            "AMS_ori_Num_Rej": "Num_Rej",
            "AMS_ori_FDR": "FDR",
            "AMS_ori_Power": "Power",
            "AMS_ori_lambda": "Selected_lambda",
            "AMS_ori_Selection_pi_0": "Selection_pi_0",
            "AMS_ori_Selection_Num_Rej": "Selection_Num_Rej",
            "AMS_ori_Selection_Objective": "Selection_Objective",
            "AMS_ori_Selection_Criterion": "Selection_Criterion",
            "AMS_ori_Selection_Gamma": "Selection_Gamma",
        }
    )
    ams_ori_df["Method"] = "PACT"

    # AMS-rob horizontal path
    ams_rob_df = df[
        base_cols
        + [
            "AMS_rob_pi_0",
            "AMS_rob_Num_Rej",
            "AMS_rob_FDR",
            "AMS_rob_Power",
            "AMS_rob_lambda",
            "AMS_rob_Selection_pi_0",
            "AMS_rob_Selection_Num_Rej",
            "AMS_rob_Selection_Objective",
            "AMS_rob_Selection_Criterion",
            "AMS_rob_Selection_Gamma",
        ]
    ].copy()

    ams_rob_df = ams_rob_df.rename(
        columns={
            "AMS_rob_pi_0": "pi_0",
            "AMS_rob_Num_Rej": "Num_Rej",
            "AMS_rob_FDR": "FDR",
            "AMS_rob_Power": "Power",
            "AMS_rob_lambda": "Selected_lambda",
            "AMS_rob_Selection_pi_0": "Selection_pi_0",
            "AMS_rob_Selection_Num_Rej": "Selection_Num_Rej",
            "AMS_rob_Selection_Objective": "Selection_Objective",
            "AMS_rob_Selection_Criterion": "Selection_Criterion",
            "AMS_rob_Selection_Gamma": "Selection_Gamma",
        }
    )
    ams_rob_df["Method"] = "AMS-rob"

    plot_df = pd.concat(
        [
            storey_df,
            gao_df,
            ams_ori_df,
            ams_rob_df,
        ],
        ignore_index=True,
    )

    plot_df["Method"] = pd.Categorical(
        plot_df["Method"],
        categories=[
            "Storey path",
            "Gao",
            "PACT",
            "AMS-rob",
        ],
        ordered=True,
    )

    return plot_df

