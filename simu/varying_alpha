exp_config = {
    'data_name': 'gaussian_one_group_mess_nonnull',
    'data_config': {
        'n_null': 10000, 'r_tr': 0.5, 'p': 1, 'm': 3000,
        'pi_value': 0.2, 'mu': 5,
        "mess_mu":1.0, "mess_sd":0.1,
    },
    "METHODS": [
        "BH",

        "BKY",
        "Storey-0.2",
        "Storey-0.5",
        "Storey-0.8",

        "Gao",

        # "PACT-R",
        "PACT",
        "greedy-AMS",
        "oracle",
    ],

    "styles": {
        "colors": {
            "BH": "gray",
            "BKY": "purple",
            "Storey-0.2": "orange",
            "Storey-0.5": "brown",
            "Storey-0.8": "goldenrod",
            "Gao": "green",
            "greedy-AMS": "green",
            "oracle": "blue",
            "PACT-R": "pink",
            "PACT": "red",
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
            "PACT-R": "h",
            "PACT": "h",

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
            "PACT-R": "-",
            "PACT": "-",

        },
    },
    'occ': default_config['occ_family']['OneClassSVM'],
    'bic': default_config['bic_family']['RandomForest'],
    'alpha': 0.1,
    'random_state': 2026,
    # AMS new selection rule
    "selection_cri_rob": "pi0_plus_se",
    "gamma_rob": 1.0,
    "tie_break_rob": "random",
    "tie_break_ori": "random",

    'candi_lam': np.linspace(0.1, 0.9, 9),
    'vary_vec': np.array([0.10, 0.20, 0.30, 0.40]),
    'pi_vec': np.array([0.4, 0.5, 0.6]),
    'x_label_name': r"$\alpha$",
    'fix_vec': [300, 3000],
    'fix_para_name': r"Test sample size $m$",
}

x_vec = exp_config['vary_vec']
pi_vec = exp_config['pi_vec']
x_label_name = exp_config['x_label_name']
fix_vec = exp_config['fix_vec']
fix_para_name = exp_config['fix_para_name']
occ = exp_config['occ']
bic = exp_config['bic']
styles = exp_config['styles']
alpha = exp_config['alpha']
candi_lam = exp_config['candi_lam']
selection_cri_rob = exp_config["selection_cri_rob"]
tie_break_rob = exp_config['tie_break_rob']
tie_break_ori = exp_config['tie_break_ori']
gamma_rob = exp_config.get("gamma_rob")

nrp = 500


def run_single_rep(fix_id, pi_id, x_id, rep_id):

    base_seed = int(exp_config['random_state']+rep_id)
    rng = np.random.default_rng(base_seed)

    current_config = copy.deepcopy(exp_config['data_config'])
    alpha_cur = float(x_vec[x_id])
    pi_cur = float(pi_vec[pi_id])
    current_config["m"] = fix_vec[fix_id]
    current_config["pi_value"] = pi_cur

    X_tr, X_cal, X_test, theta, X_mirror = SyntheticGenerator.generate(exp_config['data_name'], random_state=base_seed, **current_config)
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

            rej = multipletests(p_test, alpha=alpha_cur/pi_0, method='fdr_bh')[0]

        if name == "Storey-0.5":

            pi_0 = storey_pi0(p_test,0.5)

            rej = multipletests(p_test, alpha=alpha_cur/pi_0, method='fdr_bh')[0]

        if name == "Storey-0.8":

            pi_0 = storey_pi0(p_test,0.8)

            rej = multipletests(p_test, alpha=alpha_cur/pi_0, method='fdr_bh')[0]

        if name == "BH":

            rej = multipletests(p_test, alpha=alpha_cur, method='fdr_bh')[0]
            pi_0 = 1

        if name == "BKY":
            pi_0, rej = by_2006_adaptive_bh(
                pvals=p_test,
                q=alpha_cur,
            )

        if name == "oracle":
            oracle_results = run_oracle_candidates(
                p_test=p_test,
                theta=theta,
                candi_lam=candi_lam,
                alpha=alpha_cur,
                fixed_para=fix_vec[fix_id],
                varied_para=x_vec[x_id],
                rep_id=rep_id,
            )

            for res in oracle_results:
                res["pi_value"] = pi_cur
                res["alpha"] = alpha_cur

            results.extend(oracle_results)
            continue

        if name == "greedy-AMS":
            out = run_greedy_ams(
                p_test=p_test,
                candi_lam=candi_lam,
                alpha=alpha_cur,
                tie_break="larger_lambda",
            )

            pi_0 = out["pi_0"]
            rej = out["rej"]
            lam_sel = out["Selected_lambda"]

        if name == "PACT-R":
            out = run_ams_new(
                X_tr=X_tr,
                X_cal=X_cal,
                X_test=X_test,
                p_test=p_test,
                occ=occ,
                candi_lam=candi_lam,
                alpha=alpha_cur,
                base_seed=base_seed,
                split_random_state=base_seed,
                tie_break=tie_break_rob,
                selection_criterion=selection_cri_rob,
                gamma=gamma_rob,
            )

            pi_0 = out["pi_0"]
            rej = out["rej"]
            lam_sel = out["Selected_lambda"]

        if name == "PACT":
            out = run_ams_new(
                X_tr=X_tr,
                X_cal=X_cal,
                X_test=X_test,
                p_test=p_test,
                occ=occ,
                candi_lam=candi_lam,
                alpha=alpha_cur,
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
                q=alpha_cur,
                delta=None,  # Section 4.1 default: 50 / #{P_i > q}
                lam_max=0.8,  # Section 4.1 constraint
                robust=True,
            )


        fdp = np.sum((1 - theta) * rej) / max(np.sum(rej), 1)
        power = np.sum(theta * rej) / max(np.sum(theta), 1)

        eval_dict = evaluate_rejection(theta, rej)

        results.append(
            {
                "Fixed_para": fix_vec[fix_id],
                "pi_value": pi_cur,
                "alpha": alpha_cur,
                "Varied_para": x_vec[x_id],
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


def finalize_oracle_by_mean(df):


    df = df.copy()

    oracle_cand = df[df["Method"] == "oracle_candidate"].copy()
    df_other = df[df["Method"] != "oracle_candidate"].copy()

    if oracle_cand.empty:
        return df, pd.DataFrame()

    if "alpha" not in oracle_cand.columns:
        oracle_cand["alpha"] = oracle_cand["Varied_para"].astype(float)
    if "alpha" not in df_other.columns:
        df_other["alpha"] = df_other["Varied_para"].astype(float)

    group_cols = ["Fixed_para", "pi_value", "Varied_para", "Oracle_lambda"]

    oracle_summary = (
        oracle_cand
        .groupby(group_cols, as_index=False)
        .agg(
            alpha=("alpha", "first"),
            pi_0=("pi_0", "mean"),
            FDR=("FDR", "mean"),
            Power=("Power", "mean"),
            Num_Rej=("Num_Rej", "mean"),
            n_rep=("Rep", "nunique"),
        )
    )


    oracle_summary = oracle_summary.sort_values(
        by=["Fixed_para", "pi_value", "Varied_para", "Num_Rej", "Oracle_lambda"],
        ascending=[True, True, True, False, False],
    )

    oracle_final = (
        oracle_summary
        .drop_duplicates(subset=["Fixed_para", "pi_value", "Varied_para"], keep="first")
        .copy()
    )

    oracle_final["Method"] = "oracle"
    oracle_final["Selected_lambda"] = oracle_final["Oracle_lambda"]
    oracle_final["Rep"] = -1

    oracle_final = oracle_final[
        [
            "Fixed_para",
            "pi_value",
            "alpha",
            "Varied_para",
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



