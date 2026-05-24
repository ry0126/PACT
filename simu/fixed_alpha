
exp_config = {
    'data_name': 'gaussian_one_group',

    'data_config': {
        'n_null': 10000, 'r_tr': 0.5, 'p': 100, 'm': 3000,
        'pi_value': 0.8,  'mu': 3,
    },
    "METHODS": [
        "BH",

        "BKY",
        "Storey-0.2",
        "Storey-0.5",
        "Storey-0.8",

        "Gao",

        # "greedy-AMS",

        # "PACT-R",
        "PACT",
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
            "greedy-AMS-all": "tomato",

            "oracle": "blue",
            "oracle-all": "blue",

            "PACT-R": "pink",
            "PACT-R-all": "pink",

            "PACT": "red",
            "PACT-all": "red",
        },

        "markers": {
            "BH": "X",

            "BKY": "P",
            "Storey-0.2": "s",
            "Storey-0.5": "^",
            "Storey-0.8": "v",

            "Gao": "D",

            "greedy-AMS": "*",
            "greedy-AMS-all": "X",

            "oracle": "o",
            "oracle-all": "D",

            "PACT-R": "h",
            "PACT-R-all": "H",

            "PACT": "h",
            "PACT-all": "H",
        },

        "line_styles": {
            "BH": "-.",
            "BKY": "-.",
            "Storey-0.2": "--",
            "Storey-0.5": "--",
            "Storey-0.8": "--",
            "Gao": "-",

            "greedy-AMS": "-",
            "greedy-AMS-all": "--",

            "oracle": ":",
            "oracle-all": "--",

            "PACT-R": "-",
            "PACT-R-all": "--",

            "PACT": "-",
            "PACT-all": "--",
        },
    },
    'occ': default_config['occ_family']['OneClassSVM'],
    'bic': default_config['bic_family']['RandomForest'],
    'alpha': 0.1,
    'random_state': 2026,
    'candi_lam': np.linspace(0.1, 0.9, 9),
    "selection_cri_rob": "pi0_plus_se",
    "gamma_rob": 1.0,
    "tie_break_rob": "random",
    "tie_break_ori": "random",
        'scenarios': [
        {"sid": "a", "pi_value": 0.4, "mu": 2, "p": 1, 'data_name': 'gaussian_one_group',  "title": r"(a) low dim: weak signal"},
        {"sid": "c", "pi_value": 0.4, "mu": 5, "p": 1, "mess_mu":1.0, "mess_sd":0.1,
         'data_name': 'gaussian_one_group_mess_nonnull', "title": r"(b) low dim: mixture signal"},
        {"sid": "b", "pi_value": 0.4, "mu": 0.55, "p": 100, 'data_name': 'gaussian_one_group', "title": r"(c) high dim: weak signal"},
        {"sid": "d", "pi_value": 0.4, "mu": 0.75, "p": 100, "mess_mu":0.3, "mess_sd":0.1,
         'data_name': 'gaussian_one_group_mess_nonnull', "title": r"(d) high dim: mixture signal"},
    ],

}


occ = exp_config['occ']
bic = exp_config['bic']
styles = exp_config['styles']
alpha = exp_config['alpha']
candi_lam = exp_config['candi_lam']
scenario_list = exp_config['scenarios']
selection_cri_rob = exp_config["selection_cri_rob"]
tie_break_rob = exp_config['tie_break_rob']
tie_break_ori = exp_config['tie_break_ori']

gamma_rob = exp_config.get("gamma_rob")
nrp = 500


def run_single_rep(scenario_id, rep_id):

    scenario = scenario_list[scenario_id]

    base_seed = int(exp_config['random_state'] + 1000 * scenario_id + rep_id)
    rng = np.random.default_rng(base_seed)

    current_config = copy.deepcopy(exp_config['data_config'])
    current_config["mu"] = scenario["mu"]
    current_config["pi_value"] = scenario["pi_value"]
    current_config["p"] = scenario["p"]

    if "m" in scenario:
        current_config["m"] = scenario["m"]

    data_name = scenario["data_name"]


    if data_name == "gaussian_one_group_mess_nonnull":
        if "mess_prob" in scenario:
            current_config["mess_prob"] = scenario["mess_prob"]

        if "mess_mu" in scenario:
            current_config["mess_mu"] = scenario["mess_mu"]

        if "mess_sd" in scenario:
            current_config["mess_sd"] = scenario["mess_sd"]

    else:

        current_config.pop("mess_prob", None)
        current_config.pop("mess_mu", None)
        current_config.pop("mess_sd", None)


    X_tr, X_cal, X_test, theta, X_mirror = SyntheticGenerator.generate(
        data_name,
        random_state=base_seed,
        **current_config,
    )


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

        if name == "PACT-R":
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

        if name == "PACT":
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


def finalize_oracles_by_mean_num_rej_box(
    df,
    *,
    tie_break="larger_lambda",
):


    df = df.copy()

    oracle_specs = [
        ("oracle_candidate", "oracle"),
        ("oracle_all_candidate", "oracle-all"),
    ]

    candidate_methods = [x[0] for x in oracle_specs]

    df_other = df[~df["Method"].isin(candidate_methods)].copy()

    final_list = [df_other]
    summary_list = []

    required_cols = [
        "ScenarioID",
        "ScenarioTitle",
        "Oracle_lambda",
        "pi_0",
        "FDR",
        "Power",
        "Num_Rej",
        "Rep",
    ]

    for candidate_method, output_method in oracle_specs:

        oracle_cand = df[df["Method"] == candidate_method].copy()

        if oracle_cand.empty:
            continue

        missing_cols = [
            col for col in required_cols
            if col not in oracle_cand.columns
        ]

        if missing_cols:
            raise ValueError(
                f"{candidate_method} 缺少以下列: {missing_cols}"
            )

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

        if tie_break == "larger_lambda":
            oracle_summary = oracle_summary.sort_values(
                by=["ScenarioID", "Num_Rej", "Oracle_lambda"],
                ascending=[True, False, False],
                kind="mergesort",
            )

        elif tie_break == "smaller_lambda":
            oracle_summary = oracle_summary.sort_values(
                by=["ScenarioID", "Num_Rej", "Oracle_lambda"],
                ascending=[True, False, True],
                kind="mergesort",
            )

        else:
            raise ValueError(
                "tie_break must be 'larger_lambda' or 'smaller_lambda'."
            )

        selected_lam = (
            oracle_summary
            .drop_duplicates(subset=["ScenarioID"], keep="first")
            [["ScenarioID", "Oracle_lambda"]]
            .rename(columns={"Oracle_lambda": "Selected_lambda_oracle"})
        )

        oracle_final = oracle_cand.merge(
            selected_lam,
            on="ScenarioID",
            how="inner",
        )

        oracle_final = oracle_final[
            oracle_final["Oracle_lambda"]
            == oracle_final["Selected_lambda_oracle"]
        ].copy()

        oracle_final["Method"] = output_method
        oracle_final["Selected_lambda"] = oracle_final["Oracle_lambda"]

        oracle_final = oracle_final.drop(
            columns=["Selected_lambda_oracle"]
        )

        oracle_summary["CandidateMethod"] = candidate_method
        oracle_summary["OutputMethod"] = output_method

        final_list.append(oracle_final)
        summary_list.append(oracle_summary)

    df_plot = pd.concat(final_list, ignore_index=True)

    if len(summary_list) == 0:
        oracle_summary_all = pd.DataFrame()
    else:
        oracle_summary_all = pd.concat(
            summary_list,
            ignore_index=True,
        )

    return df_plot, oracle_summary_all



def build_current_config_for_scenario(scenario):

    current_config = copy.deepcopy(exp_config["data_config"])

    current_config["mu"] = scenario["mu"]
    current_config["pi_value"] = scenario["pi_value"]
    current_config["p"] = scenario["p"]

    if "m" in scenario:
        current_config["m"] = scenario["m"]

    data_name = scenario["data_name"]

    if data_name == "gaussian_one_group_mess_nonnull":
        if "mess_prob" in scenario:
            current_config["mess_prob"] = scenario["mess_prob"]

        if "mess_mu" in scenario:
            current_config["mess_mu"] = scenario["mess_mu"]

        if "mess_sd" in scenario:
            current_config["mess_sd"] = scenario["mess_sd"]
    else:
        current_config.pop("mess_prob", None)
        current_config.pop("mess_mu", None)
        current_config.pop("mess_sd", None)

    return data_name, current_config

def collect_pvalues_single_rep(scenario_id, rep_id):

    scenario = scenario_list[scenario_id]

    base_seed = int(exp_config["random_state"] + 1000 * scenario_id + rep_id)

    data_name, current_config = build_current_config_for_scenario(scenario)

    X_tr, X_cal, X_test, theta, X_mirror = SyntheticGenerator.generate(
        data_name,
        random_state=base_seed,
        **current_config,
    )

    theta = theta.reshape(-1).astype(int)

    base_mod = clone(occ)
    if "random_state" in base_mod.get_params(deep=False):
        base_mod.set_params(random_state=base_seed)

    base_mod.fit(X_tr)

    sco_cal = base_mod.score_samples(X_cal)
    sco_test = base_mod.score_samples(X_test)

    p_test = cp_vec(sco_test, sco_cal)
    p_test = np.asarray(p_test).reshape(-1)

    df_p = pd.DataFrame(
        {
            "ScenarioID": scenario["sid"],
            "ScenarioTitle": scenario["title"],
            "Rep": rep_id,
            "p_value": p_test,
            "theta": theta,
        }
    )

    df_p["Type"] = np.where(df_p["theta"] == 0, "null", "non-null")

    return df_p

def collect_pvalues_all_scenarios(nrp_p=10, n_jobs=1):

    tasks_p = list(
        itertools.product(
            range(len(scenario_list)),
            range(nrp_p),
        )
    )

    if n_jobs == 1:
        out_list = [
            collect_pvalues_single_rep(scenario_id, rep_id)
            for scenario_id, rep_id in tqdm(tasks_p, desc="Collecting p-values")
        ]
    else:
        out_list = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="list",
        )(
            delayed(collect_pvalues_single_rep)(scenario_id, rep_id)
            for scenario_id, rep_id in tasks_p
        )

    df_pvals = pd.concat(out_list, ignore_index=True)

    return df_pvals

def plot_conformal_pvalue_histograms(
    df_pvals,
    scenario_list,
    bins=np.linspace(0, 1, 41),
    add_lambda_lines=True,
    save_file=None,
    show=True,
):

    n_scen = len(scenario_list)

    fig, axes = plt.subplots(
        1,
        n_scen,
        figsize=(4.2 * n_scen, 3.6),
        sharex=True,
        sharey=True,
    )

    if n_scen == 1:
        axes = np.array([axes])

    for ax, scenario in zip(axes, scenario_list):
        sid = scenario["sid"]
        title = scenario["title"]

        sub = df_pvals[df_pvals["ScenarioID"] == sid].copy()

        p_null = sub.loc[sub["theta"] == 0, "p_value"].to_numpy()
        p_nonnull = sub.loc[sub["theta"] == 1, "p_value"].to_numpy()

        ax.hist(
            p_null,
            bins=bins,
            density=False,
            histtype="step",
            linewidth=1.6,
            label="null",
        )

        ax.hist(
            p_nonnull,
            bins=bins,
            density=False,
            histtype="stepfilled",
            alpha=0.35,
            label="non-null",
        )

        if add_lambda_lines:
            for lam in [0.2, 0.5, 0.8]:
                ax.axvline(
                    lam,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )

        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Conformal p-value", fontsize=11)

        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Frequency", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))

    if save_file is not None:
        fig.savefig(save_file, bbox_inches="tight")
        print(f"✅ p-value histogram saved to: {save_file}")

    if show:
        plt.show()

    return fig, axes






