

def storey_pi0(pvals, lam):

    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    if not (0 <= lam < 1):
        raise ValueError("lambda must satisfy 0 <= lambda < 1.")

    return (1.0 + np.sum(pvals > lam)) / (n * (1.0 - lam))

def storey_bh_reject(pvals, lam, alpha):

    pi_0 = storey_pi0(pvals, lam)

    rej = multipletests(
        pvals,
        alpha=alpha / pi_0,
        method="fdr_bh",
    )[0]

    rej = np.asarray(rej).reshape(-1).astype(int)

    return pi_0, rej



def storey_pi0_robust(pvals, lam):

    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    pi0 = storey_pi0(pvals, lam)

    var_hat = (1.0 / n) * pi0 * (1.0 / (1.0 - lam) - pi0)


    var_hat = max(var_hat, 0.0)

    return pi0 + np.sqrt(var_hat)


def make_lambda_grid(q, delta, lam_max=1.0):

    if not (0 < q < 1):
        raise ValueError("q must satisfy 0 < q < 1.")

    if not (q < lam_max <= 1):
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


def choose_lambda_step_down(
    pvals,
    q,
    delta=None,
    lam_max=1.0,
    robust=False,
):

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


def adaptive_storey_bh_gao(
    pvals,
    q=0.05,
    delta=None,
    lam_max=1.0,
    robust=False,
):

    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("All p-values must be in [0, 1].")

    lambda_hat = choose_lambda_step_down(
        pvals=pvals,
        q=q,
        delta=delta,
        lam_max=lam_max,
        robust=robust,
    )

    pi0_hat = storey_pi0(pvals, lambda_hat)

    p_sorted = np.sort(pvals)
    p_candidates = p_sorted[p_sorted < q]

    if len(p_candidates) == 0:
        reject = np.zeros(n, dtype=bool)
        return pi0_hat, reject

    k = np.arange(1, len(p_candidates) + 1)

    estimated_fdp = pi0_hat * n * p_candidates / k
    ok = estimated_fdp <= q

    if not np.any(ok):
        reject = np.zeros(n, dtype=bool)
    else:
        tau_hat = p_candidates[np.where(ok)[0][-1]]
        reject = pvals <= tau_hat

    return pi0_hat, reject



def by_2006_adaptive_bh(pvals, q=0.05):

    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)

    if np.any((pvals < 0) | (pvals > 1)):
        raise ValueError("All p-values must be in [0, 1].")

    if not (0 < q < 1):
        raise ValueError("q must be in (0, 1).")

    q_reduced = q / (1.0 + q)

    reject_stage1 = multipletests(
        pvals,
        alpha=q_reduced,
        method="fdr_bh"
    )[0]

    R1 = int(np.sum(reject_stage1))


    pi0_hat = (1.0 + q) * (1.0 - R1 / n)


    if R1 == n:

        reject = np.ones(n, dtype=bool)
        pi0_hat = 0.0
        return pi0_hat, reject

    alpha_eff = q / pi0_hat


    alpha_eff = min(alpha_eff, 1.0 - 1e-12)

    reject = multipletests(
        pvals,
        alpha=alpha_eff,
        method="fdr_bh"
    )[0]

    return pi0_hat, reject


def run_oracle_candidates(
    *,
    p_test,
    theta,
    candi_lam,
    alpha,
    fixed_para,
    varied_para,
    rep_id,
):

    results = []

    for lam in candi_lam:
        pi_0, rej = storey_bh_reject(
            pvals=p_test,
            lam=lam,
            alpha=alpha,
        )

        eval_dict = evaluate_rejection(theta, rej)

        results.append(
            {
                "Fixed_para": fixed_para,
                "Varied_para": varied_para,
                "Method": "oracle_candidate",
                "pi_0": pi_0,
                "FDR": eval_dict["FDR"],
                "Power": eval_dict["Power"],
                "Num_Rej": eval_dict["Num_Rej"],
                "Selected_lambda": np.nan,
                "Oracle_lambda": float(lam),
                "Rep": rep_id,
            }
        )

    return results

def run_oracle_candidates_box(
    *,
    p_test,
    theta,
    candi_lam,
    alpha,
    scenario_id,
    scenario_title,
    rep_id,
):

    results = []

    for lam in candi_lam:
        lam = float(lam)

        pi_0, rej = storey_bh_reject(
            pvals=p_test,
            lam=lam,
            alpha=alpha,
        )

        eval_dict = evaluate_rejection(theta, rej)

        results.append(
            {
                "ScenarioID": scenario_id,
                "ScenarioTitle": scenario_title,
                "Method": "oracle_candidate",

                "pi_0": pi_0,
                "FDR": eval_dict["FDR"],
                "Power": eval_dict["Power"],
                "Num_Rej": eval_dict["Num_Rej"],

                "Selected_lambda": np.nan,
                "Oracle_lambda": lam,

                "Rep": rep_id,
            }
        )

    return results


def run_greedy_ams(
    *,
    p_test,
    candi_lam,
    alpha,
    tie_break="larger_lambda",
):

    records = []

    for idx, lam in enumerate(candi_lam):
        pi_0_cur, rej_cur = storey_bh_reject(
            pvals=p_test,
            lam=lam,
            alpha=alpha,
        )

        records.append(
            {
                "idx": idx,
                "lambda": float(lam),
                "pi_0": pi_0_cur,
                "Num_Rej": int(np.sum(rej_cur)),
            }
        )

    summary = pd.DataFrame(records)

    if tie_break == "larger_lambda":
        summary = summary.sort_values(
            by=["Num_Rej", "lambda"],
            ascending=[False, False],
        )
    elif tie_break == "smaller_lambda":
        summary = summary.sort_values(
            by=["Num_Rej", "lambda"],
            ascending=[False, True],
        )
    elif tie_break == "first":
        summary = summary.sort_values(
            by=["Num_Rej", "idx"],
            ascending=[False, True],
        )
    else:
        raise ValueError("tie_break must be 'larger_lambda', 'smaller_lambda', or 'first'.")

    lam_sel = float(summary.iloc[0]["lambda"])

    pi_0, rej = storey_bh_reject(
        pvals=p_test,
        lam=lam_sel,
        alpha=alpha,
    )

    return {
        "pi_0": pi_0,
        "rej": rej,
        "Selected_lambda": lam_sel,
        "selection_summary": summary,
    }




def run_ams_new(
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
    selection_criterion="num_rej",
    gamma=1.0,
):

    allowed_criteria = {
        "num_rej",
        "pi0",
        "pi0_plus_var",
        "pi0_plus_se",
    }

    if selection_criterion not in allowed_criteria:
        raise ValueError(
            f"selection_criterion must be one of {allowed_criteria}, "
            f"but got {selection_criterion!r}."
        )

    if gamma < 0:
        raise ValueError("gamma must be nonnegative.")

    X_tr_1, X_tr_2 = train_test_split(
        X_tr,
        test_size=0.5,
        random_state=split_random_state,
        shuffle=True,
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

        if not (0 <= lam < 1):
            raise ValueError(
                f"All candidate lambda values must satisfy 0 <= lambda < 1. "
                f"Got lambda={lam}."
            )

        pi_0_modi, rej_modi = storey_bh_reject(
            pvals=p_test_modi,
            lam=lam,
            alpha=alpha,
        )

        num_rej_modi = int(np.sum(rej_modi))

        # 用 modified p-values 估计 Storey estimator 的方差和 SE
        K_lam = int(np.sum(p_test_modi >= lam))
        q_hat = K_lam / m_modi

        var_pi0_modi = q_hat * (1.0 - q_hat) / (
                m_modi * (1.0 - lam) ** 2
        )
        se_pi0_modi = np.sqrt(var_pi0_modi)


        if selection_criterion == "num_rej":
            objective = num_rej_modi

        elif selection_criterion == "pi0":
            objective = pi_0_modi

        elif selection_criterion == "pi0_plus_var":
            objective = pi_0_modi + gamma * var_pi0_modi

        elif selection_criterion == "pi0_plus_se":
            objective = pi_0_modi + gamma * se_pi0_modi

        else:
            raise ValueError(f"Unknown selection_criterion: {selection_criterion}")

        records.append(
            {
                "idx": idx,
                "lambda": lam,
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

        if tie_break == "min_pi0_plus_se":
            summary["pi0_plus_se"] = (
                    summary["pi_0_modi"] + gamma * summary["se_pi0_modi"]
            )

            summary = summary.sort_values(
                by=["Num_Rej_modi", "pi0_plus_se", "lambda"],
                ascending=[False, True, False],
                kind="mergesort",
            )

        elif tie_break == "min_pi0":
            summary = summary.sort_values(
                by=["Num_Rej_modi", "pi_0_modi", "lambda"],
                ascending=[False, True, False],
                kind="mergesort",
            )

        elif tie_break == "larger_lambda":
            summary = summary.sort_values(
                by=["Num_Rej_modi", "lambda"],
                ascending=[False, False],
                kind="mergesort",
            )

        elif tie_break == "smaller_lambda":
            summary = summary.sort_values(
                by=["Num_Rej_modi", "lambda"],
                ascending=[False, True],
                kind="mergesort",
            )

        elif tie_break == "first":
            summary = summary.sort_values(
                by=["Num_Rej_modi", "idx"],
                ascending=[False, True],
                kind="mergesort",
            )

        else:
            raise ValueError(
                "For selection_criterion='num_rej', tie_break must be one of "
                "'larger_lambda', 'smaller_lambda', 'first', 'min_pi0', "
                "or 'min_pi0_plus_se'."
            )

    else:

        if tie_break == "larger_lambda":
            summary = summary.sort_values(
                by=["objective", "lambda"],
                ascending=[True, False],
                kind="mergesort",
            )

        elif tie_break == "smaller_lambda":
            summary = summary.sort_values(
                by=["objective", "lambda"],
                ascending=[True, True],
                kind="mergesort",
            )

        elif tie_break == "first":
            summary = summary.sort_values(
                by=["objective", "idx"],
                ascending=[True, True],
                kind="mergesort",
            )

        else:
            raise ValueError(
                "For pi0-type criteria, tie_break must be one of "
                "'larger_lambda', 'smaller_lambda', or 'first'."
            )

    lam_sel = float(summary.iloc[0]["lambda"])


    pi_0, rej = storey_bh_reject(
        pvals=p_test,
        lam=lam_sel,
        alpha=alpha,
    )

    return {
        "pi_0": pi_0,
        "rej": rej,
        "Selected_lambda": lam_sel,
        "selection_summary": summary,
        "p_test_modi": p_test_modi,
        "selection_criterion": selection_criterion,
        "gamma": gamma,
    }


def finalize_oracle_by_mean_num_rej_box(
        df,
        *,
        tie_break="larger_lambda",
):
    df = df.copy()

    oracle_cand = df[df["Method"] == "oracle_candidate"].copy()
    df_other = df[df["Method"] != "oracle_candidate"].copy()

    if oracle_cand.empty:
        return df, pd.DataFrame()

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
    missing_cols = [col for col in required_cols if col not in oracle_cand.columns]
    if missing_cols:
        raise ValueError(f"oracle_cand 缺少以下列: {missing_cols}")

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
        raise ValueError("tie_break must be 'larger_lambda' or 'smaller_lambda'.")

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
        oracle_final["Oracle_lambda"] == oracle_final["Selected_lambda_oracle"]
        ].copy()

    oracle_final["Method"] = "oracle"
    oracle_final["Selected_lambda"] = oracle_final["Oracle_lambda"]
    oracle_final = oracle_final.drop(columns=["Selected_lambda_oracle"])

    df_plot = pd.concat([df_other, oracle_final], ignore_index=True)

    return df_plot, oracle_summary


def fit_cifar_occ_and_get_pvalues(
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
        pca_dim=10,
        random_state=base_seed,
    )

    base_mod = clone(exp_config["occ"])
    if "random_state" in base_mod.get_params(deep=False):
        base_mod.set_params(random_state=base_seed)

    base_mod.fit(X_tr)

    sco_cal = base_mod.score_samples(X_cal)
    sco_test = base_mod.score_samples(X_test)

    p_test = cp_vec(sco_test, sco_cal)
    p_test = np.asarray(p_test, dtype=float).reshape(-1)

    return {
        "scenario": scenario,
        "base_seed": base_seed,
        "X_tr": X_tr,
        "X_cal": X_cal,
        "X_test": X_test,
        "theta": theta,
        "p_test": p_test,
    }


def collect_pvalues_single_rep_cifar(
        scenario_id,
        rep_id,
        *,
        features_train,
        labels_train,
        features_test,
        labels_test,
        exp_config,
):
    out = fit_cifar_occ_and_get_pvalues(
        scenario_id=scenario_id,
        rep_id=rep_id,
        features_train=features_train,
        labels_train=labels_train,
        features_test=features_test,
        labels_test=labels_test,
        exp_config=exp_config,
    )

    scenario = out["scenario"]
    theta = out["theta"]
    p_test = out["p_test"]

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


def collect_pvalues_all_scenarios_cifar(
        *,
        features_train,
        labels_train,
        features_test,
        labels_test,
        exp_config,
        nrp_p=20,
        n_jobs=1,
):
    tasks_p = list(
        itertools.product(
            range(len(exp_config["scenarios"])),
            range(nrp_p),
        )
    )

    if n_jobs == 1:
        out_list = [
            collect_pvalues_single_rep_cifar(
                scenario_id=scenario_id,
                rep_id=rep_id,
                features_train=features_train,
                labels_train=labels_train,
                features_test=features_test,
                labels_test=labels_test,
                exp_config=exp_config,
            )
            for scenario_id, rep_id in tqdm(tasks_p, desc="Collecting CIFAR p-values")
        ]
    else:
        out_list = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="list",
        )(
            delayed(collect_pvalues_single_rep_cifar)(
                scenario_id=scenario_id,
                rep_id=rep_id,
                features_train=features_train,
                labels_train=labels_train,
                features_test=features_test,
                labels_test=labels_test,
                exp_config=exp_config,
            )
            for scenario_id, rep_id in tasks_p
        )

    df_pvals = pd.concat(out_list, ignore_index=True)
    return df_pvals


def plot_conformal_pvalue_histograms_cifar(
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
        figsize=(4.3 * n_scen, 3.7),
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


def make_lambda_grid_for_plot(q=0.1, delta=0.01, lam_max=0.8):
    q = float(q)
    delta = float(delta)
    lam_max = float(lam_max)

    if delta <= 0:
        return np.array([lam_max], dtype=float)

    grid = list(np.arange(q, lam_max + 0.5 * delta, delta))
    grid = [x for x in grid if x <= lam_max + 1e-12]

    if len(grid) == 0 or abs(grid[-1] - lam_max) > 1e-10:
        grid.append(lam_max)

    grid = np.unique(np.round(np.asarray(grid, dtype=float), 12))
    grid = grid[(grid >= q - 1e-12) & (grid <= lam_max + 1e-12)]
    return grid


def storey_pi0_robust_for_plot(pvals, lam):
    global _ROBUST_FALLBACK_WARNED

    if _storey_pi0_robust_diag is not None:
        return _storey_pi0_robust_diag(pvals, lam)

    if not _ROBUST_FALLBACK_WARNED:
        print(
            "⚠️ storey_pi0_robust was not found. "
            "Using capped fallback min(storey_pi0, 1.0) for diagnostic plots."
        )
        _ROBUST_FALLBACK_WARNED = True

    return float(min(storey_pi0(pvals, lam), 1.0))


def trace_gao_lambda_selection_for_plot(
        pvals,
        q=0.1,
        lam_max=0.8,
        robust=True,
):
    pvals = np.asarray(pvals, dtype=float).reshape(-1)
    tail_count = np.sum(pvals > q)

    if tail_count == 0:
        delta = lam_max - q
    else:
        delta = 50.0 / tail_count

    if delta <= 0 or delta > lam_max - q:
        delta = lam_max - q

    grid = make_lambda_grid_for_plot(q=q, delta=delta, lam_max=lam_max)

    estimator = storey_pi0_robust_for_plot if robust else storey_pi0
    values = np.array([estimator(pvals, lam) for lam in grid])

    selected_j = len(grid) - 1
    for j in range(1, len(grid)):
        if values[j - 1] <= values[j]:
            selected_j = j
            break

    path = pd.DataFrame(
        {
            "j": np.arange(len(grid)),
            "lambda_j": grid,
            "path_value": values,
        }
    )

    return {
        "delta": float(delta),
        "tail_count": int(tail_count),
        "selected_j": int(selected_j),
        "selected_lambda": float(grid[selected_j]),
        "path": path,
    }


def run_single_rep_storey_path_cifar(
        scenario_id,
        rep_id,
        *,
        features_train,
        labels_train,
        features_test,
        labels_test,
        exp_config,
        lam_plot_grid=np.linspace(0.1, 0.9, 20),
        our_method="AMS-ori",
):
    out = fit_cifar_occ_and_get_pvalues(
        scenario_id=scenario_id,
        rep_id=rep_id,
        features_train=features_train,
        labels_train=labels_train,
        features_test=features_test,
        labels_test=labels_test,
        exp_config=exp_config,
    )

    scenario = out["scenario"]
    base_seed = out["base_seed"]
    X_tr = out["X_tr"]
    X_cal = out["X_cal"]
    X_test = out["X_test"]
    theta = out["theta"]
    p_test = out["p_test"]

    alpha = exp_config["alpha"]
    candi_lam = exp_config["candi_lam"]

    path_records = []
    for lam in lam_plot_grid:
        if lam >= 1:
            continue

        pi0 = storey_pi0(p_test, lam)
        pi0_rb = storey_pi0_robust_for_plot(p_test, lam)

        path_records.append(
            {
                "ScenarioID": scenario["sid"],
                "ScenarioTitle": scenario["title"],
                "Rep": rep_id,
                "lambda": float(lam),
                "pi0": float(pi0),
                "pi0_robust": float(pi0_rb),
            }
        )

    df_path = pd.DataFrame(path_records)

    gao_trace = trace_gao_lambda_selection_for_plot(
        pvals=p_test,
        q=alpha,
        lam_max=0.8,
        robust=True,
    )
    gao_lambda = float(gao_trace["selected_lambda"])
    gao_pi0, gao_rej = adaptive_storey_bh_gao(
        p_test,
        q=alpha,
        delta=None,
        lam_max=0.8,
        robust=True,
    )
    gao_eval = evaluate_rejection(theta, gao_rej)

    if our_method == "AMS-ori":
        out_ours = run_ams_new(
            X_tr=X_tr,
            X_cal=X_cal,
            X_test=X_test,
            p_test=p_test,
            occ=exp_config["occ"],
            candi_lam=candi_lam,
            alpha=alpha,
            base_seed=base_seed,
            split_random_state=base_seed,
            tie_break=exp_config["tie_break_ori"],
            tie_break_random_state=base_seed,
            selection_criterion="num_rej",
            gamma=0.0,
        )
    elif our_method == "AMS-rob":
        out_ours = run_ams_new(
            X_tr=X_tr,
            X_cal=X_cal,
            X_test=X_test,
            p_test=p_test,
            occ=exp_config["occ"],
            candi_lam=candi_lam,
            alpha=alpha,
            base_seed=base_seed,
            split_random_state=base_seed,
            tie_break=exp_config["tie_break_rob"],
            selection_criterion=exp_config["selection_cri_rob"],
            gamma=exp_config.get("gamma_rob"),
        )
    else:
        raise ValueError("our_method must be 'AMS-ori' or 'AMS-rob'.")

    our_lambda = float(out_ours["Selected_lambda"])
    our_pi0 = float(out_ours["pi_0"])
    our_rej = np.asarray(out_ours["rej"], dtype=bool)
    our_eval = evaluate_rejection(theta, our_rej)

    gao_path_pi0_at_lambda = float(storey_pi0_robust_for_plot(p_test, gao_lambda))
    our_path_pi0_at_lambda = float(storey_pi0(p_test, our_lambda))

    df_sel = pd.DataFrame(
        [
            {
                "ScenarioID": scenario["sid"],
                "ScenarioTitle": scenario["title"],
                "Rep": rep_id,
                "Method": "Gao",
                "Selected_lambda": gao_lambda,
                "pi_0": float(gao_pi0),
                "Path_pi0_at_Selected_lambda": gao_path_pi0_at_lambda,
                "Num_Rej": float(gao_eval["Num_Rej"]),
                "FDR": float(gao_eval["FDR"]),
                "Power": float(gao_eval["Power"]),
            },
            {
                "ScenarioID": scenario["sid"],
                "ScenarioTitle": scenario["title"],
                "Rep": rep_id,
                "Method": our_method,
                "Selected_lambda": our_lambda,
                "pi_0": our_pi0,
                "Path_pi0_at_Selected_lambda": our_path_pi0_at_lambda,
                "Num_Rej": float(our_eval["Num_Rej"]),
                "FDR": float(our_eval["FDR"]),
                "Power": float(our_eval["Power"]),
            },
        ]
    )

    return df_path, df_sel


def collect_storey_path_data_cifar(
        *,
        features_train,
        labels_train,
        features_test,
        labels_test,
        exp_config,
        nrp_plot=20,
        lam_plot_grid=np.linspace(0.1, 0.9, 20),
        our_method="AMS-ori",
        n_jobs=-1,
):
    tasks = list(
        itertools.product(
            range(len(exp_config["scenarios"])),
            range(nrp_plot),
        )
    )

    if n_jobs == 1:
        out_list = [
            run_single_rep_storey_path_cifar(
                scenario_id=scenario_id,
                rep_id=rep_id,
                features_train=features_train,
                labels_train=labels_train,
                features_test=features_test,
                labels_test=labels_test,
                exp_config=exp_config,
                lam_plot_grid=lam_plot_grid,
                our_method=our_method,
            )
            for scenario_id, rep_id in tqdm(tasks, desc="Collecting CIFAR Storey paths")
        ]
    else:
        out_list = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="list",
        )(
            delayed(run_single_rep_storey_path_cifar)(
                scenario_id=scenario_id,
                rep_id=rep_id,
                features_train=features_train,
                labels_train=labels_train,
                features_test=features_test,
                labels_test=labels_test,
                exp_config=exp_config,
                lam_plot_grid=lam_plot_grid,
                our_method=our_method,
            )
            for scenario_id, rep_id in tasks
        )

    df_path_all = pd.concat([x[0] for x in out_list], ignore_index=True)
    df_sel_all = pd.concat([x[1] for x in out_list], ignore_index=True)
    return df_path_all, df_sel_all


def plot_storey_path_with_selected_lambdas_cifar(
        df_path_all,
        df_sel_all,
        scenario_list,
        figsize=None,
        show=True,
        save_file=None,
):
    if figsize is None:
        figsize = (4.8 * len(scenario_list), 4.0)

    fig, axes = plt.subplots(
        1,
        len(scenario_list),
        figsize=figsize,
        sharex=True,
        sharey=False,
    )

    if len(scenario_list) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenario_list):
        sid = scenario["sid"]
        title = scenario["title"]

        sub_path = df_path_all[df_path_all["ScenarioID"] == sid].copy()
        sub_sel = df_sel_all[df_sel_all["ScenarioID"] == sid].copy()

        path_sum = (
            sub_path
            .groupby("lambda", as_index=False)
            .agg(
                pi0_mean=("pi0", "mean"),
                pi0_sd=("pi0", "std"),
                pi0_rb_mean=("pi0_robust", "mean"),
                pi0_rb_sd=("pi0_robust", "std"),
            )
            .sort_values("lambda")
        )

        lam = path_sum["lambda"].to_numpy()

        ax.plot(
            lam,
            path_sum["pi0_mean"],
            label=r"Storey $\hat{\pi}_0(\lambda)$",
            linewidth=2,
        )
        ax.fill_between(
            lam,
            path_sum["pi0_mean"] - path_sum["pi0_sd"].fillna(0),
            path_sum["pi0_mean"] + path_sum["pi0_sd"].fillna(0),
            alpha=0.15,
        )

        ax.plot(
            lam,
            path_sum["pi0_rb_mean"],
            linestyle="--",
            label=r"Robust $\hat{\pi}_{0,\mathrm{RB}}(\lambda)$",
            linewidth=2,
        )
        ax.fill_between(
            lam,
            path_sum["pi0_rb_mean"] - path_sum["pi0_rb_sd"].fillna(0),
            path_sum["pi0_rb_mean"] + path_sum["pi0_rb_sd"].fillna(0),
            alpha=0.12,
        )

        gao_sub = sub_sel[sub_sel["Method"] == "Gao"]
        if not gao_sub.empty:
            gao_lam_mean = gao_sub["Selected_lambda"].mean()
            ax.axvline(
                gao_lam_mean,
                color="green",
                linestyle=":",
                linewidth=2.2,
                label=r"Gao mean selected $\lambda$",
            )
            if "pi_0" in gao_sub.columns:
                ax.scatter(
                    [gao_lam_mean],
                    [gao_sub["pi_0"].mean()],
                    color="green",
                    s=36,
                    zorder=5,
                    label=r"Gao mean final $\hat\pi_0$",
                )

        our_methods = [m for m in sub_sel["Method"].unique() if m != "Gao"]
        for method_name in our_methods:
            our_sub = sub_sel[sub_sel["Method"] == method_name]
            our_lam_mean = our_sub["Selected_lambda"].mean()
            ax.axvline(
                our_lam_mean,
                color="red",
                linestyle="-.",
                linewidth=2.2,
                label=fr"{method_name} mean selected $\lambda$",
            )
            if "pi_0" in our_sub.columns:
                ax.scatter(
                    [our_lam_mean],
                    [our_sub["pi_0"].mean()],
                    color="red",
                    s=36,
                    zorder=5,
                    label=fr"{method_name} mean final $\hat\pi_0$",
                )

        if "pi_out" in scenario:
            true_pi0 = 1.0 - float(scenario["pi_out"])

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$\lambda$", fontsize=12)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"Storey path: $\hat{\pi}_0(\lambda)$", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="lower center",
        ncol=min(5, max(1, len(uniq))),
        frameon=False,
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))

    if save_file is not None:
        fig.savefig(save_file, bbox_inches="tight")
        print(f"✅ Storey path plot saved to: {save_file}")

    if show:
        plt.show()

    return fig, axes


def summarize_selected_lambda_diagnostics(df_sel_all, our_method="AMS-ori"):
    if df_sel_all is None or df_sel_all.empty:
        return pd.DataFrame()

    value_cols = [
        c for c in [
            "Selected_lambda",
            "pi_0",
            "Path_pi0_at_Selected_lambda",
            "Num_Rej",
            "FDR",
            "Power",
        ]
        if c in df_sel_all.columns
    ]

    summary = (
        df_sel_all
        .groupby(["ScenarioID", "ScenarioTitle", "Method"], as_index=False)[value_cols]
        .mean()
    )

    print("\n===== Selected-lambda diagnostic summary =====")
    print(summary.to_string(index=False))

    needed_methods = {"Gao", our_method}
    if needed_methods.issubset(set(df_sel_all["Method"].unique())) and {"pi_0", "Num_Rej"}.issubset(df_sel_all.columns):
        wide = df_sel_all.pivot_table(
            index=["ScenarioID", "ScenarioTitle", "Rep"],
            columns="Method",
            values=["pi_0", "Num_Rej", "Selected_lambda"],
            aggfunc="first",
        )

        rows = []
        for sid, scenario_title in df_sel_all[["ScenarioID", "ScenarioTitle"]].drop_duplicates().itertuples(
                index=False):
            sub = wide.xs((sid, scenario_title), level=("ScenarioID", "ScenarioTitle"), drop_level=False)
            if ("pi_0", our_method) in sub.columns and ("pi_0", "Gao") in sub.columns:
                impossible = (
                        (sub[("pi_0", our_method)] > sub[("pi_0", "Gao")])
                        & (sub[("Num_Rej", our_method)] > sub[("Num_Rej", "Gao")])
                )
                rows.append(
                    {
                        "ScenarioID": sid,
                        "ScenarioTitle": scenario_title,
                        "n_rep": int(sub.shape[0]),
                        "n_our_pi0_larger_and_more_rej": int(impossible.sum()),
                    }
                )

        check = pd.DataFrame(rows)
        print("\n===== Monotonicity sanity check =====")
        print(check.to_string(index=False))

    return summary

