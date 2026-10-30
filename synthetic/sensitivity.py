from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
from baselines import (
    storey_pi0,
    by_2006_adaptive_bh,
    run_greedy_ams,
    adaptive_storey_bh_gao,
)
from pact import (
    cp_vec,
    evaluate_rejection,
    _sort_summary_for_lambda_selection,
    effective_lambda,
    pact_bh_reject,
)

SPEC = {}
CONFIGS = {}
SEEDS = {}
METHODS = [
    "BH",
    "BKY",
    "Storey-0.2",
    "Storey-0.5",
    "Storey-0.8",
    "LS(Gao)",
    "greedy-AMS",
    "PACT",
]
MS = [50, 500, 5000]
PIS = [0.2, 0.3, 0.4]
ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
GRID = np.linspace(0.1, 0.9, 9)
ROLES = ["anchor", "low_margin", "high_margin", "low_tail", "high_tail"]
NTRAIN, NCAL, NAUX = (8000, 2000, 4000)


def configure(scenario):
    global SPEC, CONFIGS, SEEDS
    SPEC = json.loads(
        (Path(__file__).resolve().parent / "settings" / f"{scenario}.json").read_text(
            encoding="utf-8"
        )
    )
    CONFIGS = {c["dim"]: c for c in SPEC["configs"]}
    for config in CONFIGS.values():
        config.setdefault(
            "null_distribution",
            {"family": "Gaussian", "mean": 0, "covariance": "identity"},
        )
    SEEDS = {int(k): int(v) for (k, v) in SPEC["seed_bases"].items()}


def component_counts(m1, config):
    expected = m1 * np.array([c["weight"] for c in config["components"]])
    counts = np.floor(expected).astype(int)
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[: m1 - counts.sum()]] += 1
    assert counts.sum() == m1
    return counts


def prepare(dim, rep):
    seed = SEEDS[dim] + rep
    rng = np.random.default_rng(seed)
    null_spec = CONFIGS[dim]["null_distribution"]

    def draw(n):
        if null_spec["family"].lower() == "laplace":
            return rng.laplace(
                null_spec.get("loc", 0), null_spec["scale"], size=(n, dim)
            )
        assert null_spec["family"].lower() == "gaussian"
        return rng.normal(size=(n, dim))

    train = draw(NTRAIN)
    cal = draw(NCAL)
    null = draw(4000)
    i1, i2 = train_test_split(np.arange(NTRAIN), test_size=0.5, random_state=seed)
    main = OneClassSVM().fit(train)
    aux = OneClassSVM().fit(train[i1])
    assert main.fit_status_ == aux.fit_status_ == 0
    mc = main.score_samples(cal)
    ac = aux.score_samples(train[i2])
    pn = cp_vec(main.score_samples(null), mc)
    an = cp_vec(aux.score_samples(null), ac)
    ap = cp_vec(aux.score_samples(cal), ac)
    features, main_p, aux_p = ([], [], [])
    for c in CONFIGS[dim]["components"]:
        rng = np.random.default_rng(seed + 50000 + 100000 * ROLES.index(c["role"]))
        z = rng.normal(size=(2000, dim))
        x = z * float(c["sd_perp"])
        x[:, 0] = z[:, 0] * float(c["sd"]) + float(c["mu"])
        features.append(x)
        main_p.append(cp_vec(main.score_samples(x), mc))
        aux_p.append(cp_vec(aux.score_samples(x), ac))
    return dict(
        dim=dim,
        rep=rep,
        seed=seed,
        train=train,
        cal=cal,
        null=null,
        features=features,
        pn=pn,
        an=an,
        ap=ap,
        main_p=main_p,
        aux_p=aux_p,
    )


def assemble(o, m, pi, include_features=False):
    m1 = int(round(m * pi))
    m0 = m - m1
    counts = component_counts(m1, CONFIGS[o["dim"]])
    p = np.concatenate([o["pn"][:m0]] + [v[:n] for (v, n) in zip(o["main_p"], counts)])
    pa = np.concatenate(
        [o["ap"], o["an"][:m0]] + [v[:n] for (v, n) in zip(o["aux_p"], counts)]
    )
    theta = np.r_[np.zeros(m0, dtype=int), np.ones(m1, dtype=int)]
    assert len(p) == m and len(pa) == NCAL + m
    x = None
    if include_features:
        x = np.concatenate(
            [o["null"][:m0]] + [v[:n] for (v, n) in zip(o["features"], counts)]
        )
    return (p, pa, theta, counts, x)


def pact_from_scores(p, pa, alpha, seed):
    records = []
    for idx, lam in enumerate(GRID):
        lam = float(lam)
        effective_lambda(lam, NCAL)
        pi0, rej = pact_bh_reject(pa, lam, alpha, NAUX)
        lp = effective_lambda(lam, NAUX)
        k = int(np.sum(pa > lam))
        q = k / len(pa)
        var = q * (1 - q) / (len(pa) * (1 - lp) ** 2)
        nr = int(np.sum(rej))
        records.append(
            dict(
                idx=idx,
                **{"lambda": lam},
                lambda_effective=lp,
                n_calibration=NAUX,
                pi_0_modi=pi0,
                Num_Rej_modi=nr,
                K_lam_modi=k,
                q_hat_modi=q,
                var_pi0_modi=var,
                se_pi0_modi=float(np.sqrt(var)),
                objective=float(nr),
            )
        )
    summary = _sort_summary_for_lambda_selection(
        pd.DataFrame(records),
        "Num_Rej_modi",
        False,
        tie_break="random",
        tie_break_random_state=seed,
    )
    lam = float(summary.iloc[0]["lambda"])
    pi0, rej = pact_bh_reject(p, lam, alpha, NCAL)
    return dict(
        pi_0=pi0,
        rej=rej,
        Selected_lambda=lam,
        Selected_lambda_effective=effective_lambda(lam, NCAL),
        Selected_lambda_aux_effective=effective_lambda(lam, NAUX),
        n_calibration_aux=NAUX,
        selection_summary=summary,
    )


def evaluate(p, pa, theta, *, dim, m, pi, alpha, rep, seed):
    meta = dict(
        dimension=dim,
        m=m,
        pi0_true=1 - pi,
        pi1_true=pi,
        alpha=alpha,
        Rep=rep,
        Seed=seed,
    )
    pact = pact_from_scores(p, pa, alpha, seed)
    rows = []
    for name in METHODS:
        lam = np.nan
        if name.startswith("Storey-"):
            pi0 = storey_pi0(p, float(name.split("-")[1]))
            rej = multipletests(p, alpha=alpha / pi0, method="fdr_bh")[0]
        elif name == "BH":
            pi0 = 1
            rej = multipletests(p, alpha=alpha, method="fdr_bh")[0]
        elif name == "BKY":
            pi0, rej = by_2006_adaptive_bh(pvals=p, q=alpha)
        elif name == "LS(Gao)":
            pi0, rej = adaptive_storey_bh_gao(
                p, q=alpha, delta=None, lam_max=0.8, robust=True
            )
        elif name == "greedy-AMS":
            out = run_greedy_ams(
                p_test=p, candi_lam=GRID, alpha=alpha, tie_break="larger_lambda"
            )
            pi0, rej, lam = (out["pi_0"], out["rej"], out["Selected_lambda"])
        elif name == "PACT":
            pi0, rej, lam = (pact["pi_0"], pact["rej"], pact["Selected_lambda"])
        rows.append(
            dict(
                **meta,
                Fixed_para=m,
                pi_value=pi,
                Varied_para=alpha,
                Method=name,
                pi_0=pi0,
                m1=int(theta.sum()),
                TP=int(np.sum(theta * rej)),
                FP=int(np.sum((1 - theta) * rej)),
                n_calibration=NCAL,
                n_calibration_aux=NAUX if name == "PACT" else np.nan,
                Selected_lambda_effective=(
                    pact["Selected_lambda_effective"] if name == "PACT" else np.nan
                ),
                Selected_lambda_aux_effective=(
                    pact["Selected_lambda_aux_effective"] if name == "PACT" else np.nan
                ),
                **evaluate_rejection(theta, rej),
                Selected_lambda=lam,
                Oracle_lambda=np.nan,
            )
        )
    candidates = [{**meta, **r} for r in pact["selection_summary"].to_dict("records")]
    return (rows, candidates)


def work(scenario, dim, rep, m_values=None):
    configure(scenario)
    o = prepare(dim, rep)
    rows, candidates = ([], [])
    for m in MS if m_values is None else m_values:
        for pi in PIS:
            p, pa, theta, counts, _ = assemble(o, m, pi)
            for alpha in ALPHAS:
                r, c = evaluate(
                    p,
                    pa,
                    theta,
                    dim=dim,
                    m=m,
                    pi=pi,
                    alpha=alpha,
                    rep=rep,
                    seed=o["seed"],
                )
                rows.extend(r)
                candidates.extend(c)
    return (dim, rep, rows, candidates)
