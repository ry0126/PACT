import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from baselines import (
    storey_pi0,
    adaptive_storey_bh_gao,
    choose_lambda_step_down,
    by_2006_adaptive_bh,
)
from pact import cp_vec, pact_pi0

METHODS = ["BH", "BKY", "Storey-0.2", "Storey-0.5", "Storey-0.8", "LS(Gao)", "PACT"]
ROLES = ["anchor", "low_margin", "high_margin", "low_tail", "high_tail"]
GRID = np.linspace(0.1, 0.9, 9)
SEED_BASES = {1: 224300000, 100: 224400000}


def bh_sorted(p, ordered, estimate, alpha=0.1):
    eligible = np.flatnonzero(
        ordered <= alpha * np.arange(1, len(p) + 1) / (len(p) * estimate)
    )
    return np.zeros(len(p), bool) if not len(eligible) else p <= ordered[eligible[-1]]


def evaluate_arrays(
    p,
    pa,
    theta,
    *,
    case,
    pi1,
    rep,
    seed,
    dimension,
    phase="",
    alpha=0.1,
    ncal=2000,
    naux=4000,
):
    p = np.asarray(p)
    pa = np.asarray(pa)
    theta = np.asarray(theta)
    ordered = np.sort(p)
    aux_ordered = np.sort(pa)
    auxiliary_estimates = np.array([pact_pi0(pa, float(lam), naux) for lam in GRID])
    auxiliary_counts = np.array(
        [int(bh_sorted(pa, aux_ordered, x, alpha).sum()) for x in auxiliary_estimates]
    )
    ties = np.flatnonzero(auxiliary_counts == auxiliary_counts.max())
    chosen = int(np.random.default_rng(seed).choice(ties))
    lam = float(GRID[chosen])
    evaluations = {"BH": (1.0, bh_sorted(p, ordered, 1.0, alpha), np.nan)}
    estimate, rejected = by_2006_adaptive_bh(p, alpha)
    evaluations["BKY"] = (estimate, rejected, np.nan)
    for fixed in [0.2, 0.5, 0.8]:
        estimate = storey_pi0(p, fixed)
        evaluations[f"Storey-{fixed:.1f}"] = (
            estimate,
            bh_sorted(p, ordered, estimate, alpha),
            fixed,
        )
    estimate, rejected = adaptive_storey_bh_gao(
        p, q=alpha, delta=None, lam_max=0.8, robust=True
    )
    evaluations["LS(Gao)"] = (
        estimate,
        rejected,
        choose_lambda_step_down(p, alpha, delta=None, lam_max=0.8, robust=True),
    )
    estimate = pact_pi0(p, lam, ncal)
    evaluations["PACT"] = (estimate, bh_sorted(p, ordered, estimate, alpha), lam)
    rows = []
    for name, (estimate, rejected, selected) in evaluations.items():
        discoveries = int(np.sum(rejected))
        tp = int(np.sum(rejected * theta))
        rows.append(
            dict(
                case=case,
                phase=phase,
                rep=int(rep),
                seed=int(seed),
                method=name,
                rho=float(pi1),
                pi0_true=1 - float(pi1),
                TPP=tp / int(theta.sum()),
                FDP=(discoveries - tp) / max(1, discoveries),
                pi0_hat=float(estimate),
                selected_lambda=selected,
                R=discoveries,
                TP=tp,
                FP=discoveries - tp,
                m=len(p),
                m1=int(theta.sum()),
                ncal=ncal,
                naux=naux,
                weak_p=np.nan,
                aux_ties=len(ties),
                aux_rejections=int(auxiliary_counts[chosen]),
                dimension=int(dimension),
            )
        )
    return rows


def generate_repetition(config, rep, spec, phase=""):
    dimension = int(config["dim"])
    seed = (
        int(spec.get("seed_bases", {}).get(str(dimension), SEED_BASES[dimension])) + rep
    )
    rng = np.random.default_rng(seed)
    null = config.get("null_distribution", dict(family="Gaussian"))

    def draw(n):
        if null["family"].lower() == "laplace":
            return rng.laplace(null.get("loc", 0), null["scale"], size=(n, dimension))
        return rng.normal(size=(n, dimension))

    train = draw(int(spec.get("training_null", 8000)))
    calibration = draw(int(spec.get("calibration_null", 2000)))
    null_test = draw(4000)
    first, second = train_test_split(
        np.arange(len(train)), test_size=0.5, random_state=seed
    )
    options = {
        k: v
        for (k, v) in spec.get("model", {}).items()
        if k in {"kernel", "nu", "gamma", "tol"}
    }
    main = OneClassSVM(**options).fit(train)
    auxiliary = OneClassSVM(**options).fit(train[first])
    assert main.fit_status_ == auxiliary.fit_status_ == 0
    main_cal = main.score_samples(calibration)
    aux_cal = auxiliary.score_samples(train[second])
    main_null = cp_vec(main.score_samples(null_test), main_cal)
    aux_cal_test = cp_vec(auxiliary.score_samples(calibration), aux_cal)
    aux_null = cp_vec(auxiliary.score_samples(null_test), aux_cal)
    components = []
    for component in config["components"]:
        role = ROLES.index(component["role"])
        z = np.random.default_rng(seed + 50000 + 100000 * role).normal(
            size=(2000, dimension)
        )
        x = z * float(component["sd_perp"])
        x[:, 0] = z[:, 0] * float(component["sd"]) + float(component["mu"])
        components.append(
            (
                component,
                cp_vec(main.score_samples(x), main_cal),
                cp_vec(auxiliary.score_samples(x), aux_cal),
            )
        )
    rows = []
    arrays = []
    for pi1 in [0.2, 0.4]:
        n1 = round(5000 * pi1)
        n0 = 5000 - n1
        final = [main_null[:n0]]
        aux = [aux_cal_test, aux_null[:n0]]
        labels = [np.zeros(n0, np.uint8)]
        for component, p, pa in components:
            n = round(n1 * component["weight"])
            final.append(p[:n])
            aux.append(pa[:n])
            labels.append(np.full(n, ROLES.index(component["role"]) + 1, np.uint8))
        p = np.concatenate(final)
        pa = np.concatenate(aux)
        theta = np.r_[np.zeros(n0, int), np.ones(n1, int)]
        assert len(p) == 5000 and len(pa) == 7000
        rows += evaluate_arrays(
            p,
            pa,
            theta,
            case=config["id"],
            pi1=pi1,
            rep=rep,
            seed=seed,
            dimension=dimension,
            phase=phase,
            alpha=spec.get("alpha", 0.1),
        )
        arrays.append(
            dict(
                case=config["id"],
                rho=pi1,
                rep=rep,
                seed=seed,
                p_final=p,
                p_aux=pa,
                theta=theta,
                component=np.concatenate(labels),
            )
        )
    return (pd.DataFrame(rows), arrays)
