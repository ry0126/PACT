# PACT

Selected source code for **Permutation-invariant Adaptive Conformal Tuning (PACT)**, accompanying *Permutation-Invariant Conformal Tuning for Adaptive False Discovery Rate Control* by Rongyi Sun, Wangcheng Li, and Wenguang Sun.

PACT selects the tuning parameter of a Storey-type multiple-testing procedure for out-of-distribution detection with conformal p-values.

## Release scope

This repository presents the core method and the main experimental constructions in the revised manuscript. It includes algorithms, synthetic data generation, scenario parameters, and CIFAR-10 sampling and preprocessing. 
Raw data, feature caches, fitted models, numerical results, plotting scripts, parameter searches, and supplementary experimental variants are omitted. 

## Core method

PACT splits the reference training sample into auxiliary training and calibration subsets. The auxiliary model scores the **pooled final calibration and test observations**, and the candidate tuning parameter is selected by maximizing the number of auxiliary rejections. The selected parameter is then transferred to the final conformal testing problem, whose model is fitted on the full reference training sample. Pooling calibration and test observations in the auxiliary problem is central to permutation invariance.

The implementation retains the finite-sample correction for discrete conformal p-values:

```math
\lambda^+(n)=\frac{\lfloor(n+1)\lambda\rfloor+1}{n+1},
\qquad
\widehat{\pi}_0^+(\lambda)=\frac{1+\sum_{j=1}^{m}\mathbf{1}(p_j>\lambda)}{m(1-\lambda^+(n))}.
```

Here, $n$ is the calibration size of the stage being evaluated. The numerator uses the strict nominal tail $p_j>\lambda$; only the denominator uses $\lambda^+(n)$. The estimate is not clipped at one. The auxiliary and final stages use their own calibration sizes, and the candidate grid must give a positive corrected denominator in both stages.

`run_pact_discrete` implements auxiliary tuning and final rejection using supplied target p-values. The main experiments use `selection_criterion="num_rej"` and seeded random tie-breaking. Setting `selection_criterion="pi0_plus_se"` with `gamma=1` implements the PACT-R selection criterion in Appendix A. `cp_vec` accepts OneClassSVM normality scores, for which larger scores indicate inliers, and counts calibration scores less than or equal to each test score.

## Source files

```text
PACT/
├── pact.py
├── baselines.py
├── synthetic/
│   ├── generation.py
│   ├── sensitivity.py
│   └── settings/{S1,S2,S3}.json
├── real_data/
│   ├── cifar_data.py
│   ├── experiment.py
│   └── settings.json
└── requirements.txt
```

| File | Content | Manuscript |
| --- | --- | --- |
| `pact.py` | Conformal p-values, discrete Storey correction, auxiliary tuning, final testing, and rejection metrics | Sections 2-3; Algorithm 1 |
| `baselines.py` | Fixed-parameter Storey-BH, BKY, the Gao stopping-rule procedure, and greedy tuning | Sections 4-5 |
| `synthetic/generation.py` | Null and heterogeneous alternative generation, fitted OneClassSVM scores, and main method comparisons | Section 4.1 |
| `synthetic/sensitivity.py` | Sample-size and FDR-level sweeps, component-count allocation, and greedy-tuning comparison | Section 4.2; Appendix D.2 |
| `synthetic/settings/` | Distribution parameters and sample allocations for S1-S3, using manuscript numbering | Section 4 |
| `real_data/cifar_data.py` | Feature-cache loading, stratified sampling without replacement, training-only standardization, and PCA | Section 5 |
| `real_data/experiment.py` | Configuration loading and one-repetition comparison of the seven main methods | Section 5 |

## Synthetic settings

`generate_repetition` accepts a dimension-specific configuration and its scenario settings. In the sensitivity module, `configure("S1")` selects a manuscript scenario, `prepare` constructs its shared score basis, and `assemble` forms a test mixture with the requested size and non-null proportion.

## CIFAR-10 settings

All three tasks use automobile and truck as inlier classes:

| Scenario | Outlier classes | CIFAR-10 class IDs |
| --- | --- | --- |
| S1 | Airplane | 0 |
| S2 | Airplane and ship | 0, 8 |
| S3 | Deer and horse | 4, 7 |

The cache loader expects the arrays `features_train`, `labels_train`, `features_test`, and `labels_test`; pass the local cache path explicitly. `load_config` creates the model and candidate grid required by `run_single_rep`.

Dataset source: [official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html).

## Dependencies

The selected code depends on NumPy, pandas, scikit-learn, and statsmodels, listed in `requirements.txt`. The file lists the required libraries, without specifying an environment for exact reproduction. Algorithm descriptions, assumptions, and theoretical guarantees are given in the manuscript.
