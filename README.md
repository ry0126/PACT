# PACT

This repository contains code for PACT, an adaptive parameter tuning Storey-type procedure for false discovery rate (FDR) control in large-scale out-of-distribution (OOD) testing with conformal p-values.

## Structure

```text
PACT/
├── functions.py
├── simu/
│   ├── fixed_alpha.py
│   └── varying_alpha.py
└── real_data/
    ├── cifar_data_prep.py
    ├── into_lambda.py
    └── three_tasks.py

```
### `functions.py`

This file contains the core functions used throughout the experiments.


### `simu/fixed_alpha.py`

This folder contains the code for the main synthetic experiments at a fixed target FDR level.

These experiments correspond to **Section 4.1** of the paper. They consider four Gaussian-mixture OOD testing scenarios:

- low-dimensional weak signal;
- low-dimensional mixture signal;
- high-dimensional weak signal;
- high-dimensional mixture signal.

### `simu/varying_alpha.py`

This folder contains the code for additional synthetic experiments with varying target FDR levels, test sample sizes, and null proportions.

These experiments correspond to **Section 4.2** of the paper. They study the finite-sample behavior of greedy tuning and illustrate the role of permutation invariance in PACT.

### `real_data/cifar_data_prep.py`

This file provides utility functions for preparing the CIFAR-10 data used in the real-data experiments.

### `real_data/into_lambda.py`

This folder contains the code for the CIFAR-10 diagnostic experiment that illustrates the sensitivity of Storey-BH to the tuning parameter $\lambda$.

This experiment corresponds to the introductory CIFAR-10 example in **Figure 1** of the paper. It evaluates how the number of rejections and Storey's null-proportion estimate $\widehat{\pi}_0(\lambda)$ vary with $\lambda$, both averaged over repeated random splits and in representative single realizations.

### `real_data/three_tasks.py`

This folder contains the code for the main CIFAR-10 real-data experiments in **Section 5** of the paper.

The experiments consider three OOD testing tasks:

- land vehicles vs. airplanes;
- land vehicles vs. air/sea vehicles;
- land vehicles vs. hoofed mammals.


## Data Source
The real-data experiments use the CIFAR-10 dataset. CIFAR-10 contains 60,000 color images of size 32 x 32 from 10 classes, with 50,000 training images and 10,000 test images. The dataset can be downloaded from the official CIFAR-10 webpage: https://www.cs.toronto.edu/~kriz/cifar.html
