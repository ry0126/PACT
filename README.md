# PACT

This repository contains code for PACT, an adaptive parameter tuning Storey-type procedure for false discovery rate (FDR) control in large-scale out-of-distribution (OOD) testing with conformal p-values.

## Structure

```text
PACT/
├── functions.py
├── simu/
│   ├── fixed_alpha
│   └── varying_alpha
└── real_data/
    ├── into_lambda
    └── three_tasks

```

## Data Source
The real-data experiments use the CIFAR-10 dataset. CIFAR-10 contains 60,000 color images of size 32 x 32 from 10 classes, with 50,000 training images and 10,000 test images. The dataset can be downloaded from the official CIFAR-10 webpage: https://www.cs.toronto.edu/~kriz/cifar.html
