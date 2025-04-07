# Us vs HyperFreeDP

## Motivation

In the paper ["Towards Hyperparameter Free Optimization with Differential Privacy"](https://openreview.net/pdf?id=2kGKsyhtvh), the authors claim state-of-the-art results on some of the datasets that we are using.

To assess the effectiveness of their method, we will conduct a comparison.

## Objective

We will train the same models on the same datasets and record the resulting accuracy to compare against HyFreeDP.

## Methodology

As usual, we will fix the number of epochs at 40, run 20 trials of hyperparameter optimization (HPO), and evaluate the model on the test set.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

Although we are comparing against training all parameters, for the sake of training time, we will use FiLM parameterization.

## Datasets

We will conduct the experiment on the following datasets:

- **CIFAR-10** (100% subset)
- **CIFAR-100** (100% subset)
- **dpdl-benchmark/svhn_cropped** (100% subset)

## Epsilon Values

We will run the experiment over ε = { 1, 8 }.

## Results

### ε = 8

| Dataset  | Tobaben et al. | HyFreeDP | DPDL  |
|----------|----------------|----------|-------|
| CIFAR10  | 98.1 ± 0.3     | 97.79    | 98.81 |
| CIFAR100 | 91.2 ± 0.2     | 87.57    | 93.06 |
| SVHN     | 84.4 ± 5.3     | 95.00    | 94.40 |

### ε = 1

| Dataset  | Tobaben et al. | HyFreeDP | DPDL  |
|----------|----------------|----------|-------|
| CIFAR10  | 95.3 ± 1.0     | 95.76    | 98.71 |
| CIFAR100 | 85.7 ± 2.9     | 67.17    | 89.90 |                
| SVHN     | 33.5 ± 14.9    | 88.24    | 91.15 |

In the results, we also include a comparison with numbers from the Tobaben et al. (2023) few-shot papers. Note that for SVHN, the results correspond to 500 shots, which is less than the full dataset.

The NonDP-GS column refers to the grid search baseline in the HyFreeDP paper.
