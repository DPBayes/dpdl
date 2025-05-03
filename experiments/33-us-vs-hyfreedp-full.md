# Full comparison: Us vs HyperFreeDP

## Motivation

In the paper [Towards Hyperparameter Free Optimization with Differential Privacy](https://openreview.net/pdf?id=2kGKsyhtvh), the authors claim state-of-the-art results.

We will use this as a baseline and need a comparison.

This is similar [the previous experiment against HyFreeDP](25-us-vs-hyfreedp.md), but this time we will evaluate against all the datasets and epsilons.

## Objective

We will train the same models on the same datasets and record the resulting accuracy to compare against HyFreeDP.

## Methodology

As usual, we will fix the number of epochs at 40, run 20 trials of hyperparameter optimization (HPO), and evaluate the model on the test set.

## HPO configuration

```
batch_size:
  type: ordered
  min: 256
  # ordered categorical search!
  # options will be considered autoamtically:
  # 256, 512, 1024, ..., Full batch
learning_rate:
  min: 1e-4
  max: 0.05
  type: float
  log_space: True
max_grad_norm:
  type: ordered
  # ordered categorical search!
  options:
    - 1e-5
    - 1e-1
    - 1
    - 10
    - 25
    - 50
```

This configuration uses ordered categorical parameters to preserve ranking in discrete choices like gradient clipping and batch size.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

Although we are comparing against training all parameters, for the sake of training time, we will use FiLM parameterization.

## Datasets

We will conduct the experiment on the following datasets:

- **CIFAR-10** (100% subset)
- **CIFAR-100** (100% subset)
- **Food-101** (100% subset)
- **dpdl-benchmark/svhn_cropped** (100% subset)
- **dpdl-benchmark/gtsrb** (100% subset)

Additionally, we will also perform a run on the full SUN397 dataset, as this number is missing

- **dpdl-benchmark/sun397** (100% subset)

## Epsilon Values

We will run the experiment over ε = { 1, 3, 8 }.

