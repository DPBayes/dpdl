# Full proper comparison: Us vs HyperFreeDP

## Motivation

In the paper [Towards Hyperparameter Free Optimization with Differential Privacy](https://openreview.net/pdf?id=2kGKsyhtvh), the authors claim state-of-the-art results.

We will use this as a baseline and need a comparison.

## Objective

We will evaluate the following algorithm for determining the hyperparameters

![Algorithm for hyperparameters](images/35-hyperparameter-flow.jpg)

We will train the same models on the same datasets and record the resulting accuracy to compare against HyFreeDP.

## Methodology

We will determine the hyperparameters following the algorithm above.

For the non-DP run, we will use the following hyperparameters
- Epochs: 10
- Learning rate: 1e-4
- Batch size: 256

### Real difficulty (NonDP-GS results from the HyFreeDP paper)

| Model     | CIFAR10 | CIFAR100 | SVHN  | GTSRB | Food101 | SUN397 |
|-----------|---------|----------|-------|-------|---------|--------|
| ViT-Small | 98.34   | 89.21    | 95.42 | 98.64 | 85.32   | N/A    |
| ViT-Base  | 98.55   | 92.37    | 96.75 | 98.39 | 89.53   | N/A    |

### Inferred difficulty metrics

| dataset                     | model                              |   S_E |   S_R |   bar_S |   bar_S_normalized |   test_mean |   train_mean |    gap |   cv_trimmed |   final_slope |   base_difficulty |   difficulty |
|:----------------------------|:-----------------------------------|------:|------:|--------:|-------------------:|------------:|-------------:|-------:|-------------:|--------------:|------------------:|-------------:|
| dpdl-benchmark/sun397       | vit_small_patch16_224.augreg_in21k | 0.115 | 0.564 |   0.612 |              0.002 |       0.747 |        0.806 |  0.059 |        0.195 |             0 |             0.253 |        0.38  |
| dpdl-benchmark/sun397       | vit_base_patch16_224.augreg_in21k  | 0.124 | 0.499 |   0.594 |              0.001 |       0.778 |        0.912 |  0.135 |        0.173 |            -0 |             0.222 |        0.376 |
| dpdl-benchmark/gtsrb        | vit_base_patch16_224.augreg_in21k  | 0.277 | 0.571 |   0.574 |              0.013 |       0.839 |        0.993 |  0.154 |        0.17  |             0 |             0.161 |        0.322 |
| dpdl-benchmark/gtsrb        | vit_small_patch16_224.augreg_in21k | 0.21  | 0.653 |   0.611 |              0.014 |       0.87  |        0.976 |  0.106 |        0.11  |            -0 |             0.13  |        0.238 |
| cifar100                    | vit_small_patch16_224.augreg_in21k | 0.195 | 0.609 |   0.603 |              0.006 |       0.904 |        0.927 |  0.023 |        0.058 |             0 |             0.096 |        0.136 |
| food101                     | vit_small_patch16_224.augreg_in21k | 0.166 | 0.461 |   0.574 |              0.006 |       0.888 |        0.87  | -0.017 |        0.06  |             0 |             0.112 |        0.134 |
| food101                     | vit_base_patch16_224.augreg_in21k  | 0.191 | 0.45  |   0.565 |              0.006 |       0.916 |        0.927 |  0.01  |        0.048 |            -0 |             0.084 |        0.113 |
| cifar100                    | vit_base_patch16_224.augreg_in21k  | 0.257 | 0.588 |   0.583 |              0.006 |       0.933 |        0.968 |  0.035 |        0.046 |             0 |             0.067 |        0.108 |
| dpdl-benchmark/svhn_cropped | vit_small_patch16_224.augreg_in21k | 0.288 | 0.506 |   0.554 |              0.055 |       0.92  |        0.907 | -0.013 |        0.039 |             0 |             0.08  |        0.092 |
| dpdl-benchmark/svhn_cropped | vit_base_patch16_224.augreg_in21k  | 0.317 | 0.48  |   0.541 |              0.054 |       0.938 |        0.928 | -0.01  |        0.029 |            -0 |             0.062 |        0.072 |
| cifar10                     | vit_small_patch16_224.augreg_in21k | 0.305 | 0.668 |   0.591 |              0.059 |       0.985 |        0.99  |  0.005 |        0.011 |             0 |             0.015 |        0.024 |
| cifar10                     | vit_base_patch16_224.augreg_in21k  | 0.346 | 0.63  |   0.571 |              0.057 |       0.989 |        0.996 |  0.007 |        0.01  |             0 |             0.011 |        0.019 |

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
- **Vision Transformer (vit_small_patch16_224.augreg_in21k)**

Although we are comparing against training all parameters, for the sake of training time, we will use FiLM parameterization.

## Datasets

We will conduct the experiment on the following datasets:

- **CIFAR-10** (100% subset)
- **CIFAR-100** (100% subset)
- **Food-101** (100% subset)
- **dpdl-benchmark/svhn_cropped** (100% subset)
- **dpdl-benchmark/gtsrb** (100% subset)

Additionally, we will also perform a run on the full SUN397 dataset, as this has been the most interesting dataset

- **dpdl-benchmark/sun397** (100% subset)

## Epsilon Values

We will run the experiment over ε = { 1, 3, 8 }.

