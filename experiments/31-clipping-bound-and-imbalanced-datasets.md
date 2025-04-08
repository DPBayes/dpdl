# Clipping bound and imbalanced datasets

## Motivation

We have seen that the clipping bound has an effect on some datasets with multiple classes and large imbalance, e.g. the SUN397 and the artifical ImageNet397. However, it is not clear what makes the clipping bound matter.

We have previously run an undocumented experiment with imbalanced classes using HPO, but unfortunately the results have been lost. Let's now re-run this experiment using the grid that has worked well for other experiments.

## Objective

We will define grids of hyperparameters and run training over this grid for multiple seeds and datasets. We will collect the accuracy evaluated on the test set.

## Methodology

We will fix the epochs at 40 and train the models on a grid of hyperparameters

- Batch size: 192, 512, 1024, 2048, 4096, Full batch
- Clipping bound: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10
- Learning rate: 0.00050, 0.00087, 0.00153, 0.00267, 0.00468, 0.00818, 0.01430, 0.02500 (`np.geomspace(5e-4, 0.025, 8)`)

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

We will use the FiLM parameterization.

## Datasets

We will run the experiment with

- **cifar100 - 10% subset**
- **dpdl-benchmark/cifar10_10pct_plus_cifar100_humans - 100% subset**

## Epsilon Values

We will run the experiment over ε = { 1, 8 }.

## Imbalance Factors

We will run the experiment over imbalance factors of 1.0, 0.5, 0.25, 0.1.

Imbalance factor 1.0 corresponds to balanced dataset and a factor of 0.1 corresponds to majority class having 10 more examples as the minority class. The number of example in other classes follow the exponential distribution.
