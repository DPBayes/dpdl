# Effect of class imbalance on the hyperparameters

## Motivation

We are planning to revive the idea of transferring the hyperparameters ε →  ε.

The idea is that we will optimize the hyperparameters on one ε and then have means of transferring those optimized hyperparameters to another ε without repeating the hyperparameter optimization.

For this, we need data to see how the hypers change as a function of epsilon.

## Objective

We will define grids of hyperparameters and run training over this grid for multiple seeds and datasets. We will collect the accuracy evaluated on the test set.

## Methodology

We will fix the epochs at 40 and train the models on a grid of hyperparameters

- Batch size: 192, 512, 1024, 2048, 4096, Full batch
- Clipping bound: 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10
- Learning rate: 0.00050, 0.00087, 0.00153, 0.00267, 0.00468, 0.00818, 0.01430, 0.02500 (`np.geomspace(5e-4, 0.025, 8)`)

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

## Datasets

We will run the experiment with

- **cifar100 - 10% subset**
- **dpdl-benchmark/sun397 - 10% subset**
- **dpdl-benchmark/patch_camelyon - 2% subset**
- **dpdl-benchmark/cassava - 100% subset**
- **dpdl-benchmark/svhn_cropped - 10% subset**
- **dpdl-benchmark/svhn_cropped_balanced - 10% subset**

## Epsilon Values

We will run the experiment over ε = { 1, 2, 4, 8 }.

