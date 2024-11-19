# Hyperparameter grid

## Motivation

We still have not been able to understand especially the clipping bound hyperparameter when fine-tuning deep learning models. Ih the previous experiments that we have ran, the effect of the clipping bound has been minimimal to non-existant. Futhermore, the hyperparameter optimization could be unstable enough the not be able to detect the effect of the differences in the less important hyperparameters, such as the clipping bound. Lastly, we have previously ran over 1D grids of the hyperparameters, but we never looked in to the joint effect of the hyperparamters.

## Objective

We want design a grid over the hyperparameters that we will train on. The initial objecticve is to record the accuracy of the different hyperparameter combinations and inspect what kind of effects the different hyperparameters have on the resulting accuracy.

We are also interested in studying if different hyperparameter pairs (e.g. the clipping bound and the batch size) have a joint effect on the resulting accuracy.

## Methodology

We will fix the epochs at 40 and train the model using combinations drawn from the following grids:

- We will construct a logarithmic grid for the learning rates in the range [1e-4, 0.05].
- For the batch sizes we will use a logarithmic grid [256, 512, 1024, 2048, 4096, -1], where -1 denotes full batch.
- For the clipping bound we will use the grid [1e-2, 0.1, 1, 3.5, 12.25, 42.96, 150.0] (`[1e-2, 0.1] + list(np.geomspace(1, 150, 5))`)

We will train the model over the grid for 3 different epsilon values:

- ε = {1, 4, 8}

## Models

We will conduct the experiment using a single model and we will train FiLM parameters:

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

## Datasets

We will use the same datasets as in the [previous experiment on HPO alternatives](18-hpo-alternatives.md):

- **datasets/dpdl-benchmark/cifar10_10pct_plus_cifar100_humans - 100% subset**
- **datasets/cifar100 - 10% subset**
- **datasets/dpdl-benchmark/svhn_cropped_balanced - 10% subset**

The motivation is that we will have some datasets that we saw in the previous experiment to be stable, so if we compare the results to the ones achieved with HPO, we don't have to wonder if the the results disagree because of the unstability of the HPO. The one more demanding dataset is added for obvious reasons.
