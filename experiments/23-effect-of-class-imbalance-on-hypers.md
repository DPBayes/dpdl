# Effect of class imbalance on the hyperparameters

## Motivation

Based on the results of the [privacy assistant real-world evaluation experiment](22-privacy-assistant-real-world-evaluation.md) it looks like the learning rate (and possibly batch size) would grow as a function of ε, but only in the presence of class imbalance. The image below shows the optimized learning rates as a function of epsilon.

![](images/23-imbalance-motivation.png)

This might be a crucial observation when planning future ideas.

Additionally, we will collect useful data for ε →  ε hyperparameter transfer paper.

## Objective

We will collect the accuracies achieved for various hyperparameter configurations arranged in a grid using different values of ε and imbalance factors.

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

## Epsilon Values

We will run the experiment over ε = { 1, 2, 4, 8 }.

## Imbalance Factors

We will run the experiment over imbalance factors of 1.0, 0.5, 0.25, 0.1.

Imbalance factor 1.0 corresponds to balanced dataset and a factor of 0.1 corresponds to majority class having 10 more examples as the minority class. The number of example in other classes follow the exponential distribution.
