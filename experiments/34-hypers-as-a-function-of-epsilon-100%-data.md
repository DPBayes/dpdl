# Hyperparameters as a function of ε with full trainset (100%)

## Motivation

For a fair comparison with HyperFree paper, it is required to run experiments with full dataset, instead of 10%, which is used in our current setting.

## Objective

We will define grids of hyperparameters and run training over this grid for multiple seeds and datasets. We will collect the accuracy evaluated on the test set.

## Methodology

We will fix the epochs at 40 and train the models on a grid of hyperparameters.

```
- Batch size(4): 128, 1024, 8192, FULL(50k)
- Clipping bound(5): 1e-5, 1e-4, 1, 10, 50
- Learning rate(4): 1e-5, 1e-4, 1e-3, 1e-2
```

For 3 epsilons (1,3,8), this will take `6x5x4x3 = 360 runs`, and will take about `270 x 1.5 = 540 hr` if not consider run experiments parallel.

Alternatively, we can build a finer grid, like
```
- Batch size: 128, 512, 2048, 8192, 32768, FULL(50k)
- Clipping bound: 1e-5, 1e-4, 1, 10, 50
- Learning rate: [1.00000000e-05, 2.68269580e-05, 7.19685673e-05, 1.93069773e-04, 5.17947468e-04, 1.38949549e-03, 3.72759372e-03, 1.00000000e-02] (np.geomspace(1e-5,1e-2,8))
```

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

HyperFree use both ViT-base and ViT-small. To reduce the compute cost, and keep consistancy with the rest of paper, which use ViT-Tiny, we may only use ViT-base.

## Datasets

We will run the experiment with

- **cifar100 - 100% subset**

## Epsilon Values

We will run the experiment over ε = { 1, 3, 8 }.

