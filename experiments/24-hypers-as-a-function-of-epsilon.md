# Hyperparameters as a function of ε

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

## Difficult datasets

We will also try experiment with hard datasets, as the results with SUN397 hints that more difficult datasets might require larger clipping bounds

- **dpdl-benchmark/imagenet-hard-medium - 100% subset** ![paper](https://openreview.net/pdf?id=76w7bsdViZf)

## Epsilon Values

We will run the experiment over ε = { 1, 2, 4, 8 }.

## Selected results

### Accuracy

| Dataset   |   Subset | Model     | PEFT   | Source   |       1.0 |      8.0 |
|-----------|----------|-----------|--------|----------|-----------|----------|
| cifar100  |      0.1 | ViT-Base  | FiLM   | grid     | 0.5328    | 0.8793   |
| cifar100  |      0.1 | ViT-Tiny  | FiLM   | grid     | 0.4781    | 0.7814   |
| cifar100  |      0.1 | ViT-Tiny  | none   | grid     | 0.3725    | 0.7437   |
| cifar100  |      0.1 | ViT-Tiny  | none   | hpo      | 0.354     | 0.736    |
| cifar100  |      1   | ViT-Base  | FiLM   | grid     | 0.9049    | 0.9305   |
| cifar100  |      1   | ViT-Base  | FiLM   | hpo      | 0.9048    | 0.9302   |
| cifar100  |      1   | ViT-Base  | none   | hpo      | 0.8766    | 0.9228   |
| cifar100  |      1   | ViT-Small | FiLM   | hpo      | 0.8734    | 0.9036   |
| cifar100  |      1   | ViT-Tiny  | FiLM   | grid     | 0.8182    | 0.8556   |
| sun397    |      0.1 | ViT-Tiny  | none   | hpo      | 0.0488509 | 0.239043 |
| sun397    |      1   | ViT-Base  | FiLM   | hpo      | 0.620273  | 0.752432 |
| sun397    |      1   | ViT-Base  | none   | hpo      | 0.458439  | 0.702761 |
| sun397    |      1   | ViT-Small | FiLM   | hpo      | 0.603152  | 0.726468 |
| sun397    |      1   | ViT-Tiny  | FiLM   | grid     | 0.54389   | 0.666428 |

### Hypers and accuracy

| Model     | Dataset   |   Subset | PEFT   |   Epsilon | Source   |   Accuracy |          LR |   BatchSize |   ClipBound |   Epochs |
|:----------|:----------|---------:|:-------|----------:|:---------|-----------:|------------:|------------:|------------:|---------:|
| ViT-Base  | cifar100  |      0.1 | FiLM   |         1 | grid     |  0.5328    | 0.00817561  |        1024 |       1     |       40 |
| ViT-Base  | cifar100  |      0.1 | FiLM   |         8 | grid     |  0.8793    | 0.00817561  |          -1 |       1e-05 |       40 |
| ViT-Base  | cifar100  |      1   | FiLM   |         1 | grid     |  0.9049    | 0.000517947 |         256 |       1e-05 |       40 |
| ViT-Base  | cifar100  |      1   | FiLM   |         1 | hpo      |  0.9048    | 0.00677475  |       45000 |       1e-05 |       40 |
| ViT-Base  | cifar100  |      1   | FiLM   |         8 | grid     |  0.9305    | 0.00372759  |       16834 |       1e-05 |       40 |
| ViT-Base  | cifar100  |      1   | FiLM   |         8 | hpo      |  0.9302    | 0.00355945  |       45000 |       1e-05 |       40 |
| ViT-Base  | cifar100  |      1   | none   |         1 | hpo      |  0.8766    | 0.000237439 |        4096 |      50     |       40 |
| ViT-Base  | cifar100  |      1   | none   |         8 | hpo      |  0.9228    | 0.000327233 |       32768 |      50     |       40 |
| ViT-Base  | sun397    |      1   | FiLM   |         1 | hpo      |  0.620273  | 0.000550406 |        2048 |      50     |       40 |
| ViT-Base  | sun397    |      1   | FiLM   |         8 | hpo      |  0.752432  | 0.00233387  |       76127 |      10     |       40 |
| ViT-Base  | sun397    |      1   | none   |         1 | hpo      |  0.458439  | 0.000498367 |       16384 |      50     |       40 |
| ViT-Base  | sun397    |      1   | none   |         8 | hpo      |  0.702761  | 0.000464013 |       32768 |      50     |       40 |
| ViT-Small | cifar100  |      1   | FiLM   |         1 | hpo      |  0.8734    | 0.00574391  |       45000 |       0.1   |       40 |
| ViT-Small | cifar100  |      1   | FiLM   |         8 | hpo      |  0.9036    | 0.0011812   |        8192 |      10     |       40 |
| ViT-Small | sun397    |      1   | FiLM   |         1 | hpo      |  0.603152  | 0.000939682 |        2048 |      10     |       40 |
| ViT-Small | sun397    |      1   | FiLM   |         8 | hpo      |  0.726468  | 0.00379717  |       76127 |      50     |       40 |
| ViT-Tiny  | cifar100  |      0.1 | FiLM   |         1 | grid     |  0.4781    | 0.00152894  |         512 |      10     |       40 |
| ViT-Tiny  | cifar100  |      0.1 | FiLM   |         8 | grid     |  0.7814    | 0.00267362  |        2048 |       0.1   |       40 |
| ViT-Tiny  | cifar100  |      0.1 | none   |         1 | grid     |  0.3725    | 0.000361064 |        2048 |       0.1   |       40 |
| ViT-Tiny  | cifar100  |      0.1 | none   |         1 | hpo      |  0.354     | 0.000118586 |         256 |       0.1   |       60 |
| ViT-Tiny  | cifar100  |      0.1 | none   |         8 | grid     |  0.7437    | 0.000553918 |        4096 |      50     |       40 |
| ViT-Tiny  | cifar100  |      0.1 | none   |         8 | hpo      |  0.736     | 0.000245582 |        4096 |      50     |       60 |
| ViT-Tiny  | cifar100  |      1   | FiLM   |         1 | grid     |  0.8182    | 0.0013895   |       16834 |      10     |       40 |
| ViT-Tiny  | cifar100  |      1   | FiLM   |         8 | grid     |  0.8556    | 0.0013895   |        4096 |       1     |       40 |
| ViT-Tiny  | sun397    |      0.1 | none   |         1 | hpo      |  0.0488509 | 0.000359914 |         512 |       1e-05 |       40 |
| ViT-Tiny  | sun397    |      0.1 | none   |         8 | hpo      |  0.239043  | 0.00154029  |        7428 |      50     |       60 |
| ViT-Tiny  | sun397    |      1   | FiLM   |         1 | grid     |  0.54389   | 0.00372759  |       32768 |      10     |       40 |
| ViT-Tiny  | sun397    |      1   | FiLM   |         8 | grid     |  0.666428  | 0.00372759  |       32768 |      10     |       40 |

