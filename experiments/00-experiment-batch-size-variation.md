# Batch size variation

## Objective

We investigate the influence of varying batch sizes on the optimal configurations of _all_ the other hyperparameters (epochs, learning_rate, max_grad_norm) using Bayesian optimization, starting with 10% of the data and then using 100% of the data for epsilon=1.

## Methodology

- **Batch Size Variation**: Systematically vary the batch size through a predefined set of values:
  - Batch sizes: \{ 2^x | x = 8, ..., 15 \} and Full batch.
- **Bayesian Optimization**: Use Bayesian optimization to find good values of the other hyperparameters (epochs, learning_rate, max_grad_norm) for each batch size.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
- **ResNet-50 (resnetv2_50x1_bitm_in21k)**

## Datasets

- **CIFAR-100**: Evaluate using both 10% and 100% of the dataset.
- **CIFAR-10 (10% Subset)**: Initially, run experiments with 10% of CIFAR-10.
- **CIFAR-100 (10% Subset)**: Initially, run experiments with 10% of CIFAR-100.

## Epsilon Values

Conduct experiments with epsilon values of \{0.25, 0.5, 1, 2, 4, 8\}. For 100% of CIFAR-100, repeat the experiment only with epsilon=1.

## Experiment Setup

For each combination of model, dataset, batch size, and epsilon value, record:

- Optimized batch size
- Optimized epochs
- Optimized learning rate
- Optimized max gradient norm
- Accuracy

## Results

### Vision Transformer (vit_base_patch16_224.augreg_in21k)

#### CIFAR-10 (10% Subset)

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256        |                  |                         |                             |          |
| 512        |                  |                         |                             |          |
| 1024       |                  |                         |                             |          |
| 2048       |                  |                         |                             |          |
| 4096       |                  |                         |                             |          |
| Full batch |                  |                         |                             |          |

#### CIFAR-100 (10% Subset)

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256        |                  |                         |                             |          |
| 512        |                  |                         |                             |          |
| 1024       |                  |                         |                             |          |
| 2048       |                  |                         |                             |          |
| 4096       |                  |                         |                             |          |
| Full batch |                  |                         |                             |          |

#### CIFAR-100 (100% Subset)

Repeat experiment only for epsilon=1.

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256        |                  |                         |                             |          |
| 512        |                  |                         |                             |          |
| 1024       |                  |                         |                             |          |
| 2048       |                  |                         |                             |          |
| 4096       |                  |                         |                             |          |
| 8192       |                  |                         |                             |          |
| 16384      |                  |                         |                             |          |
| 32768      |                  |                         |                             |          |
| Full batch |                  |                         |                             |          |

### ResNet-50 (resnetv2_50x1_bitm_in21k)

#### CIFAR-10 (10% Subset)

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256        |                  |                         |                             |          |
| 512        |                  |                         |                             |          |
| 1024       |                  |                         |                             |          |
| 2048       |                  |                         |                             |          |
| 4096       |                  |                         |                             |          |
| Full batch |                  |                         |                             |          |

#### CIFAR-100 (10% Subset)

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256        |                  |                         |                             |          |
| 512        |                  |                         |                             |          |
| 1024       |                  |                         |                             |          |
| 2048       |                  |                         |                             |          |
| 4096       |                  |                         |                             |          |
| Full batch |                  |                         |                             |          |

#### CIFAR-100 (100% Subset)

Repeat experiment only for epsilon=1.

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256        |                  |                         |                             |          |
| 512        |                  |                         |                             |          |
| 1024       |                  |                         |                             |          |
| 2048       |                  |                         |                             |          |
| 4096       |                  |                         |                             |          |
| 8192       |                  |                         |                             |          |
| 16384      |                  |                         |                             |          |
| 32768      |                  |                         |                             |          |
| Full batch |                  |                         |                             |          |

