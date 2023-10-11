# Batch size variation

## Objective

We investigate the influence of varying batch sizes on the optimal configurations of _all_ the other hyperparameters (epochs, learning_rate, max_grad_norm) using Bayesian optimization.

## Methodology

- Batch size variation: We vary the batch size systematically through a predefined set of values.
  - Batch sizes are generated according to \{ 2^x | x = 8, 15 \}. We also include full batch.
- Bayesian Optimization: For each batch size, we use Bayesian optimization to find the good values of the other hyperparameters (epochs, learning_rate, max_grad_norm).

## Model: Vision transformer (vit_base_patch16_224.augreg_in21k)

Do these for all epsilon = \{0.25, 0.5, 1, 2, 4, 8\}

### Epsilon = 0.25

#### Dataset: CIFAR100

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

#### Dataset: CIFAR100 (10% subset)

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256        |                  |                         |                             |          |
| 512        |                  |                         |                             |          |
| 1024       |                  |                         |                             |          |
| 2048       |                  |                         |                             |          |
| 4096       |                  |                         |                             |          |
| Full batch |                  |                         |                             |          |

## Model: ResNet-50 (resnetv2_50x1_bitm_in21k)

Do these for all epsilon = \{0.25, 0.5, 1, 2, 4, 8\}

### Epsilon = 0.25

#### Dataset: CIFAR100

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

#### Dataset: CIFAR100 (10% subset)

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256        |                  |                         |                             |          |
| 512        |                  |                         |                             |          |
| 1024       |                  |                         |                             |          |
| 2048       |                  |                         |                             |          |
| 4096       |                  |                         |                             |          |
| Full batch |                  |                         |                             |          |

