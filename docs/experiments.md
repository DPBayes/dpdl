# Experiment Design

## Questions/ideas

- Are the steps in batch sizes too dense?
- Compare with adaptive clipping?
- Fixed vs optimized epochs?
- Sensitivity analysis of individual hypers?
- Define baseline hypers?
- Other datasets?
- Other models?
- Other metrics to track/optimize?
- Search hypers in log space from some params?

## Future experiments

- How does dataset imbalance affect the hypers?

## Batch size variation

### Objective

To investigate the influence of varying batch sizes on the optimal configurations of other hyperparameters (epochs, learning_rate, max_grad_norm) using Bayesian optimization.

#### Methodology

1. Batch size variation: Systematically vary the batch size through a predefined set of values: 256, 512, 1024, 2048, 4096, 8192, 12288, etc.

2. Bayesian Optimization: For each batch size, use Bayesian optimization to find the optimal values of the other hyperparameters (epochs, learning_rate, max_grad_norm).

#### Experimentation

Conduct separate optimization runs for each batch size. In each run, the batch size is fixed, and Bayesian optimization is used to optimize the other hyperparameters. Here is a markdown table template that you might use to document the results:

### Configuration for Bayes optimization

```
learning_rate:
  max: 1.0e-2
  min: 1.0e-07
  type: float
epochs:
  max: 200
  min: 1
  type: int
max_grad_norm:
  max: 10
  min: 0.2
  type: float
```

#### Model: Vision transformer (vit_base_patch16_224.augreg_in21k)

Do these for all epsilon = \{1, 2, 3, 4, 5, 6, 7, 8\}

###### Epsilon = 1.0

##### Dataset: CIFAR10

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256   | | | | |
| 512   | | | | |
| 1024  | | | | |
| 2048  | | | | |
| 4096  | | | | |
| 6144  | | | | |
| 8192  | | | | |
| 10240 | | | | |
| 12288 | | | | |
| 14336 | | | | |
| 16384 | | | | |
| 18432 | | | | |
| 20480 | | | | |
| 22528 | | | | |
| 24576 | | | | |
| 26624 | | | | |
| 28672 | | | | |
| 30720 | | | | |
| 32768 | | | | |
| 34816 | | | | |
| 36864 | | | | |
| 38912 | | | | |
| 45000 | | | | |

##### Dataset: CIFAR100 (10% subset)

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256   | | | | |
| 512   | | | | |
| 1024  | | | | |
| 1536  | | | | |
| 2048  | | | | |
| 2560  | | | | |
| 3072  | | | | |
| 3584  | | | | |
| 4096  | | | | |
| 4500  | | | | |

#### Model: ResNet-50 (resnetv2_50x1_bitm_in21k)

Do these for all epsilon = \{1, 2, 3, 4, 5, 6, 7, 8\}

###### Epsilon = 1.0

##### Dataset: CIFAR10

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256   | | | | |
| 512   | | | | |
| 1024  | | | | |
| 2048  | | | | |
| 4096  | | | | |
| 6144  | | | | |
| 8192  | | | | |
| 10240 | | | | |
| 12288 | | | | |
| 14336 | | | | |
| 16384 | | | | |
| 18432 | | | | |
| 20480 | | | | |
| 22528 | | | | |
| 24576 | | | | |
| 26624 | | | | |
| 28672 | | | | |
| 30720 | | | | |
| 32768 | | | | |
| 34816 | | | | |
| 36864 | | | | |
| 38912 | | | | |
| 45000 | | | | |

##### Dataset: CIFAR100 (10% subset)

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256   | | | | |
| 512   | | | | |
| 1024  | | | | |
| 1536  | | | | |
| 2048  | | | | |
| 2560  | | | | |
| 3072  | | | | |
| 3584  | | | | |
| 4096  | | | | |
| 4500  | | | | |

