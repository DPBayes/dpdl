# Experiment Design

## Questions for Antti

- Are the steps in batch sizes too dense?
    - Comments by Antti?
- Compare with adaptive clipping?
    - Forgot to discuss. Why not? Antti?
- How about comparing the results against subsampling ratio instead of batch size?
- Which epsilons to use? Current plan of epsilon = \{1,2,...,8\} maybe too much?

## Questions/ideas

- Use test set for calculating final accuracy
- Save the optuna study in experiment directory (if we want to try more trials)
- Marlon mentioned something about epsilon -grid

## Handled questions/ideas

- Fixed vs optimized epochs?
    - Optimize epochs.
- Other datasets?
    - Ditch CIFAR10 (too easy)
    - Use full CIFAR100 and 10% subset CIFAR100
    - Later, let's use also SVHN
- Other models?
    - ResNet-50 and ViT are good because other papers use them
- Other metrics to track/optimize?
    - Let's just use accuracy
- Search hypers in log space from some params?
    - Probably easier to just use a linear range.
- Sensitivity analysis of individual hypers?
    - Maybe later.
- Define baseline hypers?
    - If we do the sensitivity analysis.

## Future experiments

- How does dataset imbalance affect the hypers?

## Goal of the experiments

We want to analyze the privacy-utility tradeoff for the hyperparameters and the interaction of the hyperparameters.

From the results of the analysis, we aim to develop methods for predicting good hyperparameters for the model in response to alterations of the privacy budget.

## General method overview

For CIFAR100 we divide the given training set (50000 examples) into a training set (45000 examples) and a validation set (5000 examples). For the Bayesian optimization we use multiclass accuracy of the validation set as optimization objective.

We run 20 trials of Bayesian optimization using the following bounds for the other hypers

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

The accuracy for the last trial is calculated using the test set (10000 examples).

When training using a subset of the data the training and validation data is divided according to the proportions. For examples, for 10% of CIFAR100 that is 4500 training examples and 5000 validation examples.

## Batch size variation

### Objective

We investigate the influence of varying batch sizes on the optimal configurations of _all_ the other hyperparameters (epochs, learning_rate, max_grad_norm) using Bayesian optimization.

### Methodology

1. Batch size variation: We vary the batch size systematically through a predefined set of values.

2. Bayesian Optimization: For each batch size, we use Bayesian optimization to find the good values of the other hyperparameters (epochs, learning_rate, max_grad_norm).

#### Model: Vision transformer (vit_base_patch16_224.augreg_in21k)

Do these for all epsilon = \{1, 2, 3, 4, 5, 6, 7, 8\}

##### Epsilon = 1.0

##### Dataset: CIFAR100

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

##### Epsilon = 1.0

##### Dataset: CIFAR100

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

