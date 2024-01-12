# Epoch Variation Experiment

## Motivation

In differential private deep learning, understanding how different hyperparameters affect privacy and model performance is critical. Our goal is to investigate the relationship between key hyperparameters, particularly the number of epochs, batch size, and maximum gradient norm, and their impact on the privacy-utility trade-off. We anticipate, this understanding to enable us to optimize models more effectively within the limits of privacy budgets.

Current methods for finding the optimal hyperparameter settings are often expensive and time-consuming. By systematically exploring and using Bayesian optimization, we aim to identify patterns and relationships that can lead to faster and more efficient model tuning. This work is not just about achieving better model accuracy within privacy constraints; it's about enhancing the entire process of model development in the realm of DP.

The insights from this study are expected to improve our strategies for hyperparameter selection in differential private deep learning, leading to models that are both more powerful and privacy-conscious.

## Objective

The goal of this experiment is to investigate the impact of varying the number of epochs on the optimized configuration of _all_ the other hyperparameters (learning rate, batch_size, max_grad_norm) using Bayesian optimization. We will start with 10% of the CIFAR-10 and CIFAR-100 datasets and then use 100% of the data for epsilon=1.

## Methodology

- **Epoch Variation**: Systematically vary the number of epochs through the following set values:
  - Epoch values: `2, 3, 6, 11, 20, 35, 63, 112, 200` (`[ int(x) for x in np.geomspace(2, 200, 9) ]`)
- **Bayesian Optimization**: Use Bayesian optimization to find good values of the other hyperparameters (learning rate, batch_size, max_grad_norm) for each epoch value.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
- **ResNet-50 (resnetv2_50x1_bit.goog_in21k)**

## Datasets

- **CIFAR-10 (10% Subset)**
- **CIFAR-100 (10% Subset)**
- **CIFAR-100 and CIFAR-10 (Full Dataset)**: Limited to epsilon=1.0.

## Epsilon Values

Conduct experiments with epsilon values of `{0.25, 0.5, 1, 2, 4, 8}`. For 100% of CIFAR-10 and CIFAR-100, repeat the experiment only with epsilon=1.

## Experiment Setup

Record the following for each combination of model, dataset, epoch value, and epsilon value:

- Selected number of epochs
- Optimized learning rate
- Optimized batch size
- Optimized max gradient norm
- Accuracy

## Results

The results section will be updated following the completion of the experiments, detailing the performance metrics across different epoch values for each model and dataset combination.

### Vision Transformer (vit_base_patch16_224.augreg_in21k)

#### CIFAR-10 (10% Subset)

| Epochs | Optimized learning rate | Optimized batch size | Optimized max gradient norm | Accuracy |
|--------|-------------------------|----------------------|-----------------------------|----------|
| 2      |                         |                      |                             |          |
| 3      |                         |                      |                             |          |
| 6      |                         |                      |                             |          |
| 11     |                         |                      |                             |          |
| 20     |                         |                      |                             |          |
| 35     |                         |                      |                             |          |
| 63     |                         |                      |                             |          |
| 112    |                         |                      |                             |          |
| 200    |                         |                      |                             |          |

#### CIFAR-100 (10% Subset)

| Epochs | Optimized learning rate | Optimized batch size | Optimized max gradient norm | Accuracy |
|--------|-------------------------|----------------------|-----------------------------|----------|
| 2      |                         |                      |                             |          |
| 3      |                         |                      |                             |          |
| 6      |                         |                      |                             |          |
| 11     |                         |                      |                             |          |
| 20     |                         |                      |                             |          |
| 35     |                         |                      |                             |          |
| 63     |                         |                      |                             |          |
| 112    |                         |                      |                             |          |
| 200    |                         |                      |                             |          |

#### CIFAR-100 (100% Subset)

Repeat experiment only for epsilon=1.

| Epochs | Optimized learning rate | Optimized batch size | Optimized max gradient norm | Accuracy |
|--------|-------------------------|----------------------|-----------------------------|----------|
| 2      |                         |                      |                             |          |
| 3      |                         |                      |                             |          |
| 6      |                         |                      |                             |          |
| 11     |                         |                      |                             |          |
| 20     |                         |                      |                             |          |
| 35     |                         |                      |                             |          |
| 63     |                         |                      |                             |          |
| 112    |                         |                      |                             |          |
| 200    |                         |                      |                             |          |

### ResNet-50 (resnetv2_50x1_bit.goog_in21k)

#### CIFAR-10 (10% Subset)

| Epochs | Optimized learning rate | Optimized batch size | Optimized max gradient norm | Accuracy |
|--------|-------------------------|----------------------|-----------------------------|----------|
| 2      |                         |                      |                             |          |
| 3      |                         |                      |                             |          |
| 6      |                         |                      |                             |          |
| 11     |                         |                      |                             |          |
| 20     |                         |                      |                             |          |
| 35     |                         |                      |                             |          |
| 63     |                         |                      |                             |          |
| 112    |                         |                      |                             |          |
| 200    |                         |                      |                             |          |

#### CIFAR-100 (10% Subset)

| Epochs | Optimized learning rate | Optimized batch size | Optimized max gradient norm | Accuracy |
|--------|-------------------------|----------------------|-----------------------------|----------|
| 2      |                         |                      |                             |          |
| 3      |                         |                      |                             |          |
| 6      |                         |                      |                             |          |
| 11     |                         |                      |                             |          |
| 20     |                         |                      |                             |          |
| 35     |                         |                      |                             |          |
| 63     |                         |                      |                             |          |
| 112    |                         |                      |                             |          |
| 200    |                         |                      |                             |          |

