# Learning Rate Variation

## Motivation

In differential private deep learning, understanding how different hyperparameters affect privacy and model performance is critical. Our goal is to investigate the relationship between key hyperparameters, particularly learning rate, batch size and maximum gradient norm, and their impact on the privacy-utility trade-off. We anticipate, this understanding to enable us to optimize models more effectively within the limits of privacy budgets.

Current methods for finding the optimal hyperparameter settings are often expensive and time-consuming. By systematically exploring and using Bayesian optimization, we aim to identify patterns and relationships that can lead to faster and more efficient model tuning. This work is not just about achieving better model accuracy within privacy constraints; it's about enhancing the entire process of model development in the realm of DP.

The insights from this study are expected to improve our strategies for hyperparameter selection in differential private deep learning, leading to models that are both more powerful and privacy-conscious.

## Objective

The goal of this experiment is to investigate the impact of varying learning rates on the optimized configuration of _all_ the other hyperparameters (epochs, batch_size, max_grad_norm) using Bayesian optimization. We will start with 10% of the CIFAR-10 and CIFAR-100 datasets and then use 100% of the data for epsilon=1.

## Methodology

- **Learning Rate Variation**: Systematically vary the learning rate through the following set values:
  - Learning rates: `0.000001`, `0.000003`, `0.000010`, `0.000032`, `0.000100`, `0.000316`, `0.001000`, `0.003162`, `0.010000`, `0.031623`, `0.100000` (`[10**x for x in np.round(np.arange(-6, -.5, .5), 2)`)
- **Bayesian Optimization**: Use Bayesian optimization to find good values of the other hyperparameters (epochs, batch_size, max_grad_norm) for each learning rate.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
- **ResNet-50 (resnetv2_50x1_bit.goog_in21k)**

## Datasets

- **CIFAR-10 (10% Subset)**: We utilize a 10% subset of CIFAR-10, focusing on initial insights and quicker iterations. The full dataset is not used as it presents less challenge and may not provide meaningful differentiation for hyperparameter tuning.
- **CIFAR-100 (10% Subset)**: We start with a 10% subset of CIFAR-100, enabling rapid preliminary analysis and quicker turnarounds in the initial phases of our experimentation.
- **CIFAR-100 and CIFAR-10 (Full Dataset)**: We extend our experimentation to the full datasets to better understand model performance. However, due to the considerable resources required, we initially limit these experiments to epsilon=1.0.

## Epsilon Values

Conduct experiments with epsilon values of `{0.25, 0.5, 1, 2, 4, 8}`. For 100% of CIFAR-10 and CIFAR-100, repeat the experiment only with epsilon=1.

## Experiment Setup

Record the following for each combination of model, dataset, learning rate, and epsilon value:

- Selected learning rate
- Optimized epochs
- Optimized batch size
- Optimized max gradient norm
- Accuracy

## Results

### Vision Transformer (vit_base_patch16_224.augreg_in21k)

#### CIFAR-10 (10% Subset)

| Learning Rate | Optimized batch size | Optimized epochs | Optimized max gradient norm | Accuracy |
|---------------|----------------------|------------------|-----------------------------|----------|
| 0.000001      |                      |                  |                             |          |
| 0.000003      |                      |                  |                             |          |
| 0.000010      |                      |                  |                             |          |
| 0.000032      |                      |                  |                             |          |
| 0.000100      |                      |                  |                             |          |
| 0.000316      |                      |                  |                             |          |
| 0.001000      |                      |                  |                             |          |
| 0.003162      |                      |                  |                             |          |
| 0.010000      |                      |                  |                             |          |
| 0.031623      |                      |                  |                             |          |
| 0.100000      |                      |                  |                             |          |

#### CIFAR-100 (10% Subset)

| Learning Rate | Optimized batch size | Optimized epochs | Optimized max gradient norm | Accuracy |
|---------------|----------------------|------------------|-----------------------------|----------|
| 0.000001      |                      |                  |                             |          |
| 0.000003      |                      |                  |                             |          |
| 0.000010      |                      |                  |                             |          |
| 0.000032      |                      |                  |                             |          |
| 0.000100      |                      |                  |                             |          |
| 0.000316      |                      |                  |                             |          |
| 0.001000      |                      |                  |                             |          |
| 0.003162      |                      |                  |                             |          |
| 0.010000      |                      |                  |                             |          |
| 0.031623      |                      |                  |                             |          |
| 0.100000      |                      |                  |                             |          |

#### CIFAR-100 (100% Subset)

Repeat experiment only for epsilon=1.

| Learning Rate | Optimized batch size | Optimized epochs | Optimized max gradient norm | Accuracy |
|---------------|----------------------|------------------|-----------------------------|----------|
| 0.000001      |                      |                  |                             |          |
| 0.000003      |                      |                  |                             |          |
| 0.000010      |                      |                  |                             |          |
| 0.000032      |                      |                  |                             |          |
| 0.000100      |                      |                  |                             |          |
| 0.000316      |                      |                  |                             |          |
| 0.001000      |                      |                  |                             |          |
| 0.003162      |                      |                  |                             |          |
| 0.010000      |                      |                  |                             |          |
| 0.031623      |                      |                  |                             |          |
| 0.100000      |                      |                  |                             |          |

### ResNet-50 (resnetv2_50x1_bit.goog_in21k)

#### CIFAR-10 (10% Subset)

| Learning Rate | Optimized batch size | Optimized epochs | Optimized max gradient norm | Accuracy |
|---------------|----------------------|------------------|-----------------------------|----------|
| 0.000001      |                      |                  |                             |          |
| 0.000003      |                      |                  |                             |          |
| 0.000010      |                      |                  |                             |          |
| 0.000032      |                      |                  |                             |          |
| 0.000100      |                      |                  |                             |          |
| 0.000316      |                      |                  |                             |          |
| 0.001000      |                      |                  |                             |          |
| 0.003162      |                      |                  |                             |          |
| 0.010000      |                      |                  |                             |          |
| 0.031623      |                      |                  |                             |          |
| 0.100000      |                      |                  |                             |          |

#### CIFAR-100 (10% Subset)

| Learning Rate | Optimized batch size | Optimized epochs | Optimized max gradient norm | Accuracy |
|---------------|----------------------|------------------|-----------------------------|----------|
| 0.000001      |                      |                  |                             |          |
| 0.000003      |                      |                  |                             |          |
| 0.000010      |                      |                  |                             |          |
| 0.000032      |                      |                  |                             |          |
| 0.000100      |                      |                  |                             |          |
| 0.000316      |                      |                  |                             |          |
| 0.001000      |                      |                  |                             |          |
| 0.003162      |                      |                  |                             |          |
| 0.010000      |                      |                  |                             |          |
| 0.031623      |                      |                  |                             |          |
| 0.100000      |                      |                  |                             |          |

