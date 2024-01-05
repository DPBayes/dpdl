# Maximum Gradient Norm Variation

## Motivation

In differential private deep learning, understanding how different hyperparameters affect privacy and model performance is critical. Our goal is to investigate the relationship between key hyperparameters, particularly learning rate, batch size and maximum gradient norm, and their impact on the privacy-utility trade-off. We anticipate, this understanding to enable us to optimize models more effectively within the limits of privacy budgets.

Current methods for finding the optimal hyperparameter settings are often expensive and time-consuming. By systematically exploring and using Bayesian optimization, we aim to identify patterns and relationships that can lead to faster and more efficient model tuning. This work is not just about achieving better model accuracy within privacy constraints; it's about enhancing the entire process of model development in the realm of DP.

The insights from this study are expected to improve our strategies for hyperparameter selection in differential private deep learning, leading to models that are both more powerful and privacy-conscious.

## Objective

The goal is to investigate the impact of varying maximum gradient norms on the optimized configuration of _all_ the other hyperparameters (epochs, learning_rate, batch_size) using Bayesian optimization. The study starts with 10% of the data and then uses 100% of the data for epsilon=1.

## Methodology

- **Maximum Gradient Norm Variation**: Systematically vary the maximum gradient norm through the following set values:
  - Max gradient norms: {0.1, 0.18, 0.32, 0.56, 1.0, 1.78, 3.16, 5.62, 10}.
- **Bayesian Optimization**: We use Bayesian optimization to find good values of the other hyperparameters (epochs, learning_rate, batch_size) for each maximum gradient norm.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
- **ResNet-50 (resnetv2_50x1_bitm_in21k)**

## Datasets

- **CIFAR-10 (10% Subset)**: We utilize a 10% subset of CIFAR-10, focusing on initial insights and quicker iterations. The full dataset is not used as it presents less challenge and may not provide meaningful differentiation for hyperparameter tuning.
- **CIFAR-100 (10% Subset)**: We start with a 10% subset of CIFAR-100, enabling rapid preliminary analysis and quicker turnarounds in the initial phases of our experimentation.
- **CIFAR-100 (Full Dataset)**: We extend our experimentation to the full CIFAR-100 dataset to thoroughly understand model performance in more complex scenarios. However, due to the considerable resources required, we initially limit these experiments to epsilon=1.0.

## Epsilon Values

Conduct experiments with epsilon values of {0.25, 0.5, 1, 2, 4, 8}. For 100% of CIFAR-100, repeat the experiment only with epsilon=1.

## Experiment Setup

For each combination of model, dataset, maximum gradient norm, and epsilon value, record:

- Maximum gradient norm used
- Optimized batch size
- Optimized epochs
- Optimized learning rate
- Accuracy

### Vision Transformer (vit_base_patch16_224.augreg_in21k)

#### CIFAR-10 (10% Subset)

| Max Gradient Norm | Optimized batch size | Optimized epochs | Optimized learning rate | Accuracy |
|-------------------|----------------------|------------------|-------------------------|----------|
| 0.1               |                      |                  |                         |          |
| 0.18              |                      |                  |                         |          |
| 0.32              |                      |                  |                         |          |
| 0.56              |                      |                  |                         |          |
| 1.0               |                      |                  |                         |          |
| 1.78              |                      |                  |                         |          |
| 3.16              |                      |                  |                         |          |
| 5.62              |                      |                  |                         |          |
| 10                |                      |                  |                         |          |

#### CIFAR-100 (10% Subset)

| Max Gradient Norm | Optimized batch size | Optimized epochs | Optimized learning rate | Accuracy |
|-------------------|----------------------|------------------|-------------------------|----------|
| 0.1               |                      |                  |                         |          |
| 0.18              |                      |                  |                         |          |
| 0.32              |                      |                  |                         |          |
| 0.56              |                      |                  |                         |          |
| 1.0               |                      |                  |                         |          |
| 1.78              |                      |                  |                         |          |
| 3.16              |                      |                  |                         |          |
| 5.62              |                      |                  |                         |          |
| 10                |                      |                  |                         |          |

#### CIFAR-100 (100% Subset)

Repeat experiment only for epsilon=1.

| Max Gradient Norm | Optimized batch size | Optimized epochs | Optimized learning rate | Accuracy |
|-------------------|----------------------|------------------|-------------------------|----------|
| 0.1               |                      |                  |                         |          |
| 0.18              |                      |                  |                         |          |
| 0.32              |                      |                  |                         |          |
| 0.56              |                      |                  |                         |          |
| 1.0               |                      |                  |                         |          |
| 1.78              |                      |                  |                         |          |
| 3.16              |                      |                  |                         |          |
| 5.62              |                      |                  |                         |          |
| 10                |                      |                  |                         |          |

### ResNet-50 (resnetv2_50x1_bitm_in21k)

#### CIFAR-10 (10% Subset)

| Max Gradient Norm | Optimized batch size | Optimized epochs | Optimized learning rate | Accuracy |
|-------------------|----------------------|------------------|-------------------------|----------|
| 0.1               |                      |                  |                         |          |
| 0.18              |                      |                  |                         |          |
| 0.32              |                      |                  |                         |          |
| 0.56              |                      |                  |                         |          |
| 1.0               |                      |                  |                         |          |
| 1.78              |                      |                  |                         |          |
| 3.16              |                      |                  |                         |          |
| 5.62              |                      |                  |                         |          |
| 10                |                      |                  |                         |          |

#### CIFAR-100 (10% Subset)

| Max Gradient Norm | Optimized batch size | Optimized epochs | Optimized learning rate | Accuracy |
|-------------------|----------------------|------------------|-------------------------|----------|
| 0.1               |                      |                  |                         |          |
| 0.18              |                      |                  |                         |          |
| 0.32              |                      |                  |                         |          |
| 0.56              |                      |                  |                         |          |
| 1.0               |                      |                  |                         |          |
| 1.78              |                      |                  |                         |          |
| 3.16              |                      |                  |                         |          |
| 5.62              |                      |                  |                         |          |
| 10                |                      |                  |                         |          |

#### CIFAR-100 (100% Subset)

Repeat experiment only for epsilon=1.

| Max Gradient Norm | Optimized batch size | Optimized epochs | Optimized learning rate | Accuracy |
|-------------------|----------------------|------------------|-------------------------|----------|
| 0.1               |                      |                  |                         |          |
| 0.18              |                      |                  |                         |          |
| 0.32              |                      |                  |                         |          |
| 0.56              |                      |                  |                         |          |
| 1.0               |                      |                  |                         |          |
| 1.78              |                      |                  |                         |          |
| 3.16              |                      |                  |                         |          |
| 5.62              |                      |                  |                         |          |
| 10                |                      |                  |                         |          |

