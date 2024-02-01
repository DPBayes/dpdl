# Batch size variation

## Motivation

In differential private deep learning, understanding how different hyperparameters affect privacy and model performance is critical. Our goal is to investigate the relationship between key hyperparameters, particularly learning rate, batch size and maximum gradient norm, and their impact on the privacy-utility trade-off. We anticipate, this understanding to enable us to optimize models more effectively within the limits of privacy budgets.

Current methods for finding the optimal hyperparameter settings are often expensive and time-consuming. By systematically exploring and using Bayesian optimization, we aim to identify patterns and relationships that can lead to faster and more efficient model tuning. This work is not just about achieving better model accuracy within privacy constraints; it's about enhancing the entire process of model development in the realm of DP.

The insights from this study are expected to improve our strategies for hyperparameter selection in differential private deep learning, leading to models that are both more powerful and privacy-conscious.

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

- **CIFAR-10 (10% Subset)**: We utilize a 10% subset of CIFAR-10, focusing on initial insights and quicker iterations. The full dataset is not used as it presents less challenge and may not provide meaningful differentiation for hyperparameter tuning.
- **CIFAR-100 (10% Subset)**: We start with a 10% subset of CIFAR-100, enabling rapid preliminary analysis and quicker turnarounds in the initial phases of our experimentation.
- **CIFAR-100 and CIFAR-10 (Full Dataset)**: We extend our experimentation to the full CIFAR-100 dataset to thoroughly understand model performance in more complex scenarios. However, due to the considerable resources required, we initially limit these experiments to epsilon=1.0.

## Epsilon Values

We conduct experiments with epsilon values of \{0.25, 0.5, 1, 2, 4, 8\}. For 100% of CIFAR-10 and CIFAR-100, we repeat the experiment only with epsilon=1.

## Experiment Setup

For each combination of model, dataset, batch size, and epsilon value, record:

- The batch size used
- Optimized epochs
- Optimized learning rate
- Optimized max gradient norm
- Accuracy

## Results

### resnetv2_50x1_bit.goog_in21k on cifar10 (10% Subset) - Epsilon 0.25

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 95 | 0.000108 | 3.99 | 0.86 |
| 512 | 143 | 0.000322 | 5.57 | 0.85 |
| 1024 | 162 | 0.000295 | 1.50 | 0.85 |
| 2048 | 103 | 0.000358 | 0.66 | 0.83 |
| 4096 | 103 | 0.000358 | 0.66 | 0.85 |
| Full batch | 103 | 0.000358 | 0.66 | 0.83 |

### resnetv2_50x1_bit.goog_in21k on cifar10 (10% Subset) - Epsilon 0.5

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 103 | 0.000358 | 0.66 | 0.90 |
| 512 | 103 | 0.000358 | 0.66 | 0.90 |
| 1024 | 103 | 0.000358 | 0.66 | 0.88 |
| 2048 | 53 | 0.000000 | 100 | 0.89 |
| 4096 | 87 | 0.000006 | 6.20 | 0.89 |
| Full batch | 67 | 0.002566 | 0.20 | 0.91 |

### resnetv2_50x1_bit.goog_in21k on cifar10 (10% Subset) - Epsilon 1.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 103 | 0.000358 | 0.66 | 0.92 |
| 512 | 103 | 0.000358 | 0.66 | 0.92 |
| 1024 | 169 | 0.000957 | 0.20 | 0.93 |
| 2048 | 125 | 0.000442 | 5.74 | 0.91 |
| 4096 | 102 | 0.000442 | 7.47 | 0.92 |
| Full batch | 132 | 0.001660 | 0.20 | 0.91 |

### resnetv2_50x1_bit.goog_in21k on cifar10 (10% Subset) - Epsilon 2.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 103 | 0.000358 | 0.66 | 0.94 |
| 512 | 60 | 0.001109 | 0.20 | 0.94 |
| 1024 | 62 | 0.000820 | 0.20 | 0.94 |
| 2048 | 136 | 0.000918 | 1.48 | 0.94 |
| 4096 | 130 | 0.000466 | 1.96 | 0.92 |
| Full batch | 74 | 0.003058 | 0.20 | 0.93 |

### resnetv2_50x1_bit.goog_in21k on cifar10 (10% Subset) - Epsilon 4.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 33 | 0.001679 | 0.20 | 0.96 |
| 512 | 92 | 0.001534 | 0.20 | 0.96 |
| 1024 | 72 | 0.002322 | 0.20 | 0.96 |
| 2048 | 140 | 0.001497 | 3.01 | 0.96 |
| 4096 | 127 | 0.000635 | 2.05 | 0.94 |
| Full batch | 86 | 0.001169 | 1.07 | 0.94 |

### resnetv2_50x1_bit.goog_in21k on cifar10 (10% Subset) - Epsilon 8.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 103 | 0.000358 | 0.66 | 0.96 |
| 512 | 52 | 0.001451 | 0.20 | 0.97 |
| 1024 | 132 | 0.000827 | 2.09 | 0.97 |
| 2048 | 129 | 0.001736 | 1.57 | 0.96 |
| 4096 | 42 | 0.010348 | 0.85 | 0.96 |
| Full batch | 79 | 0.004977 | 0.56 | 0.96 |

### resnetv2_50x1_bit.goog_in21k on cifar10 (100% Subset) - Epsilon 1.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 158 | 0.000122 | 0.20 | 0.95 |
| 512 | 39 | 0.000272 | 0.20 | 0.95 |
| 1024 | 38 | 0.000500 | 0.20 | 0.95 |
| 2048 | 50 | 0.000960 | 0.20 | 0.95 |
| 4096 | 159 | 0.000802 | 0.23 | 0.95 |
| 8192 | 69 | 0.001405 | 0.20 | 0.95 |
| 16384 | 15 | 0.006159 | 5.54 | 0.95 |
| 32768 | 73 | 0.005944 | 0.32 | 0.95 |
| Full batch | 84 | 0.002569 | 0.20 | 0.95 |

### resnetv2_50x1_bit.goog_in21k on cifar100 (10% Subset) - Epsilon 0.25

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 28 | 0.000006 | 3.79 | 0.05 |
| 512 | 65 | 0.000001 | 3.00 | 0.04 |
| 1024 | 103 | 0.000358 | 0.66 | 0.05 |
| 2048 | 137 | 0.001081 | 3.02 | 0.05 |
| 4096 | 103 | 0.000358 | 0.66 | 0.04 |
| Full batch | 192 | 0.001287 | 5.40 | 0.05 |

### resnetv2_50x1_bit.goog_in21k on cifar100 (10% Subset) - Epsilon 0.5

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 200 | 0.000133 | 0.20 | 0.14 |
| 512 | 46 | 0.000381 | 0.61 | 0.11 |
| 1024 | 103 | 0.000358 | 0.66 | 0.12 |
| 2048 | 109 | 0.000869 | 0.27 | 0.13 |
| 4096 | 200 | 0.001803 | 2.05 | 0.15 |
| Full batch | 200 | 0.001816 | 6.26 | 0.14 |

### resnetv2_50x1_bit.goog_in21k on cifar100 (10% Subset) - Epsilon 1.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 132 | 0.000264 | 5.76 | 0.34 |
| 512 | 200 | 0.000393 | 5.71 | 0.33 |
| 1024 | 129 | 0.000407 | 1.64 | 0.32 |
| 2048 | 162 | 0.000930 | 4.01 | 0.33 |
| 4096 | 57 | 0.004018 | 2.27 | 0.31 |
| Full batch | 58 | 0.002291 | 0.20 | 0.32 |

### resnetv2_50x1_bit.goog_in21k on cifar100 (10% Subset) - Epsilon 2.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 155 | 0.000342 | 1.77 | 0.57 |
| 512 | 157 | 0.000404 | 3.46 | 0.58 |
| 1024 | 121 | 0.001391 | 0.20 | 0.56 |
| 2048 | 78 | 0.001760 | 1.90 | 0.56 |
| 4096 | 85 | 0.002192 | 3.73 | 0.57 |
| Full batch | 59 | 0.002987 | 8.32 | 0.58 |

### resnetv2_50x1_bit.goog_in21k on cifar100 (10% Subset) - Epsilon 4.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 141 | 0.000465 | 0.20 | 0.73 |
| 512 | 40 | 0.002112 | 0.20 | 0.73 |
| 1024 | 140 | 0.001261 | 0.20 | 0.73 |
| 2048 | 61 | 0.002304 | 100 | 0.73 |
| 4096 | 90 | 0.007371 | 0.20 | 0.73 |
| Full batch | 132 | 0.002786 | 0.20 | 0.74 |

### resnetv2_50x1_bit.goog_in21k on cifar100 (10% Subset) - Epsilon 8.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 133 | 0.000455 | 4.46 | 0.82 |
| 512 | 86 | 0.001475 | 0.58 | 0.82 |
| 1024 | 76 | 0.002174 | 0.75 | 0.81 |
| 2048 | 57 | 0.001919 | 6.71 | 0.83 |
| 4096 | 129 | 0.002697 | 0.20 | 0.83 |
| Full batch | 44 | 0.008800 | 1.33 | 0.84 |

### resnetv2_50x1_bit.goog_in21k on cifar100 (100% Subset) - Epsilon 1.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 97 | 0.000171 | 0.67 | 0.79 |
| 512 | 200 | 0.000147 | 2.35 | 0.79 |
| 1024 | 153 | 0.000310 | 0.20 | 0.79 |
| 2048 | 103 | 0.000358 | 0.66 | 0.79 |
| 4096 | 103 | 0.000358 | 0.66 | 0.78 |
| 8192 | 62 | 0.001036 | 0.20 | 0.79 |
| 16384 | 117 | 0.000346 | 2.19 | 0.75 |
| 32768 | 74 | 0.001925 | 0.20 | 0.78 |
| Full batch | 71 | 0.003514 | 0.20 | 0.79 |

### vit_base_patch16_224.augreg_in21k on cifar10 (10% Subset) - Epsilon 0.25

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 103 | 0.000358 | 0.66 | 0.92 |
| 512 | 86 | 0.000442 | 6.35 | 0.91 |
| 1024 | 152 | 0.000442 | 5.58 | 0.90 |
| 2048 | 53 | 0.001511 | 0.20 | 0.90 |
| 4096 | 151 | 0.001178 | 0.20 | 0.90 |
| Full batch | 154 | 0.002511 | 0.20 | 0.90 |

### vit_base_patch16_224.augreg_in21k on cifar10 (10% Subset) - Epsilon 0.5

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 103 | 0.000358 | 0.66 | 0.94 |
| 512 | 103 | 0.000358 | 0.66 | 0.95 |
| 1024 | 109 | 0.000442 | 5.99 | 0.95 |
| 2048 | 130 | 0.000442 | 5.64 | 0.95 |
| 4096 | 117 | 0.000442 | 5.68 | 0.93 |
| Full batch | 168 | 0.000891 | 0.20 | 0.94 |

### vit_base_patch16_224.augreg_in21k on cifar10 (10% Subset) - Epsilon 1.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 200 | 0.000349 | 0.20 | 0.96 |
| 512 | 103 | 0.000358 | 0.66 | 0.96 |
| 1024 | 103 | 0.000358 | 0.66 | 0.96 |
| 2048 | 103 | 0.000358 | 0.66 | 0.96 |
| 4096 | 67 | 0.000442 | 6.00 | 0.96 |
| Full batch | 68 | 0.000442 | 5.17 | 0.96 |

### vit_base_patch16_224.augreg_in21k on cifar10 (10% Subset) - Epsilon 2.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 52 | 0.000422 | 0.20 | 0.97 |
| 512 | 55 | 0.001510 | 0.20 | 0.98 |
| 1024 | 115 | 0.000442 | 5.72 | 0.97 |
| 2048 | 103 | 0.000358 | 0.66 | 0.96 |
| 4096 | 144 | 0.004442 | 0.20 | 0.97 |
| Full batch | 139 | 0.004877 | 0.20 | 0.97 |

### vit_base_patch16_224.augreg_in21k on cifar10 (10% Subset) - Epsilon 4.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 103 | 0.000358 | 0.66 | 0.98 |
| 512 | 103 | 0.000358 | 0.66 | 0.97 |
| 1024 | 118 | 0.000442 | 5.67 | 0.98 |
| 2048 | 63 | 0.001384 | 0.20 | 0.98 |
| 4096 | 58 | 0.004962 | 0.59 | 0.98 |
| Full batch | 92 | 0.005141 | 2.16 | 0.98 |

### vit_base_patch16_224.augreg_in21k on cifar10 (10% Subset) - Epsilon 8.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 103 | 0.000358 | 0.66 | 0.98 |
| 512 | 117 | 0.000442 | 5.70 | 0.98 |
| 1024 | 153 | 0.002276 | 0.20 | 0.99 |
| 2048 | 93 | 0.002015 | 0.20 | 0.99 |
| 4096 | 84 | 0.003689 | 0.20 | 0.99 |
| Full batch | 92 | 0.005141 | 2.16 | 0.98 |

### vit_base_patch16_224.augreg_in21k on cifar10 (100% Subset) - Epsilon 1.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 54 | 0.000140 | 3.85 | 0.98 |
| 512 | 200 | 0.000249 | 0.20 | 0.98 |
| 1024 | 103 | 0.000358 | 0.66 | 0.98 |
| 2048 | 103 | 0.000358 | 0.66 | 0.98 |
| 4096 | 53 | 0.001395 | 0.20 | 0.98 |
| 8192 | 103 | 0.000358 | 0.66 | 0.98 |
| 16384 | 61 | 0.000879 | 0.20 | 0.98 |
| 32768 | 74 | 0.002550 | 0.20 | 0.99 |
| Full batch | 90 | 0.005129 | 0.20 | 0.98 |

### vit_base_patch16_224.augreg_in21k on cifar100 (10% Subset) - Epsilon 0.25

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 37 | 0.000007 | 5.34 | 0.08 |
| 512 | 158 | 0.000442 | 5.73 | 0.07 |
| 1024 | 37 | 0.000007 | 5.34 | 0.09 |
| 2048 | 82 | 0.000012 | 100 | 0.09 |
| 4096 | 162 | 0.000014 | 8.25 | 0.09 |
| Full batch | 176 | 0.000442 | 4.33 | 0.10 |

### vit_base_patch16_224.augreg_in21k on cifar100 (10% Subset) - Epsilon 0.5

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 119 | 0.000442 | 3.81 | 0.22 |
| 512 | 167 | 0.000002 | 1.98 | 0.19 |
| 1024 | 200 | 0.000332 | 0.20 | 0.20 |
| 2048 | 117 | 0.000632 | 2.35 | 0.21 |
| 4096 | 141 | 0.002348 | 0.96 | 0.23 |
| Full batch | 190 | 0.003369 | 0.50 | 0.21 |

### vit_base_patch16_224.augreg_in21k on cifar100 (10% Subset) - Epsilon 1.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 127 | 0.000524 | 0.84 | 0.49 |
| 512 | 173 | 0.003597 | 2.59 | 0.57 |
| 1024 | 92 | 0.005141 | 2.16 | 0.51 |
| 2048 | 124 | 0.001273 | 0.20 | 0.52 |
| 4096 | 200 | 0.005765 | 3.63 | 0.51 |
| Full batch | 157 | 0.001286 | 6.26 | 0.46 |

### vit_base_patch16_224.augreg_in21k on cifar100 (10% Subset) - Epsilon 2.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 200 | 0.000544 | 100 | 0.75 |
| 512 | 103 | 0.000358 | 0.66 | 0.71 |
| 1024 | 65 | 0.008695 | 0.20 | 0.72 |
| 2048 | 135 | 0.001593 | 7.93 | 0.74 |
| 4096 | 87 | 0.002814 | 0.48 | 0.72 |
| Full batch | 57 | 0.010826 | 0.75 | 0.71 |

### vit_base_patch16_224.augreg_in21k on cifar100 (10% Subset) - Epsilon 4.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 146 | 0.001374 | 0.82 | 0.87 |
| 512 | 200 | 0.000606 | 0.20 | 0.88 |
| 1024 | 74 | 0.001124 | 0.20 | 0.84 |
| 2048 | 124 | 0.001942 | 1.40 | 0.86 |
| 4096 | 103 | 0.002767 | 1.78 | 0.85 |
| Full batch | 49 | 0.004626 | 1.49 | 0.84 |

### vit_base_patch16_224.augreg_in21k on cifar100 (10% Subset) - Epsilon 8.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 144 | 0.000442 | 5.83 | 0.93 |
| 512 | 110 | 0.002132 | 0.20 | 0.93 |
| 1024 | 200 | 0.001103 | 0.24 | 0.93 |
| 2048 | 93 | 0.002076 | 1.18 | 0.95 |
| 4096 | 70 | 0.002066 | 2.85 | 0.91 |
| Full batch | 75 | 0.003252 | 1.79 | 0.92 |

### vit_base_patch16_224.augreg_in21k on cifar100 (100% Subset) - Epsilon 1.0

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 97 | 0.000224 | 0.20 | 0.88 |
| 512 | 55 | 0.000308 | 0.20 | 0.88 |
| 1024 | 55 | 0.000744 | 0.20 | 0.89 |
| 2048 | 103 | 0.000358 | 0.66 | 0.88 |
| 4096 | 64 | 0.001513 | 0.20 | 0.88 |
| 8192 | 165 | 0.000540 | 0.27 | 0.89 |
| 16384 | 180 | 0.001282 | 0.20 | 0.89 |
| 32768 | 88 | 0.001745 | 3.07 | 0.88 |
| Full batch | 120 | 0.002176 | 0.20 | 0.89 |

## cifar100 (100% subset) - Epsilon 0.1 (aka. DP-RAFT Figure 5)

### resnetv2_50x1_bit.goog_in21k on cifar100 (100.0% Subset) - Epsilon 0.1

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 109 | 0.000070 | 2.94 | 0.32 |
| 512 | 68 | 0.000063 | 4.00 | 0.31 |
| 1024 | 62 | 0.000164 | 0.70 | 0.33 |
| 2048 | 47 | 0.000171 | 0.20 | 0.33 |
| 4096 | 179 | 0.000168 | 2.32 | 0.32 |
| 8192 | 96 | 0.000442 | 0.20 | 0.33 |
| 16384 | 135 | 0.000377 | 0.20 | 0.34 |
| 32768 | 89 | 0.001550 | 0.20 | 0.31 |
| Full batch | 200 | 0.001087 | 4.00 | 0.33 |

### vit_base_patch16_224.augreg_in21k on cifar100 (100.0% Subset) - Epsilon 0.1

| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256 | 60 | 0.000129 | 1.92 | 0.45 |
| 512 | 200 | 0.000442 | 2.10 | 0.44 |
| 1024 | 103 | 0.000358 | 0.66 | 0.46 |
| 2048 | 78 | 0.001499 | 0.20 | 0.50 |
| 4096 | 92 | 0.005141 | 2.16 | 0.52 |
| 8192 | 122 | 0.003308 | 0.64 | 0.54 |
| 16384 | 92 | 0.005141 | 2.16 | 0.53 |
| 32768 | 92 | 0.005141 | 2.16 | 0.49 |
| Full batch | 200 | 0.002707 | 1.98 | 0.49 |



