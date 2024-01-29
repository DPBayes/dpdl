# Follow-up Experiment: Impact of Batch Size on Signal-to-Noise Ratio

## Motivation

This experiment aims to investigate the surprising observation from our previous study that batch size does not significantly impact accuracy in differential private deep learning models. We will focus on analyzing the signal-to-noise ratio per training step across different batch sizes. Specifically, we will track the gradient norms (signal) and the amount of noise added during training. The goal is to uncover insights into the relationship between batch size and model performance in the context of differential privacy.

The experiment is expected to provide clarity on whether the signal-to-noise ratio varies significantly across different batch sizes and how this variation correlates with model accuracy.

## Hypothesis

Our hypothesis is that differences in the signal-to-noise ratio across various batch sizes might explain why batch size does not seem to impact accuracy as suggested by existing literature.

## Objective

1. To track and compare the gradient norms and noise levels per training step across different batch sizes.
2. To analyze the signal-to-noise ratio for each batch size and examine its correlation with model performance.

## Methodology

- **Batch Sizes Selection**: Use batch sizes from the previous experiment, along with their optimized hyperparameters (epochs, learning_rate, max_grad_norm).
- **Gradient Norm Tracking**: For each batch size, record the gradient norms (signal) per training step.
- **Noise Measurement**: Quantify the amount of noise added per training step for each batch size.
- **Signal-to-Noise Ratio Calculation**: Compute the signal-to-noise ratio for each training step and batch size.
- **Data Visualization**: Plot the gradient norms, noise levels, and signal-to-noise ratios to visually assess trends and patterns.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
- **ResNet-50 (resnetv2_50x1_bitm_in21k)**

## Datasets

- **CIFAR-10 (10% and Full Dataset)**
- **CIFAR-100 (10% and Full Dataset)**

## Epsilon Values

Consistent with the previous experiment, we will use epsilon values of \{0.25, 0.5, 1, 2, 4, 8\} for the 10% subsets. For the full datasets, we will continue with epsilon=1.

