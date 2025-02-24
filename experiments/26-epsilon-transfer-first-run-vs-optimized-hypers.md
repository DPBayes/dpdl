# ε to ε -transfer: First run vs Optimized hypers

## Motivation

In [experiment 24](24-hypers-as-a-function-of-epsilon.md) we started exploring the idea of transferring hyperparameters from one epsilon to another.

To better understand the problem, we want to compare the results when optimizing the hyperprameters from scratch for each epsilon and simply using the hyperparameters optimized for the first epsilon as-is.

## Objective

We will first tune the hyperparameters for each epsilon and record the accuracy. Then we will take each the found hyperparameters and train using those on all the epsilon and record the accuracy.

## Methodology

1. We will fix the epochs at 40 and optimize the other hyperparameters (batch size, clipping bound, learning rate) using 20 trials of Bayesian optimization.

After the optimization is done, we will evaluate the resulting model on the test set.

2. We will train the same model using each set of the hyperparameters found in the previous step and evaluate the model on the test test.

We will also repeat the step 2 with the best hyperparameters from the [experiment 24](24-hypers-as-a-function-of-epsilon.md) grid. We refer to this method as "grid HPO".

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

## Ranges for hyperparameter optimization

```
batch_size:
  min: 192
  max: -1
  type: int
learning_rate:
  max: 0.1
  min: 1.0e-5
  type: float
  log_space: True
max_grad_norm:
  max: 10
  min: 1.0e-4
  type: float
  log_space: True
```

## Datasets

We will use the same datasets as in [experiment 24]([experiment 24](24-hypers-as-a-function-of-epsilon.md).

- **cifar100 - 10% subset**
- **dpdl-benchmark/sun397 - 10% subset**
- **dpdl-benchmark/patch_camelyon - 2% subset**
- **dpdl-benchmark/cassava - 100% subset**
- **dpdl-benchmark/svhn_cropped - 10% subset**
- **dpdl-benchmark/svhn_cropped_balanced - 10% subset**

## Epsilon values

We will conduct experiment with = { 0.5, 1, 2, 4, 8, 16 }.

