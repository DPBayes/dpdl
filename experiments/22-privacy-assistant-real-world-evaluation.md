# Privacy assistant evaluation

## Motivation

We have been developing the privacy assistant for quite a while and the basic functionality seems to be working. Next we need to demonstrate that the Privacy assistant works with "real-world" use case.

## Objective

We will collect privacy-accuracy pairs for a multitude (100) of privacy budgets (ε) on different datasets.

This data will then be used to determine the accuracy for the privacy as

## Methodology

We will fix the epochs at 40 and optimize the other hyperparameters (batch size, clipping bound, learning rate) using 20 trials of Bayesian optimization.

After the optimization is done, we will evaluate the resulting model on the test set.

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
  max: 150 
  min: 0.1
  type: float
  log_space: True

```

## Datasets

We will run the experiments at least with:

- **cifar100 - 10% subset**
- **dpdl-benchmark/svhn_cropped - 10% subset**

If we possible given time limits, we will expand to few more interesting datasets:

- **dpdl-benchmark/sun397 - 10% subset**
- **dpdl-benchmark/patch_camelyon - 2% subset**
- **dpdl-benchmark/cassava - 100% subset**

Moreover, we will also run try to run *full* CIFAR-100 and SVHN if time permits.

## Epsilon Values

We will conduct experiment with 100 epsilon values from `np.geomspace(0.01, 16, 100)`.

