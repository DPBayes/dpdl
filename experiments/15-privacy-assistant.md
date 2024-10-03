# Privacy assistant evaluation

## Motivation

We have developed a machine learning model for predicting the privacy-utility-tradeoff curve based on epsilon-accuracy pairs. We have a prior assumption that the privacy-utility-tradeoff curves have sigmoidal shape. We want to:

(a) verify our assumption about the sigmoidal shape,

(b) test if the model can fit actual privacy-utility-tradeoff curves.

Furthermore, we have the "version 0" of the the privacy-assistant ready, and we want to evaluate its functionality in a scenario similar to real-world usage.

## Objective

We want to collect epsilon-accuracy pairs for multiple models and datasets.

## Methodology

We will fix the epochs at 40 and optimize the other hyperparameters (batch size, clipping bound, learning rate) using 20 trials of Bayesian optimization.

After the optimization is done, we will evaluate the resulting model on the test set.

## Models

Let's start by experimenting with a single model and expand later if necessary.

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

## Ranges for hyperparameter optimization

```
batch_size:
  min: 192
  max: -1
  type: int
learning_rate:
  max: 1.0e-1
  min: 1.0e-07
  type: float
  log_space: True
```

## Datasets

We will run the experiments at least with the following datasets:

- **dpdl-benchmark/caltech_birds2011 - 100% subset**
- **dpdl-benchmark/sun397 - 10% subset**
- **dpdl-benchmark/eurosat - 100% subset**
- **dpdl-benchmark/oxford_iiit_ - 100% subset**
- **dpdl-benchmark/plant_village - 10% subset**
- **dpdl-benchmark/colorectal_histology - 100% subset**
- **dpdl-benchmark/cassava - 100% subset**

## Epsilon Values

Conduct experiments with 20 epsilon values from `np.geomspace(0.25, 16, 19)`.

