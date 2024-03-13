# Hyperparameter variation with fixed epochs

## Motivation

In the previous hyperparameter variation experiments (00, 02, 03, 04), we observed that varying batch sizes and maximum gradient norms had minimal impact on model accuracy, challenging existing assumptions about hyperparameter significance. In this experiment we aim to study this further.

We fix the number of epochs while repeating the variation experiments for the other hyperparameters, such as learning rates and batch sizes, to isolate their effects. By keeping the number of epochs constant, we aim isolate the effects of other hyperparameters on the model's learning dynamics and performance. We hope this to more accurately measure the impact of learning rate, batch size, and gradient norm without the confounding variable of changing epochs. Furthermore, fixing the epochs enables us to more efficiently study the signal-to-noise ratio in future experiments.

## Objective

We investigate the effect of varying batch size, learning rate, or maximum gradient norm while keeping fixing the dependant variables AND epochs. As before, we will optimize the other variables using Bayesian optimization. We record the resulting accuracy and the values of the hyperparameters.

## Methodology

### Models

We use the same models as in the previous experiments.

### Datasets

We use using the same datasets as in the previous experiments.

### Epsilon values

We use the same epsilon values as in the previous experiments.

