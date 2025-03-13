# Prediction errors of difficult examples

## Motivation

We have recently understood that very small clipping bounds can work well for some datasets. Also, we hav knowledge that too small clipping bounds can hurt fairness. These leads to the train of thought that maybe selecting the clipping bound can be seen as a bias-variance tradeoff problem?

The intuition behind this bias-variance tradeff would be that small clipping bounds limit the effect of the difficult examples (outliers), facilitating better macro accuracy (but worse micro accuracy) when the dataset has not too many difficult examples. On the other hand hand, large clipping bounds would allow the model to better fit the difficult examples facilitating a better micro accuracy, but hurting the macro accuracy as a tradeoff. 

## Objective

We know that the FashionMNIST datasets has easily confusable classes (and thus examples), so we will train the modest on this dataset. However, we have no idea of the performance on this dataset at different privacy levels nor an idea about the correct hyeprparameters.

_So at initial phase, we will simply run the standard 20 trial HPO round on FashionMNIST for various privacy budgets and clipping bounds and continue from there._

We will also save the models resulting from the HPO (with the goal of taking a closer look at the prediction errors).

After this initial exploration is done, we will consider next steps such as introducing imbalance to the training.

## Methodology

We will fix the epochs at 40, clipping bound at { 1e-15, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10 }, and optimize the other hyperparameters (batch size, learning rate) using 20 trials of Bayesian optimization.

We will conduct experiment with ε = { 1, 2, 4, 8 }.

After the optimization is done, we will evaluate the resulting model on the test set and save the final model.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

We will use the FiLM parameterization for efficiency.

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
```

## Datasets

- **FashionMNIST - 10% subset**
