# Hyperparameters vs. number of classes

## Motivation

We have noticed that for e.g. SUN397 and ImageNet397 the models prefer a much larger clipping bound than for the other datasets. However, we have not found a reason that causes this. One hypothesis is that this is related to the relative large number of classes (397) that these datasets have.

## Objective

We will train a pretrained a model on multiple ImageNet-mini variants containing 64, 128, 256, or 512 classes. These dataset are sampled from the ImageNet-1k dataset which is separate from the ImageNet-21k that our models are pretrained on.

We will run the typical 20 trials of Bayesian optimization for finding the (near) optimal hypers. As usually, we will fix the epochs at 40 and optimized the free hyperparameters (learning rate, batch size, clipping bound).

We will record the hyperparameters and the accuracy achieved.

## Methodology

We will fix the epochs at 40 and optimize the free hyperparameters (learning rate, batch size, clipping bound) using 20 trials of Bayesian optimization.

We will conduct experiment with ε = { 1, 2, 4, 8 }.

After the optimization is done, we will evaluate the resulting model on the test set.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
  - NB: The model is retrained on ImageNet-21k which is separate from ImageNet-1k.

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
  min: 1.0e-5
  type: float
  log_space: True
```

## Datasets

- **imagenet-mini-64 - 100% subset**  
  - Derived from ImageNet-1k, containing 64 classes  
- **imagenet-mini-128 - 100% subset**  
  - Derived from ImageNet-1k, containing 128 classes  
- **imagenet-mini-256 - 100% subset**  
  - Derived from ImageNet-1k, containing 256 classes  
- **imagenet-mini-512 - 100% subset**  
  - Derived from ImageNet-1k, containing 512 classes  
