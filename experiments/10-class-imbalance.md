# Effect of class imbalance on maximum gradient norm

## Motivation

In the previous experiment we noticed that maximum gradient norm does not seem affect the resulting accuracy. However, in fairness experiments we can see there exists an optimal maximum gradient norm.

The purpose of this experiment is two-fold:

(a) To understand the possible effect of class imbalance on the maximum gradient norm,

(b) To verify if a [bug](https://github.com/PROBIC/dpdl/commit/790e2dc4d05068ddd3c409e8983b05be0785140f) in the code in backpropagation could have caused the earlier result that the maximum gradient norm does not affect the accuracy.

## Objective

The objective is to investigate the impact of class imbalance on the optimal configuration of maximum gradient norms and other hyperparameters (learning rate, batch size). We will create class imbalances by sampling from balanced datasets according to an exponential distribution [1](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf) and will compare these results to those from balanced datasets.

## Methodology

- **Class Imbalance Creation**: Artificially create class imbalances from balanced datasets using exponential distribution with parameters 0.1 and 0.01.
- **Balanced Datasets**: Conduct parallel experiments using balanced datasets for comparison.
- **Maximum Gradient Norm Variation**: Test the following set values for maximum gradient norms:
  - Max gradient norms: 0.005, 0.012, 0.028, 0.067, 0.158, 0.375, 0.889, 2.108, 5.0 (`np.round(np.geomspace(5e-3, 5, 9), 3)`)
- **Bayesian Optimization**: Optimize learning rate and batch size while fixing the number of epochs at 40. We will perform 20 trials of Bayesian optimization.

## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
- **Parameterization**: We will train using the FiLM parameterization. We will also zero the head weights as a standard practice.

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

- **CIFAR-10 (10% Subset)**: Use a 10% subset of CIFAR-10 to understand the impact of class imbalance on a smaller scale.
- **CIFAR-100 (10% Subset)**: Use a 10% subset of CIFAR-100 for a more diverse class distribution and to observe the effects of imbalance.

## Epsilon Values

Conduct experiments with epsilon values of {0.25, 1, 4}.

## Experiment Setup

For each combination of class distribution, model, dataset, maximum gradient norm, and epsilon value, record:

- The class distribution parameter (0.1, 0.01, balanced)
- The maximum gradient norm used
- Optimized learning rate
- Optimized batch size
- Accuracy with macro and micro averaging
