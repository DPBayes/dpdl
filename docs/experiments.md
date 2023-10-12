# Experiment Design

## Individual experiments

00 - [Batch size variation](00-experiment-batch-size-variation.md)

## Questions/ideas

- Use LoRA instead of FiLM?
- Adam vs SGD with Momentum?
- ConvNeXT (and other modern convolutional nets) vs ResNet-50?

## TODO

- Use FiLM adaptor instead of training head/all
- Save the optuna study in experiment directory (if we want to try more trials)
- BO search for learning rate in log space
- Repeat experiments with different seeds (confidence interval and to avoid getting stuck in bad local minima by chance)

## Handled questions/ideas

- Zero the head weights
- Use test set for calculating final accuracy
- Which epsilons to use? Current plan of epsilon = \{1,2,...,8\} maybe too much?
    - Lots of action in the small epsilons so try epsilon = {0.25, 0.5, 1, 2, 4, 8}
- How about comparing the results against subsampling ratio instead of batch size?
    - Let's not do this.
- Compare with adaptive clipping?
    - Could be a good idea. Interesting would be to benchmark all the adaptive methods.
- Are the steps in batch sizes too dense?
    - Yes. Make the larger end of batch sizes less dense by growing f.ex. exponentially.
- Fixed vs optimized epochs?
    - Optimize epochs.
- Other datasets?
    - Ditch CIFAR10 (too easy)
    - Use full CIFAR100 and 10% subset CIFAR100
    - Later, let's use also SVHN
- Other models?
    - ResNet-50 and ViT are good because other papers use them
- Other metrics to track/optimize?
    - Let's just use accuracy
- Search hypers in log space from some params?
    - Yes, for example learning rate.
- Sensitivity analysis of individual hypers?
    - Maybe later.
- Define baseline hypers?
    - If we do the sensitivity analysis.

## Future experiments

- How does dataset imbalance affect the hypers?

## Goal of the experiments

We want to analyze the privacy-utility tradeoff for the hyperparameters and the interaction of the hyperparameters.

From the results of the analysis, we aim to develop methods for predicting good hyperparameters for the model in response to alterations of the privacy budget.

## General method overview

- We decouple the learning rate and clipping amount according to De et al. (2022).
- We train either using FiLM adapters or LoRA.
- We do _not_ train all parameters or head only.
- For CIFAR100, we use the training set (50000 examples) as is. We then divide the given testing set (10000 examples) into a training set (5000 examples) and a validation set (5000 examples).
- When training using a subset of the data, the training and validation data is divided according to the proportions. For examples, for 10% of CIFAR100 that is 5000 training examples and 500 validation/test examples.
- For the Bayesian optimization we use multiclass accuracy of the validation set as optimization objective.

We run 20 trials of Bayesian optimization using the following bounds for the other hypers

```
learning_rate:
  max: 1.0e-2
  min: 1.0e-07
  type: float
epochs:
  max: 200
  min: 1
  type: int
max_grad_norm:
  max: 10
  min: 0.2
  type: float
```

We then train a final model with the best params first on the training set and then on the validation set. The metrics for the final model are calculated using the test set (5000 examples).

### General method overview in pseudo-code

1. Initialize Bayesian Optimization (BO) system with a range of hyperparameters
   BO_system <- BayesianOptimization(hyperparameter_space)

2. For i <- 1 to n_BO_steps do

   2.1 hyperparams <- BO_system.get_hyperparameters()

   2.2 model <- model(hyperparams)

   2.3 model <- train_model(training_data)

   2.4 validation_metric <- evaluate_model(model, validation_data)

   2.5 BO_system.update(hyperparams, validation_metric)

3. best_hyperparams <- BO_system.get_best_hyperparameters()

4. final_model <- model(best_hyperparams)

5. train_model(final_model, training_data + validation_data)

6. final_metric_value <- evaluate_model(final_model, test_data)

7. Report the final metric value

