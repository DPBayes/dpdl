# Experiment Design

## Individual experiments

[ ] [FiLM vs LoRA evaluation](01-film-vs-lora-evaluation.md)

[ ] [Batch size variation](00-experiment-batch-size-variation.md)

## Questions/ideas

- Before running others stuff, experiment with FiLM vs LoRA.
- Adam vs SGD with Momentum?
- More trials! Gauri noticed that we need about 50 trials to find the hyperparameters.

## TODO

- Repeat experiments with different seeds (confidence interval and to avoid getting stuck in bad local minima by chance)

## Future experiments

- From the DP-RAFT paper Antti had some ideas of experimenting with noise thresholding.
- How hypers transfer from PEFT method Head only to LoRA, FiLM, None?
- How does dataset imbalance affect the hypers?
- How does the number of trainable parameters affect the optimized hypers?
- Sensitivity analysis of hypers: Which hypers need to be tuned exactly and which are more robust.

## Goal of the experiments

We want to analyze the privacy-utility tradeoff for the hyperparameters and the interaction of the hyperparameters.

From the results of the analysis, we aim to develop methods for predicting good hyperparameters for the model in response to alterations of the privacy budget.

## General method overview

- We decouple the learning rate and clipping amount according to De et al. (2022).
- We convert `epochs` to total number of steps. Most importantly, this enables us to use smooth sample rates. When using epochs, Opacus forces the sample rate to be of form `int(1/i)` where `i` is the number of batches in an epoch.
- We train either using FiLM adapters or LoRA.
- We do _not_ train all parameters or head only.
- For CIFAR100, we split the official training dataset (50k examples) into training set (45k examples) and validation set (5k) examples. We use the given testing set (10k examples) as is for evaluating the final model(s).
- When training using a subset of the data, the training and validation data is divided according to the proportions. For examples, for 10% of CIFAR100 that is 4.5k training examples, 0.5k validation examples, and 1k test examples.
- For the Bayesian optimization we use multiclass accuracy of the validation set as optimization objective.

We run 20 trials of Bayesian optimization using the following bounds for the other hypers

```
batch_size:
  options:
  - 256
  - 512
  - 1024
  - 2048
  - 4096
  - 8192
  - 16384
  - 32768
  - 45000
  type: categorical
learning_rate:
  max: 1.0e-1
  min: 1.0e-07
  type: float
  log_space: True
epochs:
  max: 200
  min: 1
  type: int
max_grad_norm:
  max: 10
  min: 0.2
  type: float
```

We then train a final model with the best params first on the training set and then on the validation set. The metrics for the final model are calculated using the test set (10k examples).

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

## Handled questions/ideas

- Use batch sizes that actually have effect on the sampling ratio. (Check Marlon's CSV.)
    - We'll use B/N and use steps instead of epochs.
- ConvNeXT (and other modern convolutional nets) vs ResNet-50?
    - ResNet-50 is fine.
- Use LoRA instead of FiLM?
    - We have both.
- Use FiLM adaptor instead of training head/all
- BO search for learning rate in log space
- Zero the head weights
- Save the optuna study in experiment directory (if we want to try more trials)
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

