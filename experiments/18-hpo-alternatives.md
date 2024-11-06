# Alternative Approaches to HPO

## Motivation

We have figured out that the HPO seems to be unstable after performing 20 trials that we are mostly limited to due to using large models and "real-world" datasets. Hence, we have been thinking if alternative approaches to HPO could make the optimization more stable. In essence, we want to test altenative approaches for HPO to see if they are more effective or not.

## Objective

We aim to compare three different methods

(1) fix the batch size and clipping bound to suitables values as they seem to be somewhat irrelevant when fine-tuning,

(2) seed the HPO with suitable parameters to use in addition to random trials as warmup,

(3) what we are currently doing, just optimize all the free variables.

The goal is to record the achieved accuracy on some dataset and see if there's any difference between these method.

## Methodology

For (1) we will fix the batch size at full batch and and clipping bound as 1. From earlier experiments, we have seen that full batch tends to perform well (NB: All batch sizes tend to do well). And the motivation for fixing the clipping bounds comes from the DeepMinds "Unlocking high-accuracy..." by De et. al (2022) paper, plus works of other groups, such as Mittal's group in Princeton.

For (2) we will use 5 manually defined trials and 5 random trials. How we collect the 5 hypers will alternate according to datasets

- For cifar10_10pct_plus_cifar100_humans we will use guesses based on common sense and intuition from the earlier research,
- For 10% of cifar100, we will check the HPO logs for some pretty good hypers (naturally ignoring the best ones)
- For 10% of svhn_cropped_balanced, we will use the best and good hyperparameters from our previous runs on cifar10. The idea is that other than the content of the images, the datasets are equal, and perhaps the hypers work

For (3) we just do what we always do and optimize all the free hypers.

We will run a total of 20 trials of HPO. For optimization, we will use DP-Adam, as usually.

As the epochs are a major confounder, for this initial experiment we will fix the epochs at 40. Also, we will conduct the experiments with ε=4.0.

## Models

We will conduct the experiment using a single model and we will train FiLM parameters:

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**

## Ranges for hyperparameter optimization

```
batch_size:
  max: 192 
  min: -1        
  type: int
learning_rate:
  max: 0.1
  min: 1.0e-7
  type: float
  log_space: True
max_grad_norm:
  max: 300
  min: 0.1
  type: float
  log_space: True
```

## Datasets

- **datasets/dpdl-benchmark/cifar10_10pct_plus_cifar100_humans - 100% subset**
- **datasets/cifar100 - 10% subset**
- **datasets/dpdl-benchmark/svhn_cropped_balanced - 10% subset**

## Hyperparameters for manual trials
We will use the following hypers for the manual trials

- **datasets/dpdl-benchmark/cifar10_10pct_plus_cifar100_humans - 100% subset**

```
batch_size,learning_rate,max_grad_norm
-1,0.0001,100
-1,0.001,200
4096,0.005,0.2
-1,0.00001,30
4096,0.000005,5
```

- **datasets/cifar100 - 10% subset**

```
batch_size,learning_rate,max_grad_norm
-1,0.01146721822610171,8.032763561513777
512,0.0018234252312577773,5.197043613978144
4096,0.0006622044316498266,0.2
2048,0.0013000468531533824,0.2
1024,0.008388927125427848,2.6501643524991607
```

- **datasets/dpdl-benchmark/svhn_cropped_balanced - 10% subset**

```
batch_size,learning_rate,max_grad_norm
552,0.002555,0.399278
2624,0.001801,3.522622
1720,0.001864,3.315861
-1,0.0011933853983893783,4.273636041943903
2048,0.000834180101442897,3.216640876127639
```

