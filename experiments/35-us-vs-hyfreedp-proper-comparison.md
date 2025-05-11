# Full proper comparison: Us vs HyperFreeDP

## Motivation

In the paper [Towards Hyperparameter Free Optimization with Differential Privacy](https://openreview.net/pdf?id=2kGKsyhtvh), the authors claim state-of-the-art results.

We will use this as a baseline and need a comparison.

## Objective

We will evaluate the following algorithm for determining the hyperparameters

![Algorithm for hyperparameters](images/35-hyperparameter-flow.jpg)

We will train the same models on the same datasets and record the resulting accuracy to compare against HyFreeDP.

## Methodology

We will determine the hyperparameters following the algorithm above.

For the non-DP run, we will use the following hyperparameters
- Epochs: 10
- Learning rate: 1e-4
- Batch size: 256

### Real difficulty (NonDP-GS results from the HyFreeDP paper)

| Model     | CIFAR10 | CIFAR100 | SVHN  | GTSRB | Food101 | SUN397 |
|-----------|---------|----------|-------|-------|---------|--------|
| ViT-Small | 98.34   | 89.21    | 95.42 | 98.64 | 85.32   | N/A    |
| ViT-Base  | 98.55   | 92.37    | 96.75 | 98.39 | 89.53   | N/A    |

### Inferred difficulty metrics

#### Similarity-Based Measures ([from embedding similarity analysis](https://arxiv.org/abs/2404.05981)):
- **S_E**: Expected intra-class similarity (how similar examples from the same class are). Higher => confusable.
- **S_R**: Expected inter-class similarity (how similar examples from different classes are). Lower => More difficult.
- **bar_S**: Normalized similarity value combining S_E and S_R.
- **difficulty_sim**: Raw difficulty estimate: `S_R / (S_E + ε)`. Higher => more confusable classes.

---

#### Accuracy & Learning Dynamics:
- **base_difficulty**: `1 - test_mean`, direct error rate from test performance.
- **cv_full**: Coefficient of variation of per-class test accuracies (std / mean).
- **cv_trimmed**: Same as above but trimmed (removes outliers from both ends).
- **difficulty**: Composite difficulty = base_difficulty + weighted gap + weighted CV.
- **gap**: Generalization gap = train_mean - test_mean.

---

#### Class Distribution:
- **gini**: Gini impurity of the class distribution (0 = imbalanced, ~1 = balanced).
- **imbalance_factor**: `1 - gini`, more intuitive: 1 = perfectly balanced, 0 = fully imbalanced.
- **lambda_s**: Optional scaling parameter from similarity analysis (used in some S-bar variants).
- **num_classes**: Number of unique classes in the dataset.

---

#### Per-Class Accuracy Stats (on test set):
- **per_class_min**: Worst-performing class accuracy.
- **per_class_max**: Best-performing class accuracy.
- **per_class_median**: Median of all per-class accuracies.
- **per_class_p25**: 25th percentile (Q1) of per-class accuracies.
- **per_class_p75**: 75th percentile (Q3) of per-class accuracies.
- **std**: Standard deviation of per-class accuracies.

---

#### Overall Accuracies:
- **test_mean**: Mean classification accuracy over the test set.
- **train_mean**: Mean classification accuracy over the train set (final step).

### Datasets as columns


## Model: vit_base_patch16_224.augreg_in21k

| metric           |   dpdl-benchmark/sun397 |   food101 |   cifar100 |   dpdl-benchmark/gtsrb |   cifar10 |   dpdl-benchmark/svhn_cropped |
|:-----------------|------------------------:|----------:|-----------:|-----------------------:|----------:|------------------------------:|
| S_E              |                   0.124 |     0.191 |      0.257 |                  0.277 |     0.346 |                         0.317 |
| S_R              |                   0.499 |     0.45  |      0.588 |                  0.571 |     0.63  |                         0.48  |
| bar_S            |                   0.594 |     0.565 |      0.583 |                  0.574 |     0.571 |                         0.541 |
| base_difficulty  |                   0.222 |     0.084 |      0.067 |                  0.161 |     0.011 |                         0.062 |
| cv_full          |                   0.219 |     0.072 |      0.062 |                  0.252 |     0.011 |                         0.03  |
| cv_trimmed       |                   0.173 |     0.048 |      0.046 |                  0.172 |     0.011 |                         0.03  |
| difficulty       |                   0.376 |     0.113 |      0.108 |                  0.323 |     0.02  |                         0.072 |
| difficulty_sim   |                   4.032 |     2.352 |      2.289 |                  2.062 |     1.818 |                         1.512 |
| gap              |                   0.135 |     0.01  |      0.035 |                  0.154 |     0.007 |                        -0.01  |
| gini             |                   0.995 |     0.99  |      0.99  |                  0.964 |     0.9   |                         0.885 |
| hat_S            |                   0.752 |     0.575 |      0.563 |                  0.515 |     0.45  |                         0.339 |
| lambda_s         |                   0.5   |     0.5   |      0.5   |                  0.5   |     0.5   |                         0.5   |
| num_classes      |                 397     |   101     |    100     |                 43     |    10     |                        10     |
| per_class_max    |                   1     |     1     |      1     |                  1     |     0.998 |                         0.964 |
| per_class_median |                   0.82  |     0.936 |      0.95  |                  0.921 |     0.994 |                         0.947 |
| per_class_min    |                   0.148 |     0.584 |      0.68  |                  0.033 |     0.969 |                         0.891 |
| per_class_p25    |                   0.69  |     0.896 |      0.91  |                  0.791 |     0.985 |                         0.914 |
| per_class_p75    |                   0.913 |     0.952 |      0.97  |                  0.967 |     0.996 |                         0.961 |
| std              |                   0.17  |     0.066 |      0.058 |                  0.212 |     0.011 |                         0.028 |
| test_mean        |                   0.778 |     0.916 |      0.933 |                  0.839 |     0.989 |                         0.938 |
| train_mean       |                   0.912 |     0.927 |      0.968 |                  0.993 |     0.996 |                         0.928 |


## Model: vit_base_patch16_clip_224.metaclip_2pt5b

| metric           |   dpdl-benchmark/sun397 |   cifar100 |   dpdl-benchmark/gtsrb |   food101 |   cifar10 |   dpdl-benchmark/svhn_cropped |
|:-----------------|------------------------:|-----------:|-----------------------:|----------:|----------:|------------------------------:|
| S_E              |                   0.11  |      0.175 |                  0.245 |     0.173 |     0.277 |                         0.356 |
| S_R              |                   0.518 |      0.552 |                  0.759 |     0.495 |     0.634 |                         0.731 |
| bar_S            |                   0.602 |      0.594 |                  0.628 |     0.58  |     0.589 |                         0.594 |
| base_difficulty  |                   0.207 |      0.086 |                  0.025 |     0.075 |     0.012 |                         0.041 |
| cv_full          |                   0.201 |      0.071 |                  0.033 |     0.061 |     0.011 |                         0.02  |
| cv_trimmed       |                   0.154 |      0.058 |                  0.023 |     0.042 |     0.011 |                         0.02  |
| difficulty       |                   0.373 |      0.146 |                  0.049 |     0.111 |     0.024 |                         0.051 |
| difficulty_sim   |                   4.699 |      3.154 |                  3.1   |     2.861 |     2.293 |                         2.055 |
| gap              |                   0.178 |      0.061 |                  0.025 |     0.03  |     0.012 |                         0     |
| gini             |                   0.995 |      0.99  |                  0.964 |     0.99  |     0.9   |                         0.885 |
| hat_S            |                   0.787 |      0.683 |                  0.677 |     0.65  |     0.564 |                         0.513 |
| lambda_s         |                   0.5   |      0.5   |                  0.5   |     0.5   |     0.5   |                         0.5   |
| num_classes      |                 397     |    100     |                 43     |   101     |    10     |                        10     |
| per_class_max    |                   1     |      1     |                  1     |     1     |     0.997 |                         0.976 |
| per_class_median |                   0.821 |      0.94  |                  0.983 |     0.936 |     0.991 |                         0.965 |
| per_class_min    |                   0.074 |      0.73  |                  0.85  |     0.624 |     0.963 |                         0.916 |
| per_class_p25    |                   0.711 |      0.88  |                  0.971 |     0.896 |     0.986 |                         0.958 |
| per_class_p75    |                   0.914 |      0.96  |                  0.998 |     0.96  |     0.995 |                         0.971 |
| std              |                   0.159 |      0.065 |                  0.032 |     0.057 |     0.01  |                         0.019 |
| test_mean        |                   0.793 |      0.914 |                  0.975 |     0.925 |     0.988 |                         0.959 |
| train_mean       |                   0.971 |      0.975 |                  1     |     0.955 |     1     |                         0.96  |


## Model: vit_small_patch16_224.augreg_in21k

| metric           |   dpdl-benchmark/sun397 |   cifar100 |   dpdl-benchmark/gtsrb |   food101 |   cifar10 |   dpdl-benchmark/svhn_cropped |
|:-----------------|------------------------:|-----------:|-----------------------:|----------:|----------:|------------------------------:|
| S_E              |                   0.115 |      0.195 |                  0.21  |     0.166 |     0.305 |                         0.288 |
| S_R              |                   0.564 |      0.609 |                  0.653 |     0.461 |     0.668 |                         0.506 |
| bar_S            |                   0.612 |      0.603 |                  0.611 |     0.574 |     0.591 |                         0.554 |
| base_difficulty  |                   0.253 |      0.096 |                  0.13  |     0.112 |     0.015 |                         0.08  |
| cv_full          |                   0.25  |      0.077 |                  0.22  |     0.091 |     0.012 |                         0.041 |
| cv_trimmed       |                   0.195 |      0.058 |                  0.112 |     0.06  |     0.012 |                         0.041 |
| difficulty       |                   0.381 |      0.136 |                  0.238 |     0.134 |     0.024 |                         0.093 |
| difficulty_sim   |                   4.922 |      3.125 |                  3.105 |     2.772 |     2.192 |                         1.753 |
| gap              |                   0.059 |      0.023 |                  0.106 |    -0.017 |     0.005 |                        -0.013 |
| gini             |                   0.995 |      0.99  |                  0.964 |     0.99  |     0.9   |                         0.885 |
| hat_S            |                   0.797 |      0.68  |                  0.678 |     0.639 |     0.544 |                         0.43  |
| lambda_s         |                   0.5   |      0.5   |                  0.5   |     0.5   |     0.5   |                         0.5   |
| num_classes      |                 397     |    100     |                 43     |   101     |    10     |                        10     |
| per_class_max    |                   1     |      1     |                  1     |     1     |     0.995 |                         0.96  |
| per_class_median |                   0.805 |      0.92  |                  0.947 |     0.912 |     0.988 |                         0.927 |
| per_class_min    |                   0.037 |      0.66  |                  0.083 |     0.528 |     0.958 |                         0.85  |
| per_class_p25    |                   0.643 |      0.88  |                  0.829 |     0.856 |     0.986 |                         0.9   |
| per_class_p75    |                   0.881 |      0.95  |                  0.98  |     0.944 |     0.992 |                         0.952 |
| std              |                   0.187 |      0.07  |                  0.191 |     0.081 |     0.012 |                         0.038 |
| test_mean        |                   0.747 |      0.904 |                  0.87  |     0.888 |     0.985 |                         0.92  |
| train_mean       |                   0.806 |      0.927 |                  0.976 |     0.87  |     0.99  |                         0.907 |


## Model: vit_tiny_patch16_224.augreg_in21k

| metric           |   dpdl-benchmark/sun397 |   cifar100 |   dpdl-benchmark/gtsrb |   cifar10 |   food101 |   dpdl-benchmark/svhn_cropped |
|:-----------------|------------------------:|-----------:|-----------------------:|----------:|----------:|------------------------------:|
| S_E              |                   0.136 |      0.154 |                  0.188 |     0.268 |     0.265 |                         0.353 |
| S_R              |                   0.557 |      0.533 |                  0.615 |     0.619 |     0.507 |                         0.595 |
| bar_S            |                   0.605 |      0.595 |                  0.607 |     0.588 |     0.561 |                         0.56  |
| base_difficulty  |                   0.318 |      0.138 |                  0.112 |     0.031 |     0.16  |                         0.089 |
| cv_full          |                   0.275 |      0.095 |                  0.131 |     0.021 |     0.115 |                         0.046 |
| cv_trimmed       |                   0.22  |      0.077 |                  0.098 |     0.021 |     0.086 |                         0.046 |
| difficulty       |                   0.506 |      0.208 |                  0.215 |     0.049 |     0.202 |                         0.112 |
| difficulty_sim   |                   4.084 |      3.456 |                  3.276 |     2.308 |     1.915 |                         1.683 |
| gap              |                   0.156 |      0.063 |                  0.108 |     0.015 |    -0.004 |                        -0     |
| gini             |                   0.995 |      0.99  |                  0.964 |     0.9   |     0.99  |                         0.885 |
| hat_S            |                   0.755 |      0.711 |                  0.695 |     0.567 |     0.478 |                         0.406 |
| lambda_s         |                   0.5   |      0.5   |                  0.5   |     0.5   |     0.5   |                         0.5   |
| num_classes      |                 397     |    100     |                 43     |    10     |   101     |                        10     |
| per_class_max    |                   0.991 |      1     |                  1     |     0.991 |     1     |                         0.955 |
| per_class_median |                   0.711 |      0.875 |                  0.935 |     0.974 |     0.864 |                         0.914 |
| per_class_min    |                   0.069 |      0.62  |                  0.483 |     0.926 |     0.504 |                         0.845 |
| per_class_p25    |                   0.571 |      0.827 |                  0.842 |     0.964 |     0.808 |                         0.893 |
| per_class_p75    |                   0.829 |      0.922 |                  0.967 |     0.98  |     0.904 |                         0.946 |
| std              |                   0.187 |      0.082 |                  0.117 |     0.02  |     0.096 |                         0.042 |
| test_mean        |                   0.682 |      0.862 |                  0.888 |     0.969 |     0.84  |                         0.911 |
| train_mean       |                   0.838 |      0.926 |                  0.996 |     0.984 |     0.835 |                         0.91  |


### Metrics as columns

#### Model: vit_base_patch16_224.augreg_in21k

| dataset                     |   S_E |   S_R |   bar_S |   base_difficulty |   cv_full |   cv_trimmed |   difficulty | difficulty_sim   |    gap |   gini |   hat_S |   lambda_s |   num_classes |   per_class_max |   per_class_median |   per_class_min |   per_class_p25 |   per_class_p75 |   std |   test_mean |   train_mean |
|:----------------------------|------:|------:|--------:|------------------:|----------:|-------------:|-------------:|:-----------------|-------:|-------:|--------:|-----------:|--------------:|----------------:|-------------------:|----------------:|----------------:|----------------:|------:|------------:|-------------:|
| dpdl-benchmark/sun397       | 0.124 | 0.499 |   0.594 |             0.222 |     0.219 |        0.173 |        0.376 | **4.032**        |  0.135 |  0.995 |   0.752 |        0.5 |           397 |           1     |              0.82  |           0.148 |           0.69  |           0.913 | 0.17  |       0.778 |        0.912 |
| food101                     | 0.191 | 0.45  |   0.565 |             0.084 |     0.072 |        0.048 |        0.113 | **2.352**        |  0.01  |  0.99  |   0.575 |        0.5 |           101 |           1     |              0.936 |           0.584 |           0.896 |           0.952 | 0.066 |       0.916 |        0.927 |
| cifar100                    | 0.257 | 0.588 |   0.583 |             0.067 |     0.062 |        0.046 |        0.108 | **2.289**        |  0.035 |  0.99  |   0.563 |        0.5 |           100 |           1     |              0.95  |           0.68  |           0.91  |           0.97  | 0.058 |       0.933 |        0.968 |
| dpdl-benchmark/gtsrb        | 0.277 | 0.571 |   0.574 |             0.161 |     0.252 |        0.172 |        0.323 | **2.062**        |  0.154 |  0.964 |   0.515 |        0.5 |            43 |           1     |              0.921 |           0.033 |           0.791 |           0.967 | 0.212 |       0.839 |        0.993 |
| cifar10                     | 0.346 | 0.63  |   0.571 |             0.011 |     0.011 |        0.011 |        0.02  | **1.818**        |  0.007 |  0.9   |   0.45  |        0.5 |            10 |           0.998 |              0.994 |           0.969 |           0.985 |           0.996 | 0.011 |       0.989 |        0.996 |
| dpdl-benchmark/svhn_cropped | 0.317 | 0.48  |   0.541 |             0.062 |     0.03  |        0.03  |        0.072 | **1.512**        | -0.01  |  0.885 |   0.339 |        0.5 |            10 |           0.964 |              0.947 |           0.891 |           0.914 |           0.961 | 0.028 |       0.938 |        0.928 |


#### Model: vit_base_patch16_clip_224.metaclip_2pt5b

| dataset                     |   S_E |   S_R |   bar_S |   base_difficulty |   cv_full |   cv_trimmed |   difficulty | difficulty_sim   |   gap |   gini |   hat_S |   lambda_s |   num_classes |   per_class_max |   per_class_median |   per_class_min |   per_class_p25 |   per_class_p75 |   std |   test_mean |   train_mean |
|:----------------------------|------:|------:|--------:|------------------:|----------:|-------------:|-------------:|:-----------------|------:|-------:|--------:|-----------:|--------------:|----------------:|-------------------:|----------------:|----------------:|----------------:|------:|------------:|-------------:|
| dpdl-benchmark/sun397       | 0.11  | 0.518 |   0.602 |             0.207 |     0.201 |        0.154 |        0.373 | **4.699**        | 0.178 |  0.995 |   0.787 |        0.5 |           397 |           1     |              0.821 |           0.074 |           0.711 |           0.914 | 0.159 |       0.793 |        0.971 |
| cifar100                    | 0.175 | 0.552 |   0.594 |             0.086 |     0.071 |        0.058 |        0.146 | **3.154**        | 0.061 |  0.99  |   0.683 |        0.5 |           100 |           1     |              0.94  |           0.73  |           0.88  |           0.96  | 0.065 |       0.914 |        0.975 |
| dpdl-benchmark/gtsrb        | 0.245 | 0.759 |   0.628 |             0.025 |     0.033 |        0.023 |        0.049 | **3.100**        | 0.025 |  0.964 |   0.677 |        0.5 |            43 |           1     |              0.983 |           0.85  |           0.971 |           0.998 | 0.032 |       0.975 |        1     |
| food101                     | 0.173 | 0.495 |   0.58  |             0.075 |     0.061 |        0.042 |        0.111 | **2.861**        | 0.03  |  0.99  |   0.65  |        0.5 |           101 |           1     |              0.936 |           0.624 |           0.896 |           0.96  | 0.057 |       0.925 |        0.955 |
| cifar10                     | 0.277 | 0.634 |   0.589 |             0.012 |     0.011 |        0.011 |        0.024 | **2.293**        | 0.012 |  0.9   |   0.564 |        0.5 |            10 |           0.997 |              0.991 |           0.963 |           0.986 |           0.995 | 0.01  |       0.988 |        1     |
| dpdl-benchmark/svhn_cropped | 0.356 | 0.731 |   0.594 |             0.041 |     0.02  |        0.02  |        0.051 | **2.055**        | 0     |  0.885 |   0.513 |        0.5 |            10 |           0.976 |              0.965 |           0.916 |           0.958 |           0.971 | 0.019 |       0.959 |        0.96  |


#### Model: vit_small_patch16_224.augreg_in21k

| dataset                     |   S_E |   S_R |   bar_S |   base_difficulty |   cv_full |   cv_trimmed |   difficulty | difficulty_sim   |    gap |   gini |   hat_S |   lambda_s |   num_classes |   per_class_max |   per_class_median |   per_class_min |   per_class_p25 |   per_class_p75 |   std |   test_mean |   train_mean |
|:----------------------------|------:|------:|--------:|------------------:|----------:|-------------:|-------------:|:-----------------|-------:|-------:|--------:|-----------:|--------------:|----------------:|-------------------:|----------------:|----------------:|----------------:|------:|------------:|-------------:|
| dpdl-benchmark/sun397       | 0.115 | 0.564 |   0.612 |             0.253 |     0.25  |        0.195 |        0.381 | **4.922**        |  0.059 |  0.995 |   0.797 |        0.5 |           397 |           1     |              0.805 |           0.037 |           0.643 |           0.881 | 0.187 |       0.747 |        0.806 |
| cifar100                    | 0.195 | 0.609 |   0.603 |             0.096 |     0.077 |        0.058 |        0.136 | **3.125**        |  0.023 |  0.99  |   0.68  |        0.5 |           100 |           1     |              0.92  |           0.66  |           0.88  |           0.95  | 0.07  |       0.904 |        0.927 |
| dpdl-benchmark/gtsrb        | 0.21  | 0.653 |   0.611 |             0.13  |     0.22  |        0.112 |        0.238 | **3.105**        |  0.106 |  0.964 |   0.678 |        0.5 |            43 |           1     |              0.947 |           0.083 |           0.829 |           0.98  | 0.191 |       0.87  |        0.976 |
| food101                     | 0.166 | 0.461 |   0.574 |             0.112 |     0.091 |        0.06  |        0.134 | **2.772**        | -0.017 |  0.99  |   0.639 |        0.5 |           101 |           1     |              0.912 |           0.528 |           0.856 |           0.944 | 0.081 |       0.888 |        0.87  |
| cifar10                     | 0.305 | 0.668 |   0.591 |             0.015 |     0.012 |        0.012 |        0.024 | **2.192**        |  0.005 |  0.9   |   0.544 |        0.5 |            10 |           0.995 |              0.988 |           0.958 |           0.986 |           0.992 | 0.012 |       0.985 |        0.99  |
| dpdl-benchmark/svhn_cropped | 0.288 | 0.506 |   0.554 |             0.08  |     0.041 |        0.041 |        0.093 | **1.753**        | -0.013 |  0.885 |   0.43  |        0.5 |            10 |           0.96  |              0.927 |           0.85  |           0.9   |           0.952 | 0.038 |       0.92  |        0.907 |


#### Model: vit_tiny_patch16_224.augreg_in21k

| dataset                     |   S_E |   S_R |   bar_S |   base_difficulty |   cv_full |   cv_trimmed |   difficulty | difficulty_sim   |    gap |   gini |   hat_S |   lambda_s |   num_classes |   per_class_max |   per_class_median |   per_class_min |   per_class_p25 |   per_class_p75 |   std |   test_mean |   train_mean |
|:----------------------------|------:|------:|--------:|------------------:|----------:|-------------:|-------------:|:-----------------|-------:|-------:|--------:|-----------:|--------------:|----------------:|-------------------:|----------------:|----------------:|----------------:|------:|------------:|-------------:|
| dpdl-benchmark/sun397       | 0.136 | 0.557 |   0.605 |             0.318 |     0.275 |        0.22  |        0.506 | **4.084**        |  0.156 |  0.995 |   0.755 |        0.5 |           397 |           0.991 |              0.711 |           0.069 |           0.571 |           0.829 | 0.187 |       0.682 |        0.838 |
| cifar100                    | 0.154 | 0.533 |   0.595 |             0.138 |     0.095 |        0.077 |        0.208 | **3.456**        |  0.063 |  0.99  |   0.711 |        0.5 |           100 |           1     |              0.875 |           0.62  |           0.827 |           0.922 | 0.082 |       0.862 |        0.926 |
| dpdl-benchmark/gtsrb        | 0.188 | 0.615 |   0.607 |             0.112 |     0.131 |        0.098 |        0.215 | **3.276**        |  0.108 |  0.964 |   0.695 |        0.5 |            43 |           1     |              0.935 |           0.483 |           0.842 |           0.967 | 0.117 |       0.888 |        0.996 |
| cifar10                     | 0.268 | 0.619 |   0.588 |             0.031 |     0.021 |        0.021 |        0.049 | **2.308**        |  0.015 |  0.9   |   0.567 |        0.5 |            10 |           0.991 |              0.974 |           0.926 |           0.964 |           0.98  | 0.02  |       0.969 |        0.984 |
| food101                     | 0.265 | 0.507 |   0.561 |             0.16  |     0.115 |        0.086 |        0.202 | **1.915**        | -0.004 |  0.99  |   0.478 |        0.5 |           101 |           1     |              0.864 |           0.504 |           0.808 |           0.904 | 0.096 |       0.84  |        0.835 |
| dpdl-benchmark/svhn_cropped | 0.353 | 0.595 |   0.56  |             0.089 |     0.046 |        0.046 |        0.112 | **1.683**        | -0     |  0.885 |   0.406 |        0.5 |            10 |           0.955 |              0.914 |           0.845 |           0.893 |           0.946 | 0.042 |       0.911 |        0.91  |


## Models

- **Vision Transformer (vit_base_patch16_224.augreg_in21k)**
- **Vision Transformer (vit_small_patch16_224.augreg_in21k)**

Although we are comparing against training all parameters, for the sake of training time, we will use FiLM parameterization.

## Datasets

We will conduct the experiment on the following datasets:

- **CIFAR-10** (100% subset)
- **CIFAR-100** (100% subset)
- **Food-101** (100% subset)
- **dpdl-benchmark/svhn_cropped** (100% subset)
- **dpdl-benchmark/gtsrb** (100% subset)

Additionally, we will also perform a run on the full SUN397 dataset, as this has been the most interesting dataset

- **dpdl-benchmark/sun397** (100% subset)

## Epsilon Values

We will run the experiment over ε = { 1, 3, 8 }.

