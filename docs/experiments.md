## Experiment Design

## Batch size variation

### Objective

To investigate the influence of varying batch sizes on the optimal configurations of other hyperparameters (epochs, learning\_rate, max\_grad\_norm) using Bayesian optimization.

Methodology

1. Batch size variation: Systematically vary the batch size through a predefined set of values: 256, 512, 1024, 2048, 4096, 8192, 12288, etc.

2. Bayesian Optimization: For each batch size, use Bayesian optimization to find the optimal values of the other hyperparameters (epochs, learning\_rate, max\_grad\_norm).

Experimentation

Conduct separate optimization runs for each batch size. In each run, the batch size is fixed, and Bayesian optimization is used to optimize the other hyperparameters. Here is a markdown table template that you might use to document the results:

#### CIFAR10

| Batch Size | Optimized Epochs | Optimized Learning Rate | Optimized Max Gradient Norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256   | | | | |
| 512   | | | | |
| 1024  | | | | |
| 2048  | | | | |
| 4096  | | | | |
| 6144  | | | | |
| 8192  | | | | |
| 10240 | | | | |
| 12288 | | | | |
| 14336 | | | | |
| 16384 | | | | |
| 18432 | | | | |
| 20480 | | | | |
| 22528 | | | | |
| 24576 | | | | |
| 26624 | | | | |
| 28672 | | | | |
| 30720 | | | | |
| 32768 | | | | |
| 34816 | | | | |
| 36864 | | | | |
| 38912 | | | | |
| 45000 | | | | |

#### CIFAR100 (10% subset)

| Batch Size | Optimized Epochs | Optimized Learning Rate | Optimized Max Gradient Norm | Accuracy |
|------------|------------------|-------------------------|-----------------------------|----------|
| 256   | | | | |
| 512   | | | | |
| 1024  | | | | |
| 1536  | | | | |
| 2048  | | | | |
| 2560  | | | | |
| 3072  | | | | |
| 3584  | | | | |
| 4096  | | | | |
| 4500  | | | | |

