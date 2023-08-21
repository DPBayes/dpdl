# Easy experimentation for Differentially Private (DP) Deep Learning

The system requires CUDA. We provide scripts for running in a Slurm environment.

Many of the ideas that we are using come from [fastai](https://github.com/fastai/fastai) and [PyTorch Lightning](https://github.com/Lightning-AI/lightning).

## Install dependencies

`pip install torch opacus timm datasets typer[all] optuna torchmetrics`

## Command line usage

Entry point is [run.py](blob/vanilla-pytorch-refactor/run.py).

### How to use?

Get help with `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:0 run.py`.
![run.py help](images/usage.png)

### Examples

#### Optimize loss for target epsilon 3.t using a pretrained BiT (Big Transfer) ResNet-50

`run.py optimize --num-workers 8 --model-name resnetv2_50x1_bitm_in21k --target-hypers batch_size --target-hypers learning_rate --target-hypers max_grad_norm --target-epsilon 3.1 --epochs 30 --n-trials 20 --seed 42 --physical-batch-size 40 --optuna-config conf/optuna_hypers-bs1024.conf --experiment-name experiment-with-epsilon-3.1`

#### Optimize accuracy for target epsilon 3.t using a pretrained BiT (Big Transfer) ResNet-50

`run.py optimize --num-workers 8 --model-name resnetv2_50x1_bitm_in21k --target-hypers batch_size --target-hypers learning_rate --target-hypers max_grad_norm --target-epsilon 3.1 --epochs 30 --n-trials 20 --seed 42 --physical-batch-size 40 --optuna-config conf/optuna_hypers-bs1024.conf --optuna-target-metric MulticlassAccuracy --optuna-direction maximize --experiment-name experiment-with-epsilon-3.1`

### The same but without DP

`run.py optimize --num-workers 8 --model-name resnetv2_50x1_bitm_in21k --target-hypers batch_size --target-hypers learning_rate --target-hypers max_grad_norm --target-epsilon 3.1 --epochs 30 --n-trials 20 --seed 42 --physical-batch-size 40 --optuna-config conf/optuna_hypers-bs1024.conf --optuna-target-metric MulticlassAccuracy --optuna-direction maximize --no-privacy --experiment-name experiment-with-epsilon-3.1-no-privacy`

## Architecture

![DPDL Architecture](images/dpdl-architecture.cli)

### Entry point

The entrypoint [run.py](run.py) provides a CLI using Python's Typer module.

### Command-line interface

The CLI implementation is in [dpdl/cli.py](dpdl/cli.py)

### Training

The CLI calls the `fit` method of [trainer](dpdl/trainer.py) 

### Callbacks

The system provides a flexible [callback system](dpdl/callbacks.py).

### Hyperparameter optimization

The CLI class the `optimize_hypers` method of [hyperparameteroptimizer](dpdl/hyperparameteroptimizer.py).

## How to?

### Add a new dataset?

Create a new [datamodule](dpdl/datamodules.py).

### Add a new model?

Create a new [model](dpdl/models.py).

### Add a new optimizer?

Add a new optimizer in [optimizers](dpdl/optimizers.py).

