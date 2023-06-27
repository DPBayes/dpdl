#!/usr/bin/env python3

import os
import warnings

import optuna
import torch
import typer
import yaml

from functools import partial
from typing import Optional, List
from typing_extensions import Annotated
from rich import print

import lightning as L

from dpdl.callbacks import PrintStateCallback
from dpdl.datamodules import CIFAR10DataModule
from dpdl.models import ImageClassificationModel
from dpdl.trainer import Trainer, DifferentiallyPrivateTrainer


def main(
        ctx: typer.Context,
        command: Annotated[
            str,
            typer.Argument(
                help='Command to run (train|optimize)',
            )
        ],
        epochs: Annotated[
            int,
            typer.Option(
                help='Number of epochs to train',
                rich_help_panel='Training options',
            )
        ] = 10,
        learning_rate: Annotated[
            float,
            typer.Option(
                help='Learning rate',
                rich_help_panel='Training options',
            )
        ] = 1e-3,
        batch_size: Annotated[
            int,
            typer.Option(
                help='Batch size',
                rich_help_panel='Training options',
            )
        ] = 256,
        num_workers: Annotated[
            int,
            typer.Option(
                help='Number of workers for data loading (per GPU)',
                rich_help_panel='Training options',
            )
        ] = 8,
        model_name: Annotated[
            str,
            typer.Option(
                help='PyTorch Image Models (timm) model name',
                rich_help_panel='Training options',
            )
        ] = 'resnet50',
        validation_frequency: Annotated[
            float,
            typer.Option(
                help='Validation frequency',
                rich_help_panel='Training options',
            )
        ] = 1.0,
        privacy: Annotated[
            bool,
            typer.Option(
                help='Enable privacy (Opacus)',
                rich_help_panel='Training options',
            )
        ] = True,
        log_dir: Annotated[
            str,
            typer.Option(
                help='Log directory',
                rich_help_panel='Logging options',
            )
        ] = 'logs',
        experiment_name: Annotated[
            Optional[str],
            typer.Option(
                help='Experiment name for logging',
                rich_help_panel='Logging options',
            )
        ] = None,
        num_classes: Annotated[
            Optional[int],
            typer.Option(
                help='Number of classes for a classification model',
                rich_help_panel='Classification model options',
            )
        ] = 10,
        accelerator: Annotated[
            Optional[str],
            typer.Option(
                help='Accelerator to use',
                rich_help_panel='Fabric options',
            )
        ] = 'gpu',
        strategy: Annotated[
            Optional[str],
            typer.Option(
                help='Distributed strategy',
                rich_help_panel='Fabric options',
            )
        ] = 'ddp',
        devices: Annotated[
            Optional[int],
            typer.Option(
                help='Number of devices to use',
                rich_help_panel='Fabric options',
            )
        ] = 0,
        precision: Annotated[
            Optional[int],
            typer.Option(
                help='Training precision',
                rich_help_panel='Fabric options',
            )
        ] = 32,
        noise_multiplier: Annotated[
            Optional[float],
            typer.Option(
                help='Noise multiplier',
                rich_help_panel='Opacus options',
            )
        ] = 1.0,
        max_grad_norm: Annotated[
            Optional[float],
            typer.Option(
                help='Maximum gradient norm (clipping)',
                rich_help_panel='Opacus options',
            )
        ] = 1.0,
        clipping: Annotated[
            Optional[str],
            typer.Option(
                help='Clipping mode (see Opacus)',
                rich_help_panel='Opacus options',
            )
        ] = 'flat',
        secure_mode: Annotated[
            Optional[bool],
            typer.Option(
                help='Enable secure mode for production use',
                rich_help_panel='Opacus options',
            )
        ] = False,
        accountant: Annotated[
            Optional[str],
            typer.Option(
                help='Privacy accountant',
                rich_help_panel='Opacus options',
            )
        ] = 'rdp',
        target_epsilon: Annotated[
            Optional[float],
            typer.Option(
                help='Target epsilon for the privacy accountant (implies delta = 1/N)',
                rich_help_panel='Opacus options',
            )
        ] = None,
        study_name: Annotated[
            Optional[str],
            typer.Option(
                help='Optuna study name',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'Default study',
        target_hypers: Annotated[
            Optional[List[str]],
            typer.Option(
                help='Hyperparameters to optimize (use multiple times if necessary)',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = [],
        n_trials: Annotated[
            Optional[int],
            typer.Option(
                help='Number of optimization rounds',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 20,
        optuna_config_fpath: Annotated[
            Optional[str],
            typer.Option(
                help='Configuration file containing ranges/options for hypers',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'conf/optuna_hypers.conf',
    ):

    if command not in ['train', 'optimize']:
        raise(typer.BadParameter('Command must be "train" or "optimize".'))

    configuration = ctx.params
    configuration['devices'] = devices if devices else 'auto'
    hyperparam_names = [
        'batch_size',
        'learning_rate',
        'epochs',
        'model_name',
    ]

    if privacy:
        hyperparam_names.extend([
            'noise_multiplier',
            'max_grad_norm',
        ])

    hyperparams = {}
    for name in hyperparam_names:
        hyperparams[name] = configuration.pop(name)

    if command == 'train':
        train(configuration, hyperparams)

    if command == 'optimize':
        optimize_hypers(configuration, hyperparams)

def optimize_hypers(configuration, hyperparams):
    def read_optuna_config(config_fpath):
        with open(config_fpath, 'rb') as fh:
            config = yaml.safe_load(fh)

        return config

    target_hypers = configuration['target_hypers']

    if len(target_hypers) == 0:
        raise(typer.BadParameter('Bayesian optimization enabled and no target hyperparameters for optimization.'))

    optuna_config = read_optuna_config(configuration['optuna_config_fpath'])

    for target_hyper in target_hypers:
        if not target_hyper in hyperparams:
            raise(typer.BadParameter(f'Cannot optimize unknown hyperparameter "{target_hyper}".'))
        if not target_hyper in optuna_config:
            config_fpath = configuration['optuna_config_fpath']
            raise(typer.BadParameter(f'Hyperparameter "{target_hyper}" not found in Optuna configuration file "{config_fpath}".'))

    # instantiate fabric
    fabric = get_fabric(configuration, hyperparams)

    # the optimization objective
    objective = partial(optuna_objective, fabric, configuration, hyperparams, optuna_config, target_hypers)

    # we want sequential studies. only run a study if we are
    # the global rank zero process
    if fabric.is_global_zero:
        study = optuna.create_study(
            storage='sqlite:///optuna.sqlite3',
            study_name=configuration['study_name'],
        )
        study.optimize(
            objective,
            n_trials=configuration['n_trials'],
            gc_after_trial=True, # garbage collect after each trial
        )
    else:
        # not sure why need to call the objective with None here.
        # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
        for _ in range(configuration['n_trials']):
            objective(None)

    if fabric.is_global_zero:
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        trial = study.best_trial
        print(f'Best objective ralue: {trial.value}', trial.value)
        print('Params: ')
        for key, value in trial.params.items():
            print(f' - {key}: {value}')

def optuna_objective(fabric, configuration, hyperparams, optuna_config, target_hypers, trial):
    # make the trial support distributed
    # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
    trial = optuna.integration.TorchDistributedTrial(trial)

    for target_hyper in target_hypers:
        if optuna_config[target_hyper]['type'] == 'float':
            hyper_value = trial.suggest_float(
                target_hyper,
                optuna_config[target_hyper]['min'],
                optuna_config[target_hyper]['max'],
                log=True,
            )

        if optuna_config[target_hyper]['type'] == 'int':
            hyper_value = trial.suggest_int(
                target_hyper,
                optuna_config[target_hyper]['min'],
                optuna_config[target_hyper]['max'],
                log=True,
            )

        if optuna_config[target_hyper]['type'] == 'categorical':
            hyper_value = trial.suggest_int(
                target_hyper,
                optuna_config[target_hyper]['options'],
                log=True,
            )

        hyperparams[target_hyper] = hyper_value

    # while tuning hypers, do not validate
    configuration['validation_frequency'] = 0

    # train the model
    trainer = get_trainer(fabric, configuration, hyperparams)
    trainer.fit()

    # optimization objective is the validation loss
    objective = trainer.validate()

    # trainer has reference to the fabric object, let's
    # make sure the trainer gets garbage collected
    del trainer

    return objective

def get_model(configuration, hyperparams):
    model = ImageClassificationModel(hyperparams['model_name'], configuration['num_classes'])
    return model

def get_datamodule(configuration, hyperparams):
    datamodule = CIFAR10DataModule(
        num_workers=configuration['num_workers'],
        batch_size=hyperparams['batch_size'],
        image_size=(224, 224),
    )

    return datamodule

def get_optimizer(configuration, hyperparams, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    return optimizer

def get_fabric(configuration, hyperparams):
    loggers = get_loggers(configuration, hyperparams)
    callbacks = get_callbacks(configuration, hyperparams)

    fabric = L.Fabric(
        accelerator=configuration['accelerator'],
        strategy=configuration['strategy'],
        devices=configuration['devices'],
        precision=configuration['precision'],
        callbacks=callbacks,
        loggers=loggers,
    )
    fabric.launch()

    return fabric

def get_loggers(configuration, hyperparams):
    loggers = []

    log_dir = configuration['log_dir']
    logger = L.pytorch.loggers.TensorBoardLogger(f'{log_dir}/tensorboard')
    logger.log_hyperparams({
        'hyperparams': hyperparams,
        'configuration': configuration,
    })

    loggers.append(logger)

    return loggers

def get_callbacks(configuration, hyperparams):
    callbacks = [
        PrintStateCallback(),
    ]

    return callbacks

def get_basic_trainer(fabric, configuration, hyperparams):
    # setup data, model, and optimizer
    datamodule = get_datamodule(configuration, hyperparams)
    model = get_model(configuration, hyperparams)
    optimizer = get_optimizer(configuration, hyperparams, model)

    # instantiate a trainer without dp
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        datamodule=datamodule,
        fabric=fabric,
        epochs=hyperparams['epochs'],
    )

    return trainer

def get_differentially_private_trainer(fabric, configuration, hyperparams):
    # setup data, model, and optimizer
    datamodule = get_datamodule(configuration, hyperparams)
    model = get_model(configuration, hyperparams)
    optimizer = get_optimizer(configuration, hyperparams, model)

    # are we given a target epsilon?
    if 'target_epsilon' in hyperparams:
        # if we have target epsilon, set target delta = 1/N
        target_delta = 1 / len(datamodule.train_dataloader.dataset)
        target_epsilon = hyperparams['target_epsilon']

        # if target epsilon is given, then opacus will calculate
        # the noise multiplier for us
        hyperparams['noise_multiplier'] = None
    else:
        target_delta = None
        target_epsilon = None

    # instantiate a differentialy private trained
    trainer = DifferentiallyPrivateTrainer(
        model=model,
        optimizer=optimizer,
        datamodule=datamodule,
        fabric=fabric,
        # hypers
        epochs=hyperparams['epochs'],
        noise_multiplier=hyperparams['noise_multiplier'],
        max_grad_norm=hyperparams['max_grad_norm'],
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        # config
        secure_mode=configuration['secure_mode'],
        clipping=configuration['clipping'],
    )

    return trainer

def get_trainer(fabric, configuration, hyperparams):
    # are we differentially private?
    if configuration['privacy']:
        return get_differentially_private_trainer(fabric, configuration, hyperparams)

    return get_basic_trainer(fabric, configuration, hyperparams)

def train(configuration, hyperparams):
    # instantiate fabric
    fabric = get_fabric(configuration, hyperparams)

    trainer = get_trainer(fabric, configuration, hyperparams)
    trainer.fit()

if __name__ == '__main__':
    typer.run(main)

