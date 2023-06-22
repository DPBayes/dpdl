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

import lightning as L

from dpdl.callbacks import PrintStateCallback
from dpdl.datamodules import CIFAR10DataModule
from dpdl.models import ImageClassificationModel
from dpdl.trainer import Trainer, DifferentiallyPrivateTrainer

def main(
        ctx: typer.Context,
        command: Annotated[
            str,
            typer.Argument(help='command to run in (currently only "train" available)')
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
                help='Leraning rate',
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
        optimize: Annotated[
            Optional[bool],
            typer.Option(
                help='Run Bayesian optimization',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = False,
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

    configuration = ctx.params
    configuration['devices'] = devices if devices else 'auto'
    hyperparam_names = [
        'batch_size',
        'learning_rate',
        'epochs',
        'clipping',
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

    if not optimize:
        train(configuration, hyperparams)
    else:
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

    study = optuna.create_study()

    objective = partial(optuna_objective, configuration, hyperparams, optuna_config, target_hypers)
    study.optimize(objective, n_trials=configuration['n_trials'])

def optuna_objective(configuration, hyperparams, optuna_config, target_hypers, trial):
    print(f'HYPERPARAMS BEFORE: {hyperparams}')
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

    print(f'HYPERPARAMS AFTER: {hyperparams}')
    train(configuration, hyperparams)

def train(configuration, hyperparams):
    def get_tensorboard_logger(log_dir, hyperparams):
        logger = L.pytorch.loggers.TensorBoardLogger(f'{log_dir}/tensorboard')
        logger.log_hyperparams({
            'hyperparams': hyperparams,
            'configuration': configuration,
        })
        return logger

    # setup data, model, and optimizer
    datamodule = CIFAR10DataModule(
        num_workers=configuration['num_workers'],
        batch_size=hyperparams['batch_size'],
    )

    model = ImageClassificationModel(hyperparams['model_name'], configuration['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    callbacks = [
        PrintStateCallback(),
    ]

    loggers = [
        get_tensorboard_logger(configuration['log_dir'], hyperparams),
    ]

    # are we differentially private?
    if configuration['privacy']:

        # if we have target epsilon, set target delta = 1/N
        if 'target_epsilon' in hyperparams:
            target_delta = 1 / len(datamodule.train_dataloader.dataset)
            target_epsilon = hyperparams['target_epsilon']
        else:
            target_delta = None
            target_epsilon = None

        trainer = DifferentiallyPrivateTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            # hypers
            epochs=hyperparams['epochs'],
            accountant=configuration['accountant'],
            noise_multiplier=hyperparams['noise_multiplier'],
            max_grad_norm=hyperparams['max_grad_norm'],
            clipping=hyperparams['clipping'],
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            # config
            accelerator=configuration['accelerator'],
            strategy=configuration['strategy'],
            devices=configuration['devices'],
            precision=configuration['precision'],
            secure_mode=configuration['secure_mode'],
            # callbacks and logging
            callbacks=callbacks,
            loggers=loggers,
        )
    else:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            epochs=configuration['epochs'],
            accelerator=configuration['accelerator'],
            strategy=configuration['strategy'],
            devices=configuration['devices'],
            precision=configuration['precision'],
            callbacks=callbacks,
            loggers=loggers,
        )

    trainer.fit()

    print('Done.')

if __name__ == '__main__':
    typer.run(main)

