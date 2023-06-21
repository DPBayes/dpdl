epochs#!/usr/bin/env python3

import os
import warnings

import torch
import typer

from typing import Optional
from typing_extensions import Annotated

import lightning as L

from callbacks import PrintStateCallback
from datamodules import CIFAR10DataModule
from models import ImageClassificationModel
from trainer import Trainer, DifferentiallyPrivateTrainer

def get_cli_configuration(
        ctx: typer.Context,
        epochs: Annotated[
            int,
            typer.Option(
                help='Number of epochs to train',
                rich_help_panel='Training options',
            )
        ] = 10,
        lr: Annotated[
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
        target_delta: Annotated[
            Optional[float],
            typer.Option(
                help='Target delta for the privacy accountant (requires target_epsilon)',
                rich_help_panel='Opacus options',
            )
        ] = None,
        target_epsilon: Annotated[
            Optional[float],
            typer.Option(
                help='Target delta for the privacy accountant (requires target_delta)',
                rich_help_panel='Opacus options',
            )
        ] = None,
    ):

    if not all([target_epsilon, target_delta]):
        raise typer.BadParameter('Both arguments "target_epsilon" and "target_delta" are required.')

    configuration = ctx.params
    configuration['devices'] = devices if devices else 'auto'

    hyperparam_names = [
        'model_name',
        'batch_size',
        'lr',
    ]

    if privacy:
        hyperparam_names.extend([
            'accountant',
            'noise_multiplier'
            'max_grad_norm',
            'clipping',
            'target_delta',
            'target_epsilon',
        ])

    hyperparams = {}
    for name in hyperparam_names:
        hyperparams[name] = configuration.pop(name)

    train(configuration, hyperparams)

def train(configuration, hyperparams):
    def get_tensorboard_logger(log_dir, hyperparams):
        logger = L.pytorch.loggers.TensorBoardLogger(f'{log_dir}/tensorboard'),
        logger.log_hyperparams(hyperparams)
        return logger

    # setup data, model, and optimizer
    datamodule = CIFAR10DataModule(
        num_workers=configuration['num_workers'],
        batch_size=hyperparams['batch_size'],
    )
    model = ImageClassificationModel(hyperparams['model_name'], configuration['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])

    callbacks = [
        PrintStateCallback(),
    ]

    log_dir = configuration['log_dir']
    loggers = [
        get_tensorboard_logger(configuration['log_dir'], hyperparams),
    ]

    if configuration['privacy']:
        trainer = DifferentiallyPrivateTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            # hypers
            epochs=hyperparams['epochs'],
            accountant=configuration['accountant'],
            noise_multiplier=hyperparams['noise_multiplier'],
            max_grad_norm=hyperparams['max_grad_norm'],
            target_epsilon=hyperparams['target_epsilon'],
            target_delta=hyperparams['target_delta'],
            clipping=hyperparams['clipping'],
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
    typer.run(get_cli_configuration)

