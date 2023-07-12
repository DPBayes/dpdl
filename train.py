#!/usr/bin/env python3
import os
import sys
import logging

import optuna
import torch
import typer
import yaml

from functools import partial
from typing import Optional, List
from typing_extensions import Annotated
from rich import print

import dpdl.utils

from dpdl.callbacks import CallbackHandler, PrintStateCallback
from dpdl.datamodules import CIFAR10DataModule
from dpdl.models import ImageClassificationModel
from dpdl.trainer import Trainer, DifferentiallyPrivateTrainer

log = logging.getLogger(__name__)

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
        seed: Annotated[
            int,
            typer.Option(
                help='Random seed',
                rich_help_panel='Training options',
            )
        ] = 0,
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
        ] = 'default',
        experiment_version: Annotated[
            Optional[str],
            typer.Option(
                help='Experiment version for logging',
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
        physical_batch_size: Annotated[
            Optional[int],
            typer.Option(
                help='Largest size batch that fits in GPU memory',
                rich_help_panel='Opacus options',
            )
        ] = 60,
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
        optuna_config: Annotated[
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
    # reproducible results
    if seed := configuration['seed']:
        dpdl.utils.seed_everything(seed)

    # helper function for reading Optuna hyperparameter configuration that
    # includes types, options, lower and upper bounds, etc for hyperparams
    def read_optuna_config(config_fpath):
        with open(config_fpath, 'rb') as fh:
            config = yaml.safe_load(fh)

        return config

    # the targeted hypers from command line
    target_hypers = configuration['target_hypers']

    if len(target_hypers) == 0:
        raise(typer.BadParameter('Bayesian optimization enabled and no target hyperparameters for optimization.'))

    optuna_config = read_optuna_config(configuration['optuna_config'])

    # check that the target hypers are known and appear in Optuna configuration
    for target_hyper in target_hypers:
        if not target_hyper in hyperparams:
            raise(typer.BadParameter(f'Cannot optimize unknown hyperparameter "{target_hyper}".'))
        if not target_hyper in optuna_config:
            config_fpath = configuration['optuna_config']
            raise(typer.BadParameter(f'Hyperparameter "{target_hyper}" not found in Optuna configuration file "{config_fpath}".'))

    # optuna needs a global process group with 'gloo' backend for communication.
    # "Create a global gloo backend when group is None and WORLD is nccl."
    # - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.TorchDistributedTrial.html
    optuna_process_group = torch.distributed.new_group(
        backend='gloo',
    )

    # the optimization objective
    objective = partial(optuna_objective, configuration, hyperparams, optuna_config, target_hypers, optuna_process_group)

    # start Optuna study only on rank zero
    if torch.distributed.get_rank() == 0:
        print('Starting optimization.')
        print('Configuration:')
        for key, value in configuration.items():
            print(f'- {key:<20}: {value}')

        sampler = optuna.samplers.TPESampler(seed=configuration['seed'])

        study = optuna.create_study(
            storage='sqlite:///optuna.sqlite3',
            study_name=configuration['experiment_name'],
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=configuration['n_trials'],
            gc_after_trial=True, # garbage collect after each trial
        )
    else:
        # "Please set trial object in rank-0 node and set `None` in the other rank node."
        # - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.TorchDistributedTrial.html
        for _ in range(configuration['n_trials']):
            objective(None)

    if torch.distributed.get_rank() == 0:
        trial = study.best_trial
        print(f'Best objective ralue: {trial.value}', trial.value)
        print('Params: ')
        for key, value in trial.params.items():
            print(f' - {key}: {value}')

def optuna_objective(configuration, hyperparams, optuna_config, target_hypers, process_group, trial):
    # make the trial support distributed
    # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
    trial = optuna.integration.TorchDistributedTrial(trial, group=process_group)

    for target_hyper in target_hypers:
        if optuna_config[target_hyper]['type'] == 'float':
            hyper_value = trial.suggest_float(
                target_hyper,
                optuna_config[target_hyper]['min'],
                optuna_config[target_hyper]['max'],
                )

        if optuna_config[target_hyper]['type'] == 'int':
            hyper_value = trial.suggest_int(
                target_hyper,
                optuna_config[target_hyper]['min'],
                optuna_config[target_hyper]['max'],
            )

        if optuna_config[target_hyper]['type'] == 'categorical':
            hyper_value = trial.suggest_int(
                target_hyper,
                optuna_config[target_hyper]['options'],
            )

        hyperparams[target_hyper] = hyper_value

    # while tuning hypers, do not validate
    configuration['validation_frequency'] = 0

    # train the model
    trainer = get_trainer(configuration, hyperparams)

    if torch.distributed.get_rank() == 0:
        print(f'Starting trial {trial.number}.')
        print(f'Hyperparams:')
        for hyper, value in hyperparams.items():
            print(f'- {hyper:<20}: {value}')

    trainer.fit()

    # optimization objective is the validation loss
    objective = trainer.validate()

#    # trainer has reference to the fabric object, let's
#    # make sure the trainer gets garbage collected
#    del trainer
#
#    # we don't need anything in the CUDA cache anymore so, let's clear
#    # it to make sure nothing there interferes with the next run
#    torch.cuda.empty_cache()

    return objective

def get_model(configuration, hyperparams):
    model = ImageClassificationModel(
        model_name=hyperparams['model_name'],
        num_classes=configuration['num_classes']
    )
    return model

def get_datamodule(configuration, hyperparams):
    datamodule = CIFAR10DataModule(
        num_workers=configuration['num_workers'],
        batch_size=hyperparams['batch_size'],
        seed=configuration['seed'],
        image_size=(224, 224),
    )

    return datamodule

def get_optimizer(configuration, hyperparams, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    #optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams['learning_rate'])
    return optimizer

def get_loggers(configuration, hyperparams):
    loggers = []

    log_dir = configuration['log_dir']
    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=f'{log_dir}',
        name=configuration['experiment_name'],
        version=configuration['experiment_version'],
    )
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

def get_basic_trainer(configuration, hyperparams):
    # setup data, model, and optimizer
    datamodule = get_datamodule(configuration, hyperparams)
    model = get_model(configuration, hyperparams)
    optimizer = get_optimizer(configuration, hyperparams, model)
    callback_handler = CallbackHandler(
        get_callbacks(configuration, hyperparams)
    )

    # instantiate a trainer without dp
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        datamodule=datamodule,
        epochs=hyperparams['epochs'],
        seed=configuration['seed'],
        callback_handler=callback_handler,
    )

    return trainer

def get_differentially_private_trainer(configuration, hyperparams):
    # setup data, model, and optimizer
    datamodule = get_datamodule(configuration, hyperparams)
    model = get_model(configuration, hyperparams)
    optimizer = get_optimizer(configuration, hyperparams, model)
    callback_handler = CallbackHandler(
        get_callbacks(configuration, hyperparams)
    )

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
        # hypers
        epochs=hyperparams['epochs'],
        noise_multiplier=hyperparams['noise_multiplier'],
        max_grad_norm=hyperparams['max_grad_norm'],
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        # config
        secure_mode=configuration['secure_mode'],
        clipping=configuration['clipping'],
        physical_batch_size=configuration['physical_batch_size'],
        seed=configuration['seed'],
        callback_handler=callback_handler,
    )

    return trainer

def get_trainer(configuration, hyperparams):
    # are we differentially private?
    if configuration['privacy']:
        return get_differentially_private_trainer(configuration, hyperparams)

    return get_basic_trainer(configuration, hyperparams)

def train(configuration, hyperparams):
    if seed := configuration['seed']:
        dpdl.utils.seed_everything(seed)

    trainer = get_trainer(configuration, hyperparams)

    if torch.distributed.get_rank() == 0:
        print('Starting training.')
        print(f'Configuration:')
        for key, value in configuration.items():
            print(f'- {key:<20}: {value}')

        print(f'Hyperparams:')
        for hyper, value in hyperparams.items():
            print(f'- {hyper:<20}: {value}')

    trainer.fit()

if __name__ == '__main__':
    torch.distributed.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    log.debug('Running application.')
    typer.run(main)
