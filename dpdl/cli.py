import logging
import torch
import typer

from typing import Optional, List
from typing_extensions import Annotated

from .configurationmanager import ConfigurationManager
from .hyperparameteroptimizer import HyperparameterOptimizer
from .trainer import TrainerFactory
from .utils import seed_everything

log = logging.getLogger(__name__)

def cli(
        ctx: typer.Context,
        command: Annotated[
            str,
            typer.Argument(
                help='Command to run ("train" or "optimize")',
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
        physical_batch_size: Annotated[
            Optional[int],
            typer.Option(
                help='Largest size batch that fits in GPU memory',
                rich_help_panel='Training options',
            )
        ] = 40,
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
                help='Maximum gradient norm (for clipping)',
                rich_help_panel='Opacus options',
            )
        ] = 1.0,
        clipping_mode: Annotated[
            Optional[str],
            typer.Option(
                help='Opacus clipping mode ("flat" or "per_layer" or "adaptive")',
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
        modulevalidator_fix: Annotated[
            Optional[bool],
            typer.Option(
                help="Use ModuleValidator.fix() from Opacus to fix incompatible layers in the model (use with caution)",
                rich_help_panel='Opacus options',
            )
        ] = False,
        accountant: Annotated[
            Optional[str],
            typer.Option(
                help='Privacy accountant',
                rich_help_panel='Opacus options',
            )
        ] = 'prv',
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
        optuna_target_metric: Annotated[
            Optional[str],
            typer.Option(
                help='Target metric for Bayesian optimization',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'loss',
        optuna_direction: Annotated[
            Optional[str],
            typer.Option(
                help='Direction for Bayesian optimization ("minimize" or "maximize")',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'minimize',
        optuna_config: Annotated[
            Optional[str],
            typer.Option(
                help='Configuration file containing ranges/options for hypers',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'conf/optuna_hypers.conf',
        optuna_journal: Annotated[
            Optional[str],
            typer.Option(
                help='Optuna journal (logging) file path',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'optuna-journal.log',
        optuna_resume: Annotated[
            Optional[bool],
            typer.Option(
                help='Resume previous Optuna study',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = False,
    ):

    configurationmanager = ConfigurationManager(ctx.params)

    if configurationmanager.get_command() == 'train':
        if torch.distributed.get_rank() == 0:
            log.info('Starting training.')

        configurationmanager.print_configuration()
        configurationmanager.print_hyperparams()

        seed_everything(configurationmanager.get_value('seed'))
        trainer = TrainerFactory.get_trainer(configurationmanager)

        trainer.fit()

    if configurationmanager.get_command() == 'optimize':
        if torch.distributed.get_rank() == 0:
            log.info('Starting hyperparameter optimization.')

        configurationmanager.print_configuration()

        seed_everything(configurationmanager.get_value('seed'))
        HyperparameterOptimizer.optimize_hypers(configurationmanager)

