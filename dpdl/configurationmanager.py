import logging
import pathlib
import torch
import typer

from pydantic import BaseModel, root_validator
from typing import Optional, List, Literal

log = logging.getLogger(__name__)

class Hyperparameters(BaseModel):
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    noise_multiplier: Optional[float]
    max_grad_norm: Optional[float]
    target_epsilon: Optional[float]
    privacy: bool = True # Only used in __str__

    def __str__(self):
        hypers = [
            ('Epochs', self.epochs),
            ('Batch Size', self.batch_size),
            ('Learning Rate', self.learning_rate),
        ]

        if self.privacy:
            privacy_hypers = [
                ('Noise Multiplier', self.noise_multiplier),
                ('Max Grad Norm', self.max_grad_norm),
                ('Target Epsilon', self.target_epsilon),
            ]
            hypers.extend(privacy_hypers)

        max_key_length = max(len(hyper[0]) for hyper in hypers)
        hyper_str = [f'{hyper[0]:<{max_key_length}}: {hyper[1]}' for hyper in hypers]

        return 'Hyperparameters:\n  ' + '\n  '.join(hyper_str)

class Configuration(BaseModel):
    command: Literal['train', 'optimize']
    privacy: bool = True
    model_name: str = 'resnet50'
    dataset_name: str = 'cifar10'
    physical_batch_size: int = 40
    num_workers: int = 8
    validation_frequency: float = 1.0
    seed: int = 0
    log_dir: str = 'logs'
    experiment_name: str = 'default-experiment'
    overwrite_experiment: bool = False
    clipping_mode: str = 'flat'
    secure_mode: bool = False
    modulevalidator_fix: bool = False
    accountant: str = 'prv'
    n_trials: int = 20
    target_hypers: List[str] = []
    optuna_target_metric: str = 'loss'
    optuna_direction: Literal['minimize', 'maximize'] = 'minimize'
    optuna_config: str = 'conf/optuna_hypers.conf'
    optuna_journal: str = 'optuna-journal.log'
    optuna_resume: bool = False
    subset_size: Optional[float]
    num_classes: Optional[int]

    def __str__(self):
        attributes = [
            ('Command', self.command),
            ('Privacy', self.privacy),
            ('Model Name', self.model_name),
            ('Dataset Name', self.dataset_name),
            ('Physical Batch Size', self.physical_batch_size),
            ('Num Workers', self.num_workers),
            ('Validation Frequency', self.validation_frequency),
            ('Seed', self.seed),
            ('Log Dir', self.log_dir),
            ('Experiment Name', self.experiment_name),
            ('Overwrite Experiment', self.overwrite_experiment),
            ('Subset Size', self.subset_size),
            ('Num Classes', self.num_classes),
        ]

        if self.command == 'optimize':
            privacy_attributes = [
                ('Clipping Mode', self.clipping_mode),
                ('Secure Mode', self.secure_mode),
                ('ModuleValidator Fix', self.modulevalidator_fix),
                ('Accountant', self.accountant),
            ]
            attributes.extend(privacy_attributes)

        if self.command == 'optimize':
            optuna_attributes = [
                ('N Trials', self.n_trials),
                ('Target Hypers', ', '.join(self.target_hypers)),
                ('Optuna Target Metric', self.optuna_target_metric),
                ('Optuna Direction', self.optuna_direction),
                ('Optuna Config', self.optuna_config),
                ('Optuna Journal', self.optuna_journal),
                ('Optuna Resume', self.optuna_resume),
            ]
            attributes.extend(optuna_attributes)

        max_key_length = max(len(attr[0]) for attr in attributes)
        attribute_str = [f'{attr[0]:<{max_key_length}}: {attr[1]}' for attr in attributes]

        return 'Configuration:\n  ' + '\n  '.join(attribute_str)

class ConfigurationManager:
    def __init__(self, cli_params: dict):
        self.command = cli_params['command']
        self._check_command()

        self.configuration = Configuration(**cli_params)
        self.hyperparams = Hyperparameters(**cli_params)

        # Opacus calculates noise multiplier is target epsilon is given
        if self.hyperparams.target_epsilon is not None:
            if torch.distributed.get_rank() == 0:
                log.warn('We have "target_epsilon" defined. Removing "noise_multiplier".')

            self.hyperparams.noise_multiplier = None

        # remove the target hypers from hyperparams as they will be set in trials
        for target_hyper in self.configuration.target_hypers:
            setattr(self.hyperparams, target_hyper, None)

    def get_command(self):
        return self.command

    def _check_command(self):
        if self.command not in ['train', 'optimize']:
            raise typer.BadParameter('Command must be "train" or "optimize".')

    def print_configuration(self):
        if torch.distributed.get_rank() == 0:
            log.info(f'Configuration:')
            log.info(f'Command: {self.command}\n')
            log.info(self.configuration.json(indent=4))

    def save_configuration(self, directory: pathlib.Path):
        if torch.distributed.get_rank() == 0:
            with open(directory / 'configuration.txt', 'w') as fh:
                fh.write('Configuration:\n')
                fh.write(str(self.configuration))

            log.info(f'Configuration saved to {directory}/configuration.txt')

    def save_hyperparameters(self, directory: pathlib.Path):
        if torch.distributed.get_rank() == 0:
            with open(directory / 'hyperparameters.txt', 'w') as fh:
                fh.write('Hyperparameters:\n')
                fh.write(str(self.hyperparams))

            log.info(f'Hyperparameters saved to {directory}/hyperparameters.txt')
