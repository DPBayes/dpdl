import logging
import pathlib
import torch
import typer

from pydantic import BaseModel
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
            ('Batch size', self.batch_size),
            ('Learning rate', self.learning_rate),
        ]

        if self.privacy:
            privacy_hypers = [
                ('Noise multiplier', self.noise_multiplier),
                ('Max grad norn', self.max_grad_norm),
                ('Target epsilon', self.target_epsilon),
            ]
            hypers.extend(privacy_hypers)

        max_key_length = max(len(hyper[0]) for hyper in hypers)
        hyper_str = [f'{hyper[0]:<{max_key_length}}: {hyper[1]}' for hyper in hypers]

        return 'Hyperparameters:\n  ' + '\n  '.join(hyper_str) + '\n'

class Configuration(BaseModel):
    command: Literal['train', 'optimize', 'show-layers']
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
    poisson_sampling: bool = True
    n_trials: int = 20
    target_hypers: List[str] = []
    optuna_target_metric: str = 'loss'
    optuna_direction: Literal['minimize', 'maximize'] = 'minimize'
    optuna_config: str = 'conf/optuna_hypers.conf'
    optuna_journal: str = 'optuna-journal.log'
    optuna_resume: bool = False
    optuna_sampler: str = 'TPESampler'
    subset_size: Optional[float]
    num_classes: Optional[int]
    zero_head: bool = False
    lora: bool = False

    def __str__(self):
        attributes = [
            ('Command', self.command),
            ('Privacy', self.privacy),
            ('Model name', self.model_name),
            ('Dataset name', self.dataset_name),
            ('Physical batch size', self.physical_batch_size),
            ('Num workers', self.num_workers),
            ('Validation frequency', self.validation_frequency),
            ('Seed', self.seed),
            ('Log dir', self.log_dir),
            ('Experiment dame', self.experiment_name),
            ('Overwrite experiment', self.overwrite_experiment),
            ('Subset size', self.subset_size),
            ('Num classes', self.num_classes),
            ('Use LoRA', self.lora),
            ('Zero head weights', self.zero_head),
        ]

        if self.command == 'optimize':
            privacy_attributes = [
                ('Clipping mode', self.clipping_mode),
                ('Secure mode', self.secure_mode),
                ('Modulevalidator fix', self.modulevalidator_fix),
                ('Accountant', self.accountant),
                ('Poisson sampling', self.poisson_sampling),
            ]
            attributes.extend(privacy_attributes)

        if self.command == 'optimize':
            optuna_attributes = [
                ('N trials', self.n_trials),
                ('Target hypers', ', '.join(self.target_hypers)),
                ('Optuna target metric', self.optuna_target_metric),
                ('Optuna direction', self.optuna_direction),
                ('Optuna config', self.optuna_config),
                ('Optuna journal', self.optuna_journal),
                ('Optuna resume', self.optuna_resume),
            ]
            attributes.extend(optuna_attributes)

        max_key_length = max(len(attr[0]) for attr in attributes)
        attribute_str = [f'{attr[0]:<{max_key_length}}: {attr[1]}' for attr in attributes]

        return 'Configuration:\n  ' + '\n  '.join(attribute_str) + '\n'

class ConfigurationManager:
    def __init__(self, cli_params: dict):
        self.command = cli_params['command']
        self._check_command()

        self.configuration = Configuration(**cli_params)
        self.hyperparams = Hyperparameters(**cli_params)

        # Opacus calculates noise multiplier is target epsilon is given
        if self.hyperparams.target_epsilon is not None:
            if torch.distributed.get_rank() == 0:
                log.info('We have "target_epsilon" defined. Removing "noise_multiplier".')

            self.hyperparams.noise_multiplier = None

        # remove the target hypers from hyperparams as they will be set in trials
        for target_hyper in self.configuration.target_hypers:
            setattr(self.hyperparams, target_hyper, None)

    def get_command(self):
        return self.command

    def _check_command(self):
        if self.command not in ['train', 'optimize', 'show-layers']:
            raise typer.BadParameter('Command must be "train", "optimize", or "show-layers".')

    def save_configuration(self, directory: pathlib.Path):
        if torch.distributed.get_rank() == 0:
            with open(directory / 'configuration.txt', 'w') as fh:
                fh.write(str(self.configuration))

            with open(directory / 'configuration.json', 'w') as fh:
                fh.write(self.configuration.json())

            log.info(f'Configuration saved to {directory}.')

    def save_hyperparameters(self, directory: pathlib.Path):
        if torch.distributed.get_rank() == 0:
            with open(directory / 'hyperparameters.txt', 'w') as fh:
                fh.write(str(self.hyperparams))

            with open(directory / 'hyperparameters.json', 'w') as fh:
                fh.write(self.hyperparams.json())

            log.info(f'Hyperparameters saved to {directory}/.')
