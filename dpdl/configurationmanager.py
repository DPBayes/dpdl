import logging

import torch
import typer

log = logging.getLogger(__name__)

class ConfigurationManager():
    def __init__(self, cli_params: dict):
        self.command = cli_params['command']
        self._check_command()

        self.configuration = cli_params
        self.hyperparams = {}

        self.hyperparam_names = [
            'batch_size',
            'learning_rate',
            'epochs',
            'model_name',
        ]

        if cli_params['privacy']:
            self.hyperparam_names.extend([
                'noise_multiplier',
                'max_grad_norm',
            ])

        self._get_hypers_from_params()

    def get_command(self):
        return self.command

    def get_value(self, key):
        if not key in self.configuration:
            raise(RuntimeError(f'{__name__} - Unknown configuration key "{key}".'))

        return self.configuration[key]

    def set_value(self, key, value):
        self.configuration[key] = value

    def set_hyper(self, key, value):
        self.hyperparams[key] = value

    def get_hyper(self, key):
        if not key in self.hyperparams:
            raise(RuntimeError(f'{__name__} - Unknown hyperparameter "{key}".'))

        return self.hyperparams[key]

    def get_configuration(self):
        return self.configuration

    def get_hyperparams(self):
        return self.hyperparams

    def _get_hypers_from_params(self):
        for name in self.hyperparam_names:
            self.hyperparams[name] = self.configuration.pop(name)

    def _check_command(self):
        if self.command not in ['train', 'optimize']:
            raise(typer.BadParameter('Command must be "train" or "optimize".'))

    def print_configuration(self):
        if torch.distributed.get_rank() == 0:
            log.info(f'Configuration:')
            self._print_dict(self.configuration)

    def print_hyperparams(self):
        self.print_hyperparameters()

    def print_hyperparameters(self):
        if torch.distributed.get_rank() == 0:
            log.info(f'Hyperparams:')
            self._print_dict(self.hyperparams)

    @staticmethod
    def _print_dict(d: dict) -> None:
        if torch.distributed.get_rank() == 0:
            for key, value in d.items():
                log.info(f'{key:<20}: {value}')

