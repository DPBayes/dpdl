import logging
import pathlib
import torch
import typer

log = logging.getLogger(__name__)

class ConfigurationManager:
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
                'target_epsilon',
            ])

        self._get_hypers_from_params()

        # Opacus calculates noise multiplier is target epsilon is given
        if 'target_epsilon' in self.hyperparams:
            log.warn('We have "target_epsilon" defined. Removing "noise_multiplier".')
            self.hyperparams['noise_multiplier'] = None

        # remove the target hypers from hyperparams as they will be set in trials
        for target_hyper in self.get_value('target_hypers'):
            self.hyperparams[target_hyper] = None

    def get_command(self):
        return self.command

    def get_value(self, key):
        if not key in self.configuration:
            raise RuntimeError(f'{__name__} - Unknown configuration key "{key}".')

        return self.configuration[key]

    def set_value(self, key, value):
        self.configuration[key] = value

    def set_hyper(self, key, value):
        self.hyperparams[key] = value

    def get_hyper(self, key):
        if not key in self.hyperparams:
            raise RuntimeError(f'{__name__} - Unknown hyperparameter "{key}".')

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
            raise typer.BadParameter('Command must be "train" or "optimize".')

    def _check_optuna_direction(self):
        if self.optuna_direction not in ['minimize', 'maximize']:
            raise typer.BadParameter('Optuna direction must be "minimize" or "maximize".')

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

    def save_configuration(self, directory: pathlib.Path):
        max_key_length = max(len(key) for key in self.configuration.keys())
        with open(directory / 'configuration.txt', 'w') as fh:
            fh.write('Configuration:\n')
            for key, value in self.configuration.items():
                fh.write(f'{key:<{max_key_length}} : {value}\n')

        log.info(f'Configuration saved to {directory}/configuration.txt')

    def save_hyperparameters(self, directory: pathlib.Path):
        max_key_length = max(len(key) for key in self.hyperparams.keys())
        with open(directory / 'hyperparameters.txt', 'w') as fh:
            fh.write('Hyperparameters:\n')
            for key, value in self.hyperparams.items():
                if value is not None:
                    fh.write(f'{key:<{max_key_length}} : {value}\n')

        log.info(f'Hyperparameters saved to {directory}/hyperparameters.txt')

    @staticmethod
    def _print_dict(d: dict) -> None:
        if torch.distributed.get_rank() == 0:
            for key, value in d.items():
                log.info(f'{key:<20}: {value}')

