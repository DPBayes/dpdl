import json
import logging
import pathlib
import pickle
import shutil
import subprocess

import optuna

from .configurationmanager import ConfigurationManager

def save_study(
        config_manager: ConfigurationManager,
        study: optuna.study.Study,
    ):
    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name

    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    with open(full_log_dir / 'trials.json', 'w') as fh:
        fh.write(study.trials_dataframe().to_json())

    with open(full_log_dir / 'trials.csv', 'w') as fh:
        fh.write(study.trials_dataframe().to_csv())

    with open(full_log_dir / 'optuna-study.pkl', 'wb') as fh:
        git_hash = _get_git_hash()
        pickle.dump(study, fh)

    with open(full_log_dir / 'best-params.json', 'w') as fh:
        json.dump(study.best_params, fh)

    with open(full_log_dir / 'best-value', 'w') as fh:
        fh.write(str(study.best_value) + '\n')

    with open(full_log_dir / 'git-hash', 'w') as fh:
        git_hash = _get_git_hash()
        fh.write(str(git_hash) + '\n')

    with open(full_log_dir / 'results-and-configuration.json', 'w') as fh:
        d = {}
        d['best_params'] = study.best_params
        d['best_value'] = study.best_value
        d['configuration'] = config_manager.configuration.dict()
        d['git-hash'] = study.best_value
        d['hyperparameters'] = config_manager.hyperparams.dict()

        json.dump(d, fh)

def start_experiment_logging(
        log: logging.Logger,
        config_manager: ConfigurationManager,
        overwrite: bool = False,
    ):

    log_dir = config_manager.configuration.log_dir
    experiment_name = config_manager.configuration.experiment_name

    # create a directory for the experiments and start logging there
    overwrite = config_manager.configuration.overwrite_experiment
    experiment_directory = _create_experiment_directory(log_dir, experiment_name, overwrite)
    _start_logging_to_files(log, experiment_directory)

    # save configuration
    config_manager.save_configuration(experiment_directory)
    config_manager.save_hyperparameters(experiment_directory)

def _get_git_hash():
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    git_hash = process.communicate()[0].strip()
    return git_hash

def _create_experiment_directory(
        log_dir: str = 'log_dir',
        experiment_name: str = 'Default experiment',
        overwrite: bool = False,
    ) -> pathlib.Path:

    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    if full_log_dir.exists() and not overwrite:
        raise FileExistsError(f'The directory {full_log_dir} already exists and not asked to overwrite.')

    if full_log_dir.exists() and overwrite:
        shutil.rmtree(full_log_dir)

    full_log_dir.mkdir(parents=True)

    return full_log_dir

def _start_logging_to_files(log: logging.Logger, log_path: pathlib.Path):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a file handler for saving stdout logs to a file
    stdout_file_handler = logging.FileHandler(log_path / 'stdout.txt')
    stdout_file_handler.setLevel(logging.INFO)
    stdout_file_handler.setFormatter(formatter)
    log.addHandler(stdout_file_handler)

    # create a file handler for saving stderr logs to a file
    stderr_file_handler = logging.FileHandler(log_path / 'stderr.txt')
    stderr_file_handler.setLevel(logging.WARNING)
    stderr_file_handler.setFormatter(formatter)
    log.addHandler(stderr_file_handler)
