import logging
import pathlib
import sys

from .configurationmanager import ConfigurationManager

def configure_logger() -> logging.Logger:
    log = logging.getLogger('dpdl')
    log.setLevel(logging.INFO)

    # create a stream handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the new handler
    log.addHandler(handler)

    return log

def start_experiment_logging(
        log: logging.Logger,
        configurationmanager: ConfigurationManager,
        overwrite: bool = False,
    ):

    log_dir = configurationmanager.get_value('log_dir')
    experiment_name = configurationmanager.get_value('experiment_name')

    # create a directory for the experiments and start logging there
    overwrite = configurationmanager.get_value('overwrite_experiment')
    experiment_directory = _create_experiment_directory(log_dir, experiment_name, overwrite)
    _update_logger_with_file_handler(log, experiment_directory)

    # Save configuration and hyperparameters
    configurationmanager.save_configuration(experiment_directory)
    configurationmanager.save_hyperparameters(experiment_directory)

def _create_experiment_directory(
        log_dir: str = 'log_dir',
        experiment_name: str = 'Default experiment',
        overwrite: bool = False,
    ) -> pathlib.Path:

    full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

    if full_log_dir.exists() and not overwrite:
        raise FileExistsError(f'The directory {full_log_dir} already exists and not asked to overwrite.')

    if full_log_dir.exists() and overwrite:
        for item in full_log_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()

        full_log_dir.rmdir()

    full_log_dir.mkdir(parents=True, exist_ok=True)

    return full_log_dir

def _update_logger_with_file_handler(log: logging.Logger, log_path: pathlib.Path):
    # create a file handler for saving logs to a file
    file_handler = logging.FileHandler(log_path / 'logs.txt')
    file_handler.setLevel(logging.INFO)

    # get the existing formatter from the stream handler
    formatter = log.handlers[0].formatter

    # set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # add the file handler to the logger
    log.addHandler(file_handler)
