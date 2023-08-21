import gc
import logging
import optuna
import torch
import yaml

from functools import partial

from .trainer import TrainerFactory
from .configurationmanager import ConfigurationManager

log = logging.getLogger(__name__)

class HyperparameterOptimizer:
    # helper function for reading Optuna hyperparameter configuration that
    # includes types, options, lower and upper bounds, etc for hyperparams
    @staticmethod
    def read_optuna_config(config_fpath):
        with open(config_fpath, 'rb') as fh:
            config = yaml.safe_load(fh)

        return config

    @staticmethod
    def optimize_hypers(configurationmanager: ConfigurationManager) -> None:
        configuration = configurationmanager.get_configuration()
        hyperparams = configurationmanager.get_hyperparams()

        # the targeted hypers from command line
        target_hypers = configuration['target_hypers']

        if len(target_hypers) == 0:
            raise typer.BadParameter('Bayesian optimization enabled and no target hyperparameters for optimization.')

        optuna_config = HyperparameterOptimizer.read_optuna_config(configuration['optuna_config'])

        # check that the target hypers are known and appear in Optuna configuration
        for target_hyper in target_hypers:
            if not target_hyper in hyperparams:
                raise typer.BadParameter(f'Cannot optimize unknown hyperparameter "{target_hyper}".')
            if not target_hyper in optuna_config:
                config_fpath = configuration['optuna_config']
                raise typer.BadParameter(f'Hyperparameter "{target_hyper}" not found in Optuna configuration file "{config_fpath}".')

        # optuna needs a global process group with 'gloo' backend for communication.
        # "Create a global gloo backend when group is None and WORLD is nccl."
        # - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.TorchDistributedTrial.html
        optuna_process_group = torch.distributed.new_group(
            backend='gloo',
        )

        # the optimization objective
        objective = partial(
            HyperparameterOptimizer.objective,
            configurationmanager,
            optuna_config,
            target_hypers,
            optuna_process_group,
        )

        # start Optuna study only on rank zero
        if torch.distributed.get_rank() == 0:
            journal_fpath = configuration['optuna_journal']

            # we manually define the sampler to be able to set the seed
            sampler = optuna.samplers.TPESampler(seed=configuration['seed'])

            # we will store the information about the trials on disk in a journal file
            storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(journal_fpath))

            # should we try to resume an existing study?
            load_if_exists = configuration['optuna_resume']

            study = optuna.create_study(
                storage=storage,
                study_name=configuration['experiment_name'],
                sampler=sampler,
                load_if_exists=load_if_exists,
                direction=configuration['optuna_direction'],
            )

            # no we can actually run the study
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

        # log the results of the best trial
        if torch.distributed.get_rank() == 0:
            trial = study.best_trial
            log.info(f'Best objective ralue: {trial.value}')
            log.info('Params: ')
            for key, value in trial.params.items():
                log.info(f' - {key}: {value}')

    @staticmethod
    def objective(
        configurationmanager: ConfigurationManager,
        optuna_config: dict,
        target_hypers: list,
        process_group: torch.distributed.ProcessGroupGloo,
        trial: optuna.trial.Trial,
    ):
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

            # update the hyperparameter value in configuration
            configurationmanager.set_hyper(target_hyper, hyper_value)

        # while tuning hypers, do not validate
        configurationmanager.set_value('validation_frequency', 0)

        # train the model
        trainer = TrainerFactory.get_trainer(configurationmanager)

        if torch.distributed.get_rank() == 0:
            log.info(f'Starting trial {trial.number}.')
            configurationmanager.print_hyperparams()

        trainer.fit()

        # optimization objective is the validation loss
        loss, metrics = trainer.validate()

        # find the correct metric value to use as optimization objective
        target_metric = configurationmanager.get_value('optuna_target_metric')
        if target_metric == 'loss':
            objective = loss
        elif target_metric in metrics:
            objective = metrics[target_metric]
        else:
            raise f'Metric "{target_metric}" not found in metrics (' + ', '.join(metrics.keys()) + ')'

        # without these, this randomly leads to a nasty segmentation fault
        # on AMD, and a memory related CUDA error on Nvidia.
        del trainer, process_group, trial
        gc.collect()

        return objective
