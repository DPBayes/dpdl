import logging
import time
import torch
import typer
import sys

from typing import Optional, List
from typing_extensions import Annotated

from .configurationmanager import ConfigurationManager
from .hyperparameteroptimizer import HyperparameterOptimizer
from .trainer import TrainerFactory
from .models.model_factory import ModelFactory
from .utils import seed_everything
from .experimentmanager import start_experiment_logging, log_runtime, log_test_metrics, log_final_epsilon

log = logging.getLogger(__name__)

def cli(
        ctx: typer.Context,
        command: Annotated[
            str,
            typer.Argument(
                help='Command to run ("train", "optimize", or "show-layers")',
            )
        ],
        use_steps: Annotated[
            bool,
            typer.Option(
                help='Use steps instead of epochs',
                rich_help_panel='Training options',
            )
        ] = False,
        epochs: Annotated[
            int,
            typer.Option(
                help='Number of epochs to train',
                rich_help_panel='Training options',
            )
        ] = None,
        total_steps: Annotated[
            int,
            typer.Option(
                help='Total number of gradient updates',
                rich_help_panel='Training options',
            )
        ] = None,
        learning_rate: Annotated[
            float,
            typer.Option(
                help='Learning rate',
                rich_help_panel='Training options',
            )
        ] = 1e-3,
        batch_size: Annotated[
            Optional[int],
            typer.Option(
                help='Batch size',
                rich_help_panel='Training options',
            )
        ] = None,
        sample_rate: Annotated[
            Optional[float],
            typer.Option(
                help='Sample rate',
                rich_help_panel='Training options',
            )
        ] = None,
        optimizer: Annotated[
            str,
            typer.Option(
                help='Optimizer',
                rich_help_panel='Training options',
            )
        ] = 'Adam',
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
        cache_features: Annotated[
            Optional[bool],
            typer.Option(
                help='Cache features from feature extractor (requires head-only training)',
                rich_help_panel='Training options',
            )
        ] = False,
        evaluation_mode: Annotated[
            bool,
            typer.Option(
                help='Enable evaluation mode (train on train+valid and validate on test)',
                rich_help_panel='Training options',
            )
        ] = False,
        model_name: Annotated[
            str,
            typer.Option(
                help='PyTorch Image Models (timm) model name',
                rich_help_panel='Model options',
            )
        ] = 'resnetv2_50x1_bit.goog_in21k',
        pretrained: Annotated[
            bool,
            typer.Option(
                help='Use pretrained model',
                rich_help_panel='Model options',
            )
        ] = True,
        zero_head: Annotated[
            bool,
            typer.Option(
                help='Set model head weights to zero',
                rich_help_panel='Model options',
            )
        ] = False,
        peft: Annotated[
            str,
            typer.Option(
                help='Use Parameter Efficient Fine-tuning ("lora", "film", "head-only")',
                rich_help_panel='Model options',
            )
        ] = None,
        weight_perturbation_level: Annotated[
            float,
            typer.Option(
                help='Pretrained weight perturbation noise level',
                rich_help_panel='Model options',
            )
        ] = 0,
        model_save_fpath: Annotated[
            Optional[str],
            typer.Option(
                help='File path for saving of the trained model',
                rich_help_panel='Model options',
            )
        ] = None,
        dataset_name: Annotated[
            str,
            typer.Option(
                help='Dataset name',
                rich_help_panel='Dataset options',
            )
        ] = 'cifar10',
        subset_size: Annotated[
            float,
            typer.Option(
                help='Only load subset of the dataset (0.1 indicate 10%)',
                rich_help_panel='Dataset options',
            )
        ] = None,
        validation_size: Annotated[
            Optional[float],
            typer.Option(
                help='Validation set size, if we need to split it from train (0.1 indicates 10%)',
                rich_help_panel='Dataset options',
            )
        ] = 0.1,
        test_size: Annotated[
            Optional[float],
            typer.Option(
                help='Test set size, if we need to split it from train (0.1 indicates 10%)',
                rich_help_panel='Dataset options',
            )
        ] = 0.1,
        shots: Annotated[
            Optional[int],
            typer.Option(
                help='Number of shots (training example per class) to use',
                rich_help_panel='Dataset options',
            )
        ] = None,
        stratify_shots: Annotated[
            Optional[bool],
            typer.Option(
                help='Use stratified sampling when constructing few-shot dataset',
                rich_help_panel='Dataset options',
            )
        ] = True,
        dataset_label_field: Annotated[
            Optional[str],
            typer.Option(
                help='Name of the field that determines label for the dataset',
                rich_help_panel='Dataset options',
            )
        ] = None,
        imbalance_factor: Annotated[
            Optional[float],
            typer.Option(
                help='Parameter of the exponential distribution for imbalancing an dataset',
                rich_help_panel='Dataset options',
            )
        ] = None,
        max_test_examples: Annotated[
            Optional[int],
            typer.Option(
                help='Limit for the maximum number of examples in test/validation set (truncate)',
                rich_help_panel='Dataset options',
            )
        ] = None,
        cache_dataset_transforms: Annotated[
            Optional[bool],
            typer.Option(
                help='Cache the image transformations on disk (faster to disable for small images)',
                rich_help_panel='Dataset options',
            )
        ] = False,
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
        overwrite_experiment: Annotated[
            bool,
            typer.Option(
                help='Overwrite existing experiment logs',
                rich_help_panel='Logging options',
            )
        ] = False,
        noise_multiplier: Annotated[
            Optional[float],
            typer.Option(
                help='Noise multiplier',
                rich_help_panel='Opacus options',
            )
        ] = None,
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
        poisson_sampling: Annotated[
            Optional[bool],
            typer.Option(
                help='Enable Opacus Poisson sampling',
                rich_help_panel='Opacus options',
            )
        ] = True,
        normalize_clipping: Annotated[
            Optional[bool],
            typer.Option(
                help='Normalize clipping (to decouple the learning rate and max_grad_norm)',
                rich_help_panel='Opacus options',
            )
        ] = False,
        record_snr: Annotated[
            Optional[bool],
            typer.Option(
                help='Record the Signal-to-Noise ratio per step (requires --use-steps)',
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
        ] = 'optuna.journal',
        optuna_resume: Annotated[
            Optional[bool],
            typer.Option(
                help='Resume previous Optuna study',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = False,
        optuna_sampler: Annotated[
            Optional[str],
            typer.Option(
                help='Optuna sampler (a class from optuna.samplers)',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'BoTorchSampler',
        record_gradient_norms: Annotated[
            Optional[bool],
            typer.Option(
                help='Record layer-wise gradients before clipping',
                rich_help_panel='',
            )
        ] = False,
        verbose_callback: Annotated[
            Optional[bool],
            typer.Option(
                help='Enable debug callback for detailed output',
                rich_help_panel='',
            )
        ] = False,
        wandb_record: Annotated[
            Optional[bool],
            typer.Option(
                help="Enable Weights & Biases logging",
                rich_help_panel="",
            ),
        ] = False,
    ):

    config_manager = ConfigurationManager(ctx.params)

    if config_manager.get_command() == 'show-layers':
        log.info(config_manager.configuration)
        log.info('Showing model layers.')
        model, _ = ModelFactory.get_model(
            config_manager.configuration,
            config_manager.hyperparams,
        )
        model.show_layers()

        return

    # ConfigurationManager knows our experiment directory, so let's start logging also there
    if torch.distributed.get_rank() == 0:
        start_experiment_logging(log.parent, config_manager)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()

    if config_manager.get_command() == 'train':
        if torch.distributed.get_rank() == 0:
            log.info('Starting training.')
            log.info(config_manager.hyperparams)
            log.info(config_manager.configuration)

        seed_everything(config_manager.configuration.seed)

        trainer = TrainerFactory.get_trainer(config_manager)

        start_time = time.time()
        trainer.fit()
        end_time = time.time()

        # log test accuracy and run time, and save model if asked
        if torch.distributed.get_rank() == 0:
            log.info('Evaluating on test set..')
            _, test_metrics = trainer.test()

            log_test_metrics(config_manager, test_metrics)
            log_runtime(config_manager, start_time, end_time)
            log_final_epsilon(config_manager, trainer)

            if save_fpath := config_manager.configuration.model_save_fpath:
                log.info(f'Saving model to: "{save_fpath}"')
                trainer.save_model(save_fpath)

    if config_manager.get_command() == 'optimize':
        if torch.distributed.get_rank() == 0:
            log.info('Starting hyperparameter optimization.')
            log.info(config_manager.configuration)

        seed_everything(config_manager.configuration.seed)

        start_time = time.time()
        HyperparameterOptimizer.optimize_hypers(config_manager)
        end_time = time.time()

        # log the runtime
        if torch.distributed.get_rank() == 0:
            log_runtime(config_manager, start_time, end_time)

