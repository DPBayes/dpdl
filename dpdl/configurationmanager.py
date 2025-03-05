import logging
import pathlib
import torch
import typer

from pydantic import BaseModel, root_validator
from typing import Optional, List, Literal

log = logging.getLogger(__name__)

class Hyperparameters(BaseModel):
    learning_rate: float = 1e-3
    epochs: Optional[int] = None
    total_steps: Optional[int] = None
    batch_size: Optional[int] = None
    sample_rate: Optional[float] = None
    noise_multiplier: Optional[float]
    max_grad_norm: Optional[float]
    target_epsilon: Optional[float]
    noise_batch_ratio: Optional[float]
    privacy: bool = True # Only used in __str__

    @root_validator(pre=True)
    def check_batch_size_or_sample_rate(cls, values):
        batch_size, sample_rate = values.get('batch_size'), values.get('sample_rate')

        if all([batch_size, sample_rate]):
            raise ValueError('Either batch_size or sample_rate must be set, but not both.')

        return values

    @root_validator(pre=True)
    def check_target_epsilon_or_noise_multiplier(cls, values):
        target_epsilon, noise_multiplier = values.get('target_epsilon'), values.get('noise_multiplier')

        if all([target_epsilon, noise_multiplier]):
            raise ValueError('Both, target_epsilon and noise_multiplier given.')

        return values

    @root_validator(pre=True)
    def check_target_epsilon_or_noise_batch_ratio(cls, values):
        target_epsilon, noise_batch_ratio = values.get('target_epsilon'), values.get('noise_batch_ratio')

        if all([target_epsilon, noise_batch_ratio]):
            raise ValueError('Both, target_epsilon and noise_batch_ratio given.')

        return values

    @root_validator(pre=True)
    def check_noise_batch_ratio_or_noise_multiplier(cls, values):
        noise_multiplier, noise_batch_ratio = values.get('noise_multiplier'), values.get('noise_batch_ratio')

        if all([noise_multiplier, noise_batch_ratio]):
            raise ValueError('Both, noise_multiplier and noise_batch_ratio given.')

        return values

    def __str__(self):
        hypers = [
            ('Epochs', self.epochs),
            ('Total steps', self.total_steps),
            ('Learning rate', self.learning_rate),
            ('Batch size', self.batch_size),
        ]

        if self.privacy:
            privacy_hypers = [
                ('Sample rate', self.sample_rate),
                ('Noise multiplier', self.noise_multiplier),
                ('Max grad norn', self.max_grad_norm),
                ('Target epsilon', self.target_epsilon),
                ('Noise-batch ratio', self.noise_batch_ratio),
            ]
            hypers.extend(privacy_hypers)

        max_key_length = max(len(hyper[0]) for hyper in hypers)
        hyper_str = [f'{hyper[0]:<{max_key_length}}: {hyper[1]}' for hyper in hypers]

        return 'Hyperparameters:\n  ' + '\n  '.join(hyper_str) + '\n'

class Configuration(BaseModel):
    command: Literal['train', 'optimize', 'show-layers']
    privacy: bool = True
    model_name: str = 'resnet50'
    optimizer: str = 'Adam'
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
    accountant: str = 'prv'
    poisson_sampling: bool = True
    normalize_clipping: bool = False
    n_trials: int = 20
    optuna_random_trials: int = 10
    target_hypers: List[str] = []
    optuna_target_metric: str = 'loss'
    optuna_direction: Literal['minimize', 'maximize'] = 'minimize'
    optuna_config: str = 'conf/optuna_hypers.conf'
    optuna_manual_trials: Optional[str] = None
    optuna_journal: str = 'optuna.journal'
    optuna_resume: bool = False
    optuna_sampler: str = 'BoTorchSampler'
    subset_size: Optional[float]
    shots: Optional[int]
    stratify_shots: Optional[bool] = True
    zero_head: bool = False
    peft: Optional[Literal['lora', 'film', 'head-only']]
    pretrained: bool = True
    cache_features: Optional[bool] = False
    use_steps: Optional[bool] = False
    evaluation_mode: Optional[bool] = False
    dataset_label_field: Optional[str] = None
    max_test_examples: Optional[int] = None
    imbalance_factor: Optional[float] = None
    validation_size: Optional[float] = 0.1
    test_size: Optional[float] = 0.1
    model_save_fpath: Optional[str] = None
    record_gradient_norms: Optional[bool] = False
    verbose_callback: Optional[bool] = False
    cache_dataset_transforms: Optional[bool] = False
    weight_perturbation_level: float = 0
    record_loss_by_step: Optional[bool] = False
    record_loss_by_epoch: Optional[bool] = False
    checkpoint_step_interval: Optional[int] = None

    class Config:
        # Fix Pydantic warning:
        # UserWarning: Field "model_name" has conflict with protected namespace "model_".
        protected_namespaces = ()

    @root_validator(pre=True)
    def check_record_loss_by_step(cls, values):
        record_loss_by_step = values.get('record_loss_by_step')
        use_steps = values.get('use_steps')

        if record_loss_by_step and not use_steps:
            raise ValueError('Unable to record trian loss by step when using epochs. Hint: `--use-steps`')

        return values

    @root_validator(pre=True)
    def check_command(cls, values):
        command = values.get('command')

        if command not in ['train', 'optimize', 'show-layers']:
            raise ValueError('Command must be "train", "optimize", or "show-layers".')

        return values

    @root_validator(pre=True)
    def check_total_steps(cls, values):
        total_steps = values.get('total_steps')
        use_steps = values.get('use_steps')
        epochs = values.get('epochs')

        if total_steps and epochs:
            raise ValueError('Parameters "epochs" and "total_steps" are exclusive.')

        if total_steps and not use_steps:
            raise ValueError('Parameter "total_steps" requires also "use_steps".')

        return values

    @root_validator(pre=True)
    def check_shots_and_subset_size(cls, values):
        shots = values.get('shots')
        subset_size = values.get('subset_size')

        if shots and subset_size:
            raise ValueError('Parameters "shots" and "subset_size" are exclusive.')

        return values

    @root_validator(pre=True)
    def check_feature_cache(cls, values):
        cache_features = values.get('cache_features')
        peft_method = values.get('peft')

        if cache_features and peft_method != 'head-only':
            raise ValueError(f'Head only training required if feature cache is enabled.')

        return values

    def __str__(self):
        attributes = [
            ('Command', self.command),
            ('Privacy', self.privacy),
            ('Model name', self.model_name),
            ('Optimizer', self.optimizer),
            ('Dataset name', self.dataset_name),
            ('Dataset label field', self.dataset_label_field),
            ('Dataset imbalance factor', self.imbalance_factor),
            ('Cache dataset transforms', self.cache_dataset_transforms),
            ('Validation size', self.validation_size),
            ('Test size', self.test_size),
            ('Physical batch size', self.physical_batch_size),
            ('Num workers', self.num_workers),
            ('Validation frequency', self.validation_frequency),
            ('Seed', self.seed),
            ('Log dir', self.log_dir),
            ('Experiment dame', self.experiment_name),
            ('Overwrite experiment', self.overwrite_experiment),
            ('Shots', self.shots),
            ('Use stratified sampling for few-shot dataset', self.stratify_shots),
            ('Subset size', self.subset_size),
            ('Zero head weights', self.zero_head),
            ('PEFT method', self.peft),
            ('Use pretrained model', self.pretrained),
            ('Pretrained model weight perturbation noise level', self.weight_perturbation_level),
            ('Use precomputed features', self.cache_features),
            ('Use steps instead of epochs', self.use_steps),
            ('Evaluation mode', self.evaluation_mode),
            ('Model save file path', self.model_save_fpath),
            ('Record gradient norms', self.record_gradient_norms),
            ('Record train loss by step', self.record_loss_by_step),
            ('Record train/valid loss by epoch', self.record_loss_by_epoch),
            ('Checkpoint every nth step', self.checkpoint_step_interval),
            ('Enable callback debug logging', self.verbose_callback),
        ]

        if self.privacy:
            privacy_attributes = [
                ('Clipping mode', self.clipping_mode),
                ('Secure mode', self.secure_mode),
                ('Accountant', self.accountant),
                ('Poisson sampling', self.poisson_sampling),
                ('Normalize clipping', self.normalize_clipping),
            ]
            attributes.extend(privacy_attributes)

        if self.command == 'optimize':
            optuna_attributes = [
                ('N trials', self.n_trials),
                ('Target hypers', ', '.join(self.target_hypers)),
                ('Optuna target metric', self.optuna_target_metric),
                ('Optuna direction', self.optuna_direction),
                ('Optuna config', self.optuna_config),
                ('Optuna manual trials configuration', self.optuna_manual_trials),
                ('Optuna journal', self.optuna_journal),
                ('Optuna resume', self.optuna_resume),
                ('Optuna number of random trials', self.optuna_random_trials),
            ]
            attributes.extend(optuna_attributes)

        max_key_length = max(len(attr[0]) for attr in attributes)
        attribute_str = [f'{attr[0]:<{max_key_length}}: {attr[1]}' for attr in attributes]

        return 'Configuration:\n  ' + '\n  '.join(attribute_str) + '\n'

class ConfigurationManager:
    def __init__(self, cli_params: dict):
        self.command = cli_params['command']

        self.configuration = Configuration(**cli_params)
        self.hyperparams = Hyperparameters(**cli_params)

        # remove the target hypers from hyperparams as they will be set in trials
        for target_hyper in self.configuration.target_hypers:
            setattr(self.hyperparams, target_hyper, None)

    def get_command(self):
        return self.command

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
