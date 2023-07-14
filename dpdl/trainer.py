import logging
import opacus
import torch

from opacus.utils.batch_memory_manager import BatchMemoryManager

from .callbacks import CallbackHandler, CallbackFactory
from .cli import ConfigurationManager
from .configurationmanager import ConfigurationManager
from .datamodules import DataModule, DataModuleFactory
from .models import ModelFactory
from .optimizers import OptimizerFactory

# You are using a CUDA device ('AMD Radeon Graphics') that has Tensor Cores.
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.
# For more details, read
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')

log = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,

        # essentials
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: DataModule,

        # generic params
        epochs: int = 10,
        validation_frequency: int = 1,
        seed: int = 0,
        #checkpoint_dir: str = "./checkpoints",
        #checkpoint_frequency: int = 1,
        callback_handler: CallbackHandler = None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.datamodule = datamodule
        self.epochs = epochs
        self.validation_frequency = validation_frequency
        self.seed = seed

        if not callback_handler:
            self.callback_handler = CallbackHandler()
        else:
            self.callback_handler = callback_handler

        self.setup()

    def setup(self):
        self.model = self.model.cuda()

        local_rank = torch.distributed.get_rank()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

    def fit(self):
        self.callback_handler.call('on_train_start', self)

        for epoch in range(self.epochs):
            epoch_loss = self.fit_one_epoch(epoch)

            if self.validation_frequency and epoch % self.validation_frequency == 0:
                self.validate(epoch)

        self.callback_handler.call('on_train_end', self)

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        total_loss = 0
        for batch_idx, batch in enumerate(self.datamodule.train_dataloader):
            batch_loss = self.fit_one_batch(batch_idx, batch)
            total_loss = total_loss + batch_loss

        epoch_loss = total_loss / (batch_idx + 1)
        #self.fabric.log('Train/loss', epoch_loss, epoch)

        self.callback_handler.call('on_train_epoch_end', self, epoch, epoch_loss)

        return epoch_loss

    def _unwrap_model(self):
        return self.model.module

    def fit_one_batch(self, batch_idx, batch):
        self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)

        X, y = batch
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        self.optimizer.zero_grad()
        logits = self.model(X)

        loss = self._unwrap_model().criterion(logits, y)
        loss.backward(loss)
        self.optimizer.step()
        loss = loss.item()

        self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, loss)
        return loss

    def validate(self, epoch=None):
        self.model.eval()
        torch.set_grad_enabled(False)

        total_loss = 0
        self.callback_handler.call('on_validation_epoch_start', self, epoch)
        for batch_idx, batch in enumerate(self.datamodule.val_dataloader):
            batch_loss = self.validate_one_batch(batch_idx, batch)
            total_loss = total_loss + batch_loss

        valid_loss = total_loss / (batch_idx + 1)
        self.callback_handler.call('on_validation_epoch_end', self, epoch, valid_loss)

        torch.set_grad_enabled(True)
        self.model.train()

        return valid_loss

    def validate_one_batch(self, batch_idx, batch):
        self.callback_handler.call('on_validation_batch_start', self, batch_idx, batch)

        X, y = batch
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        logits = self.model(X)
        loss = self._unwrap_model().criterion(logits, y)

        loss = loss.item()
        self.callback_handler.call('on_validation_batch_end', self, batch_idx, batch, loss)

        return loss

class DifferentiallyPrivateTrainer(Trainer):
    def __init__(
        self,
        *,
        # privacy params
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        clipping: str = 'flat',
        accountant: str = 'rdp',
        secure_mode: bool = False,
        target_epsilon: float = 0,
        target_delta: float = 0,
        physical_batch_size: int = 64,
        seed: int = 0,
        **kwargs,
    ):

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clipping = clipping
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.physical_batch_size = physical_batch_size
        self.seed = seed

        # setup opacus privacy engine
        self.privacy_engine = opacus.PrivacyEngine(accountant=accountant, secure_mode=secure_mode)

        super().__init__(**kwargs)

    def _has_target_privacy_params(self):
        if not any([self.target_epsilon, self.target_delta]):
            return False

        if self.target_epsilon and not self.target_delta:
            raise(RuntimeError('Parameter "target_delta" present, but "target_epsilon" is missing.'))

        if self.target_delta and not self.target_epsilon:
            raise(RuntimeError('Parameter "target_delta" present, but "target_epsilon" is missing.'))

        if self.target_epsilon and self.noise_multiplier:
            raise(RuntimeError('Parameter "noise_multiplier" can not be used when target epsilon is given.'))

        return True

    def setup(self):
        super().setup()

        noise_generator = torch.Generator(device=torch.cuda.current_device())
        if self.seed:
            noise_generator.manual_seed(self.seed)

        model = self.model
        optimizer = self.optimizer
        train_dataloader = self.datamodule.train_dataloader

        # setup differential privacy for the model, optimize, and dataloader
        if self._has_target_privacy_params():
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=self.epochs,
                noise_generator=noise_generator,
            )
        else:
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping,
                noise_generator=noise_generator,
            )

        # are we distributed?
        if torch.distributed.get_world_size() > 1:
            # DifferentiallyPrivateDistributedDataParallel is actually a no-op in Opacus, but
            # let's wrap anyway in case of future api changes. https://opacus.ai/tutorials/ddp_tutorial
            dp_model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(dp_model)

        # put the DP'ifyed stuff back into Fabric wrappers
        self.model = dp_model
        self.datamodule.train_dataloader = dp_dataloader
        self.optimizer = dp_optimizer

    def get_epsilon(self, delta):
        return self.privacy_engine.get_epsilon(delta)

    def _unwrap_model(self):
        return self.model.module._module.module

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        # save the current dataloader, we are going to use a virtual
        # dataloader to enable larger batches
        original_dataloader = self.datamodule.train_dataloader

        total_loss = 0
        with BatchMemoryManager(
            data_loader=self.datamodule.train_dataloader,
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:
            # the virtual data loader created by BatchMemoryManager enables us to use larger
            # logical batch sizes that fit in a GPU.
            self.datamodule.train_dataloader = virtual_dataloader

            for batch_idx, batch in enumerate(self.datamodule.train_dataloader):
                batch_loss = self.fit_one_batch(batch_idx, batch)
                total_loss = total_loss + batch_loss

        epoch_loss = total_loss / (batch_idx + 1)
        #self.fabric.log('Train/loss', epoch_loss, epoch)

        # we have enumerated the batch, let's swap back to the original dataloader
        self.datamodule.train_dataloader = original_dataloader

        self.callback_handler.call('on_train_epoch_end', self, epoch, epoch_loss)
        return epoch_loss

class TrainerFactory():
    @staticmethod
    def _get_basic_trainer(configuration: dict, hyperparams: dict) -> Trainer:
        # setup data, model, and optimizer
        datamodule = DataModule.get_datamodule(configuration, hyperparams)
        model = Model.get_model(configuration, hyperparams)
        optimizer = get_optimizer(configuration, hyperparams, model)
        callback_handler = CallbackHandler(
            get_callbacks(configuration, hyperparams)
        )

        # instantiate a trainer without dp
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            epochs=hyperparams['epochs'],
            seed=configuration['seed'],
            callback_handler=callback_handler,
        )

        return trainer

    @staticmethod
    def _get_differentially_private_trainer(configuration: dict, hyperparams: dict) -> Trainer:
        # setup data, model, and optimizer
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams)
        model = ModelFactory.get_model(configuration, hyperparams)
        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams)
        )

        # are we given a target epsilon?
        if 'target_epsilon' in hyperparams:
            # if we have target epsilon, set target delta = 1/N
            target_delta = 1 / len(datamodule.train_dataloader.dataset)
            target_epsilon = hyperparams['target_epsilon']

            # if target epsilon is given, then opacus will calculate
            # the noise multiplier for us
            hyperparams['noise_multiplier'] = None
        else:
            target_delta = None
            target_epsilon = None

        # instantiate a differentialy private trained
        trainer = DifferentiallyPrivateTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            # hypers
            epochs=hyperparams['epochs'],
            noise_multiplier=hyperparams['noise_multiplier'],
            max_grad_norm=hyperparams['max_grad_norm'],
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            # config
            secure_mode=configuration['secure_mode'],
            clipping=configuration['clipping'],
            physical_batch_size=configuration['physical_batch_size'],
            seed=configuration['seed'],
            callback_handler=callback_handler,
        )

        return trainer

    @staticmethod
    def get_trainer(config: ConfigurationManager) -> Trainer:
        configuration = config.get_configuration()
        hyperparams = config.get_hyperparams()

        # are we differentially private?
        if configuration['privacy']:
            return TrainerFactory._get_differentially_private_trainer(configuration, hyperparams)

        return TrainerFactory._get_basic_trainer(configuration, hyperparams)

