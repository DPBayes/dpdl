import logging
import math
import opacus
import torch
import torchmetrics

from opacus.utils.batch_memory_manager import BatchMemoryManager

from .callbacks import CallbackHandler, CallbackFactory
from .configurationmanager import ConfigurationManager, Configuration, Hyperparameters
from .datamodules import DataModule, DataModuleFactory
from .models import ModelFactory
from .optimizers import OptimizerFactory

# You are using a CUDA device ('AMD Radeon Graphics') that has Tensor Cores.
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.
# For more details, read
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('high')

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
        physical_batch_size: int = 40,
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
        self.physical_batch_size = physical_batch_size

        # use torchmetrics mean aggregation to track the losses
        self.train_loss = torchmetrics.aggregation.MeanMetric().cuda()

        # for validation and test sets
        self.evaluation_loss = torchmetrics.aggregation.MeanMetric().cuda()

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
            self.fit_one_epoch(epoch)

            if self.validation_frequency and epoch % self.validation_frequency == 0:
                self.validate(epoch)

        self.callback_handler.call('on_train_end', self)

    def fit_on_train_and_valid(self):
        # safe the current training dataloader as we are going to
        # temporarily change it.
        original_train_dataloader = self.datamodule.get_dataloader('train')
        original_valid_dataloader = self.datamodule.get_dataloader('valid')

        # when all the training have been done, we want to train also
        # on the validation set to squeeze the last performance out
        self.datamodule.set_dataloader('train', self.datamodule.get_dataloader('train_and_valid'))
        self.datamodule.set_dataloader('valid', self.datamodule.get_dataloader('test'))

        # now let's just fit as usual
        self.fit()

        # restore the original dataloader
        self.datamodule.set_dataloader('train', original_train_dataloader)
        self.datamodule.set_dataloader('valid', original_valid_dataloader)

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):
            self.fit_one_batch(batch_idx, batch)

        # compute the train loss for the epoch and reset
        epoch_loss = self.train_loss.compute()
        self.train_loss.reset()

        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()

        self.callback_handler.call('on_train_epoch_end', self, epoch, epoch_loss, metrics)

    def _unwrap_model(self):
        # the model is wrapped inside torch distributed,
        # here we just return the vanilla model
        return self.model.module

    def fit_one_batch(self, batch_idx, batch):
        self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)

        X, y = batch
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
        # then process the sub batches one at a time and call backward.
        # when all the sub batches have been processed we can finally step the optimizer.
        X_split = X.split(self.physical_batch_size, dim=0)
        y_split = y.split(self.physical_batch_size, dim=0)

        # zero the grads as usually before doing anything
        self.optimizer.zero_grad()

        batch_loss = 0
        # process the sub batches one at a time
        N = len(X_split)
        for i in range(N):
            X_splitted = X_split[i]
            y_splitted = y_split[i]

            logits = self.model(X_splitted)
            loss = self._unwrap_model().criterion(logits, y_splitted) / N # NB: normalize loss
            loss.backward(loss)

            # keep track of the batch loss
            batch_loss = batch_loss + loss.item()

            # update the metrics if there are any
            preds = torch.argmax(logits, dim=1)
            self._unwrap_model().train_metrics.update(preds, y_split[i])

        # after accumulating the gradients for all the sub batches we can finally update weights.
        self.optimizer.step()

        self.train_loss.update(batch_loss)

        self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, loss)

    def validate(self, epoch=None):
        return self._evaluate('validation', epoch)

    def test(self):
        return self._evaluate('test')

    def _evaluate(self, mode, epoch=None):
        self.callback_handler.call(f'on_{mode}_epoch_start', self, epoch)

        self.model.eval()
        torch.set_grad_enabled(False)

        dataloader_name = 'valid' if mode == 'validate' else 'test'
        dataloader = self.datamodule.get_dataloader(dataloader_name)

        for batch_idx, batch in enumerate(dataloader):
            self._evaluate_one_batch(mode, batch_idx, batch)

        # let's just use the same loss for both test and validation,
        # as we reset them anyway after the evaluation is done
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        # similarly for metrics
        metrics = self._unwrap_model().valid_metrics.compute()
        self._unwrap_model().valid_metrics.reset()

        torch.set_grad_enabled(True)
        self.model.train()

        self.callback_handler.call(f'on_{mode}_epoch_end', self, epoch, loss, metrics)
        return loss, metrics

    def _evaluate_one_batch(self, mode, batch_idx, batch):
        self.callback_handler.call(f'on_{mode}_batch_start', self, batch_idx, batch)

        X, y = batch
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        logits = self.model(X)
        loss = self._unwrap_model().criterion(logits, y)
        loss = loss.item()

        self.evaluation_loss.update(loss)

        preds = torch.argmax(logits, dim=1)
        self._unwrap_model().valid_metrics.update(preds, y)

        self.callback_handler.call(f'on_{mode}_batch_end', self, batch_idx, batch, loss)

class DifferentiallyPrivateTrainer(Trainer):
    def __init__(
        self,
        *,
        # privacy params
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        clipping_mode: str = 'flat',
        accountant: str = 'prv',
        poisson_sampling: bool = True,
        normalize_clipping: bool = False,
        secure_mode: bool = False,
        target_epsilon: float = 0,
        target_delta: float = 0,
        physical_batch_size: int = 64,
        seed: int = 0,
        **kwargs,
    ):

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clipping_mode = clipping_mode
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.physical_batch_size = physical_batch_size
        self.seed = seed
        self.poisson_sampling = poisson_sampling
        self.normalize_clipping = normalize_clipping

        # setup opacus privacy engine
        self.privacy_engine = opacus.PrivacyEngine(accountant=accountant, secure_mode=secure_mode)

        super().__init__(**kwargs)

    def _has_target_privacy_params(self):
        if not any([self.target_epsilon, self.target_delta]):
            return False

        if self.target_epsilon and not self.target_delta:
            raise RuntimeError('Parameter "target_delta" present, but "target_epsilon" is missing.')

        if self.target_delta and not self.target_epsilon:
            raise RuntimeError('Parameter "target_delta" present, but "target_epsilon" is missing.')

        if self.target_epsilon and self.noise_multiplier:
            raise RuntimeError('Parameter "noise_multiplier" can not be used when target epsilon is given.')

        return True

    def setup(self):
        noise_generator = torch.Generator(device=torch.cuda.current_device())
        if self.seed:
            noise_generator.manual_seed(self.seed)

        self.model = self.model.cuda()

        # let's be distributed by default and wrap the model for Opacus DDP.
        # DifferentiallyPrivateDistributedDataParallel is actually a no-op in Opacus, but
        # let's wrap anyway in case of future api changes. https://opacus.ai/tutorials/ddp_tutorial
        model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(self.model)

        optimizer = self.optimizer
        train_dataloader = self.datamodule.get_dataloader('train')

        # setup differential privacy for the model, optimize, and dataloader
        if self._has_target_privacy_params():
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=self.epochs,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
            )
        else:
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
            )

        # put the DP'ifyed stuff back into Fabric wrappers
        self.model = dp_model
        self.datamodule.set_dataloader('train', dp_dataloader)
        self.optimizer = dp_optimizer

    def get_epsilon(self, delta):
        return self.privacy_engine.get_epsilon(delta)

    def _unwrap_model(self):
        # the model is wrapped inside Opacus, and Opacus distributed.
        # let's unwrap the vanilla model and return it
        return self.model._module.module

    def fit_one_batch(self, batch_idx, batch):
        self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)
        self.optimizer.zero_grad()

        X, y = batch
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        logits = self.model(X)
        loss = self._unwrap_model().criterion(logits, y)
        loss.backward(loss)
        self.optimizer.step()

        loss = loss.item()

        # update the mean loss
        self.train_loss.update(loss)

        # update metrics if there are any
        preds = torch.argmax(logits, dim=1)
        self._unwrap_model().train_metrics(preds, y)

        self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, loss)

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        with BatchMemoryManager(
            data_loader=self.datamodule.get_dataloader('train'),
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:
            # the virtual data loader created by BatchMemoryManager enables us to use larger
            # logical batch sizes that fit in a GPU.
            for batch_idx, batch in enumerate(virtual_dataloader):
                self.fit_one_batch(batch_idx, batch)

        # compute and reset the training loss
        epoch_loss = self.train_loss.compute()
        self.train_loss.reset()

        # compute and reset the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()

        self.callback_handler.call('on_train_epoch_end', self, epoch, epoch_loss, metrics)

class TrainerFactory:
    @staticmethod
    def _get_basic_trainer(configuration: Configuration, hyperparams: Hyperparameters) -> Trainer:
        # setup data, model, and optimizer
        model = ModelFactory.get_model(configuration, hyperparams)
        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)

        # from the pretrained model
        transforms = ModelFactory.get_model_transforms(configuration, hyperparams)

        # now we can create the datamodule that uses the transformations
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams, transforms)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams)
        )

        # instantiate a trainer without dp
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            callback_handler=callback_handler,
            epochs=hyperparams.epochs,
            seed=configuration.seed,
        )

        return trainer

    @staticmethod
    def _get_differentially_private_trainer(configuration: Configuration, hyperparams: Hyperparameters) -> Trainer:
        # Target delta calculation: A common heuristic is to use 1/N', with N'
        # being the size of the dataset rounded up to the nearest power of 10.
        # To avoid too large values of delta, let's pick a somewhat sensible
        # minimum of 1e-5.
        def _round_up_to_nearest_power_of_10(n):
            return 10 ** math.ceil(math.log10(n))

        def _calculate_target_delta(N):
            N_prime = _round_up_to_nearest_power_of_10(N)
            return min(1e-5, 1 / N_prime)

        # setup data, model, and optimizer
        model = ModelFactory.get_model(configuration, hyperparams)
        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)

        # before creating the data module, let's first get the image transformations
        # from the pretrained model
        transforms = ModelFactory.get_model_transforms(configuration, hyperparams)

        # now we can create the datamodule that uses the transformations
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams, transforms)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams)
        )

        # are we given a target epsilon?
        if hyperparams.target_epsilon is not None:
            N = len(datamodule.get_dataloader('train').dataset)
            target_delta = _calculate_target_delta(N)
            target_epsilon = hyperparams.target_epsilon
        else:
            target_delta = None
            target_epsilon = None

        # instantiate a differentialy private trained
        trainer = DifferentiallyPrivateTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            # hypers
            epochs=hyperparams.epochs,
            noise_multiplier=hyperparams.noise_multiplier,
            max_grad_norm=hyperparams.max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            poisson_sampling=configuration.poisson_sampling,
            normalize_clipping=configuration.normalize_clipping,
            # config
            accountant=configuration.accountant,
            secure_mode=configuration.secure_mode,
            clipping_mode=configuration.clipping_mode,
            physical_batch_size=configuration.physical_batch_size,
            seed=configuration.seed,
            callback_handler=callback_handler,
        )

        return trainer

    @staticmethod
    def get_trainer(config_manager: ConfigurationManager) -> Trainer:
        # are we differentially private?
        if config_manager.configuration.privacy:
            return TrainerFactory._get_differentially_private_trainer(config_manager.configuration, config_manager.hyperparams)

        return TrainerFactory._get_basic_trainer(config_manager.configuration, config_manager.hyperparams)

