import lightning as L
import torch
import opacus

from typing import Any, List, Optional, Union

from .datamodules import DataModule

# You are using a CUDA device ('AMD Radeon Graphics') that has Tensor Cores.
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.
# For more details, read
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')

class Trainer:
    def __init__(
        self,

        # essentials
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: DataModule,
        fabric: L.Fabric = None,

        # generic params
        epochs: int = 10,
        validation_frequency: int = 1,
        #checkpoint_dir: str = "./checkpoints",
        #checkpoint_frequency: int = 1,
    ):

        if not fabric:
            raise(RuntimeError('Initialized fabric not passed to {self.__class__.__name}.'))

        self.model = model
        self.optimizer = optimizer
        self.datamodule = datamodule
        self.epochs = epochs
        self.validation_frequency = validation_frequency
        self.fabric = fabric

    def setup(self):
        # call fabric to setup possible distributed training
        model, optimizer = self.fabric.setup(self.model, self.optimizer)
        train_dataloader, val_dataloader, test_dataloader = self.fabric.setup_dataloaders(
            self.datamodule.train_dataloader,
            self.datamodule.val_dataloader,
            self.datamodule.test_dataloader,
        )

        self.model = model
        self.optimizer = optimizer

        self.datamodule.train_dataloader = train_dataloader
        self.datamodule.val_dataloader = val_dataloader
        self.datamodule.test_dataloader = test_dataloader

    def fit(self):
        self.setup()

        self.fabric.call('on_train_start', self)
        for epoch in range(self.epochs):
            epoch_loss = self.fit_one_epoch(epoch)

            if self.validation_frequency and epoch % self.validation_frequency == 0:
                self.validate(epoch)

        self.fabric.call('on_train_end', self)

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.fabric.call('on_train_epoch_start', self, epoch)

        total_loss = 0
        for batch_idx, batch in enumerate(self.datamodule.train_dataloader):
            batch_loss = self.fit_one_batch(batch_idx, batch)
            total_loss = total_loss + batch_loss

        epoch_loss = total_loss / (batch_idx + 1)
        self.fabric.log('Train/loss', epoch_loss, epoch)

        self.fabric.call('on_train_epoch_end', self, epoch, epoch_loss)

        return epoch_loss

    def fit_one_batch(self, batch_idx, batch):
        self.fabric.call('on_train_batch_start', self, batch_idx, batch)

        X, y = batch
        self.optimizer.zero_grad()
        logits = self.model(X)

        loss = self.datamodule.criterion(logits, y)
        self.fabric.backward(loss)
        self.optimizer.step()

        loss = loss.item()

        self.fabric.call('on_train_batch_end', self, batch_idx, batch, loss)

        return loss

    def validate(self, epoch=None):
        self.model.eval()
        torch.set_grad_enabled(False)

        total_loss = 0
        self.fabric.call('on_validation_epoch_start', self, epoch)
        for batch_idx, batch in enumerate(self.datamodule.val_dataloader):
            batch_loss = self.validate_one_batch(batch_idx, batch)
            total_loss = total_loss + batch_loss

        valid_loss = total_loss / (batch_idx + 1)
        self.fabric.call('on_validation_epoch_end', self, epoch, valid_loss)

        torch.set_grad_enabled(True)
        self.model.train()

        return valid_loss

    def validate_one_batch(self, batch_idx, batch):
        self.fabric.call('on_validation_batch_start', self, batch_idx, batch)

        X, y = batch
        logits = self.model(X)
        loss = self.datamodule.criterion(logits, y)

        loss = loss.item()
        self.fabric.call('on_validation_batch_end', self, batch_idx, batch, loss)

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
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clipping = clipping
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        # setup opacus privacy engine
        self.privacy_engine = opacus.PrivacyEngine(accountant=accountant, secure_mode=secure_mode)

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
        # call super class to initialize fabric
        super().setup()

        # fabric has wrapped the model, optimizer, and module. so let's grab
        # the wrapped modules an DP'ify them.
        model = self.model._forward_module
        optimizer = self.optimizer._optimizer
        train_dataloader = self.datamodule.train_dataloader._dataloader

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
            )
        else:
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping,
            )

        # are we distributed?
        if self.fabric.world_size > 1:
            # DifferentiallyPrivateDistributedDataParallel is actually a no-op in Opacus, but
            # let's wrap anyway in case of future api changes. https://opacus.ai/tutorials/ddp_tutorial
            dp_model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(dp_model)

        # put the DP'ifyed stuff back into Fabric wrappers
        self.model._forward_module = dp_model
        self.datamodule.train_dataloader._dataloader = dp_dataloader
        self.optimizer._optimizer = dp_optimizer

    def get_epsilon(self, delta):
        return self.privacy_engine.get_epsilon(delta)
