#!/usr/bin/env python3
import os
from pathlib import Path

from typing import Callable, Iterable

import torch
import torchmetrics

from lightning.pytorch.cli import LightningCLI

import lightning as L

import opacus

# models
import timm

# use Huggingface datasets
import datasets

# NB: Set data cache directory with environment variable HF_DATASETS_CACHE

# You are using a CUDA device ('AMD Radeon Graphics') that has Tensor Cores.
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.
# For more details, read
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')

class MyHuggingFaceCIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        train = datasets.load_dataset('cifar10', split='train').with_format('torch')
        self.train, self.val = torch.utils.data.random_split(train, [45000, 5000])

    @staticmethod
    def collate_fn(batch):
        B = len(batch)
        H, W, C = batch[0]['img'].shape

        images = torch.empty((B, C, H, W))
        labels = torch.empty(B, dtype=torch.long)

        for i in range(B):
            images[i] = batch[i]['img'].permute(2, 0, 1)
            labels[i] = batch[i]['label']

        return images, labels

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

class MyHuggingFaceCIFAR10DataModuleDP(MyHuggingFaceCIFAR10DataModule):
    def train_dataloader(self):
        # check if we should do DDP
        distributed = isinstance(
            self.trainer.model,
            (
                opacus.distributed.DifferentiallyPrivateDistributedDataParallel,
                torch.nn.parallel.DistributedDataParallel,
            )
        )

        # For DP we need dataloader with Poisson uniform sampling
        # 2b - https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb
        dataloader = super().train_dataloader()
        return opacus.data_loader.DPDataLoader.from_data_loader(dataloader, distributed=distributed)

class CIFAR10ClassificationModel(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable[[Iterable], torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        num_classes: int = 10,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr

        self.model = timm.create_model('resnet18', num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        # HACK(ish): jsonargparse does not seem to flexible enough to support learning rate in init!
        for g in optimizer.param_groups:
            g['lr'] = self.lr

        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        self.accuracy(preds, y)
        self.log('train_acc', self.accuracy, on_epoch=True, on_step=False, prog_bar=True)

        loss = self.criterion(preds, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)

        self.accuracy(preds, y)
        self.log('valid_acc', self.accuracy, on_epoch=True, on_step=False, prog_bar=True)

        loss = self.criterion(preds, y)
        self.log('valid_loss', loss, on_epoch=True, on_step=False, prog_bar=True)

class CIFAR10ClassificationModelDP(CIFAR10ClassificationModel):
    def __init__(
        self,
        optimizer: Callable[[Iterable], torch.optim.Optimizer] = torch.optim.Adam,
        lr: float = 1e-3,
        num_classes: int = 10,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = None, # defaults to training dataset size
        clipping: str = 'flat',
    ):
        """
        Add privacy-related responsibilities to the parent class.

        - Model is wrapped to also compute per sample gradients.
        - Optimizer is now responsible for gradient clipping and adding noise to the gradients.
        - DataLoader is updated to perform Poisson sampling.

        Args:
            optimizer: Optimizer to be used for training
            lr: Learning rate for the training.
            num_classes: Number of classes for classification.
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
                (How much noise to add)
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            delta: The delta to be used when calculating epsilon. Default: 1/N
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer" or "adaptive").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, per layer clipping sets individual norms for
                every parameter tensor, and adaptive clipping updates clipping bound per iteration.
                Flat clipping is usually preferred, but using per layer clipping in combination
                with distributed training can provide notable performance gains.
        """

        super().__init__()

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta

        self.privacy_engine = opacus.PrivacyEngine()

        # some layers are not supported for DP
        # automatically fix unsupported layers (e.g. BatchNorm -> GroupNorm)
        if not opacus.validators.ModuleValidator.is_valid(self.model):
            self.model = opacus.validators.ModuleValidator.fix(self.model)

        # For DP we need per-sample gradients that Opacus can handle for us.
        # 2a - https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb
        dp_model = opacus.GradSampleModule(self.model)
        self.model = dp_model

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()
        dataloader = self.trainer.datamodule.train_dataloader()

        # For DP we need optimizer that clips the per-sample gradients and adds noise to the gradient.
        # 2c - https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb
        # 3d - https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb
        _, dp_optimizer, _ = self.privacy_engine.make_private(
            module = self,
            optimizer = optimizer,
            data_loader = dataloader,
            noise_multiplier = self.noise_multiplier,
            max_grad_norm = self.max_grad_norm,
        )

        return dp_optimizer

    def _get_epsilon(self):
        if not self.delta:
            N = len(self.trainer.train_dataloader.dataset)
            delta = 1/N
        else:
            delta = self.delta

        epsilon = self.privacy_engine.get_epsilon(delta)

        return epsilon

class LogEpsilonCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epsilon = trainer.model._get_epsilon()
        pl_module.log('epsilon', epsilon, prog_bar=True)

class LogDPModulesCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print('-------------------------------------------------------- [DP STUFF START]')
        print(f'Model: {trainer.model.model.__class__.__name__}')
        print(f'Dataloader: {trainer.train_dataloader.__class__.__name__}')
        print(f'Optimizer: {pl_module.optimizers().__class__}')
        print('-------------------------------------------------------- [DP STUFF END]')

class AssertDPModulesCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        assert isinstance(trainer.model.model, opacus.GradSampleModule), f'Model must be instance of opacus.GradSampleModule, we got: {trainer.model.model.__class__}'
        assert isinstance(trainer.train_dataloader, opacus.data_loader.DPDataLoader), f'Data loader must be instance of opacus.data_loader.DPDataLoader, we got: {trainer.train_dataloader.__class__}'
        assert isinstance(pl_module.optimizers(), opacus.optimizers.DPOptimizer), f'Optimizer must be instance of opacus.optimizers.DPOptimizer, we got: {pl_module.optimizers().__class__}'

    on_train_epoch_end = on_train_epoch_start

def main():
    callbacks = [
        LogEpsilonCallback(),
        AssertDPModulesCallback(),
        LogDPModulesCallback(),
    ]

    cli = LightningCLI(
        CIFAR10ClassificationModelDP,
        MyHuggingFaceCIFAR10DataModuleDP,
        trainer_defaults = {
            'max_epochs': 10,
            'callbacks' : callbacks,
        },
        save_config_kwargs = {
            'overwrite': True
        },
        auto_configure_optimizers = False,
    )
    print('Done.')

if __name__ == '__main__':
    main()
