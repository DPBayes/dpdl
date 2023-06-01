#!/usr/bin/env python3
import os
from pathlib import Path

import torch
import torchmetrics

from lightning.pytorch.cli import LightningCLI
import lightning as L

import opacus

# models
import timm

# use Huggingface datasets
import datasets

# download Huggingface datasets to custom directory if requested
#if DATA_DIR := os.environ.get('HUGGINGFACE_DATA_DIR'):
#    datasets.config.DOWNLOADED_DATASETS_PATH = Path(DATA_DIR)
#    datasets.config.DEFAULT_HF_DATASETS_CACHE = = Path(DATA_DIR)
# NB: These can be set with an environment variable HF_DATASETS_CACHE

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

class CIFAR10ClassificationModel(L.LightningModule):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.model = timm.create_model('resnet18', num_classes=num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

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
    def __init__(self):
        super().__init__()

        # resnet18 has batchnorm layers that we can't use with DP-SGD
        dp_model = opacus.validators.ModuleValidator.fix(self.model)

        # For DP we need per-sample gradients that Opacus can handle for us.
        # 2a - https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb
        dp_model = opacus.GradSampleModule(dp_model)
        #import ipdb
        #ipdb.set_trace()
        #dp_model.forbid_grad_accumulation()
        #dp_model.register_forward_pre_hook(opacus.privacy_engine.forbid_accumulation_hook)

        self.model = dp_model

class MyHuggingFaceCIFAR10DataModuleDP(MyHuggingFaceCIFAR10DataModule):
    def train_dataloader(self):
        # For DP we need dataloader with Poisson uniform sampling
        # 2b - https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb
        # XXX: Distributed dataloader
        dataloader = super().train_dataloader()
        return opacus.data_loader.DPDataLoader.from_data_loader(dataloader)

class SetDPOptimizerCallback(L.pytorch.callbacks.Callback):
    def __init__(self, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        super().__init__()
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def on_train_start(self, trainer, pl_module):
        if len(trainer.optimizers) > 1:
            raise NotImplementedError('Support for more than one optimizer not implemented.')

        optimizer = trainer.optimizers[0]

        # For DP we need optimizer that clips the per-sample gradients and adds noise to the gradient.
        # 2c - https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb
        sample_rate = trainer.train_dataloader.sample_rate
        expected_batch_size = int(len(trainer.train_dataloader.dataset) * sample_rate)

        dp_optimizer = opacus.optimizers.DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=expected_batch_size,
        )

        # Setup privacy accountant.
        # 3d - https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb
        accountant = opacus.accountants.RDPAccountant()
        dp_optimizer.attach_step_hook(accountant.get_optimizer_hook_fn(sample_rate=sample_rate))

        trainer.optimizers = [dp_optimizer]

def cli_main():

    callbacks = [
        SetDPOptimizerCallback(noise_multiplier=1.0, max_grad_norm=1.0),
    ]

    cli = LightningCLI(
        CIFAR10ClassificationModelDP,
        MyHuggingFaceCIFAR10DataModuleDP,
        trainer_defaults = {
            'max_epochs': 10,
            'callbacks': callbacks,
        },
        save_config_kwargs = {
            'overwrite': True
        },
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    if os.environ.get('LIGHTNING_VANILLA') == 'true':
        main()
    else:
        cli_main()
