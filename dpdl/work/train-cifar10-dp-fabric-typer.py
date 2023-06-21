#!/usr/bin/env python3
import os
import warnings
from pathlib import Path

import typer
from typing import Any, List, Optional, Union
from typing_extensions import Annotated

import torch

# You are using a CUDA device ('AMD Radeon Graphics') that has Tensor Cores.
# To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')`
# which will trade-off precision for performance.
# For more details, read
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('medium')

import torchmetrics

import lightning as L
import opacus

# for models
import timm

# use Huggingface datasets
# NB: Set data cache directory with the environment variable HF_DATASETS_CACHE
import datasets

class DataModule():
    def __init__(self, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def criterion(self, logits, y):
        raise(NotImplementedError('Criterion not implemented for class: {self.__class__.__name__}'))

    def accuracy(self, logits, y):
        raise(NotImplementedError('Accuracy not implemented for class: {self.__class__.__name__}'))

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, dataloader):
        self._train_dataloader = dataloader

    @property
    def val_dataloader(self):
        return self._val_dataloader

    @val_dataloader.setter
    def val_dataloader(self, dataloader):
        self._val_dataloader = dataloader

    @property
    def test_dataloader(self):
        return self._test_dataloader

    @test_dataloader.setter
    def test_dataloader(self, dataloader):
        self._test_dataloader = dataloader

class CIFAR10DataModule(DataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 4):
        super().__init__(batch_size, num_workers)

        self.num_classes = 10

        self._criterion = torch.nn.CrossEntropyLoss()
        self._accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)

        self.setup()

    def setup(self):
        dataset = datasets.load_dataset('cifar10', split='train').with_format('torch')
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [45000, 5000])

        self._train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

        self._val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

        test_dataset = datasets.load_dataset('cifar10', split='test').with_format('torch')

        self._test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def criterion(self, logits, y):
        return self._criterion(logits, y)

    def accuracy(self, logits, y):
        preds = torch.argmax(logits, dim=1)
        return self._accuracy(preds, y)

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

class ImageClassificationModel(torch.nn.Module):
    def __init__(self, model_name: str ='resnet18', num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, num_classes=num_classes)

        if not opacus.validators.ModuleValidator.is_valid(self.model):
            self.model = opacus.validators.ModuleValidator.fix(self.model)

    def forward(self, x):
        return self.model(x)

class Trainer:
    def __init__(
        self,

        # essentials
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: DataModule,

        # generic params
        max_epochs: int = 10,
        validation_frequency: int = 1,

        # fabric params
        accelerator: str = 'auto',
        strategy: str = 'auto',
        devices: str = 'auto',
        precision: int = 32,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[L.fabric.loggers.Logger, List[L.fabric.loggers.Logger]]] = None,
        #checkpoint_dir: str = "./checkpoints",
        #checkpoint_frequency: int = 1,
    ):

        self.model = model
        self.optimizer = optimizer
        self.datamodule = datamodule

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.fabric.launch()

        self.max_epochs = max_epochs
        self.validation_frequency = validation_frequency

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
        for epoch in range(self.max_epochs):
            epoch_loss = self.fit_one_epoch(epoch)

            if epoch % self.validation_frequency == 0:
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

    def validate(self, epoch):
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

class DPTrainer(Trainer):
    def __init__(
        self,
        *,
        # privacy params
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        clipping: str = 'flat',
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clipping = clipping

        # setup opacus privacy engine
        self.privacy_engine = opacus.PrivacyEngine()

    def setup(self):
        super().setup()

        # fabric has wrapped the model, optimizer, and module. so let's grab
        # the wrapped modules an DP'ify them.
        model = self.model._forward_module
        optimizer = self.optimizer._optimizer
        train_dataloader = self.datamodule.train_dataloader._dataloader

        # setup differential privacy for the model, optimize, and dataloader
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

        # For some reason
        self.optimizer._optimizer = dp_optimizer

class PrintStateCallback():
    def on_train_start(self, trainer):
        print(f'Starting training for {trainer.max_epochs} epochs.')

    def on_train_end(self, trainer):
        print('Training finished.')

    def on_train_epoch_start(self, trainer, epoch):
        print(f' - Starting epoch {epoch+1}.')

    def on_train_epoch_end(self, trainer, epoch, loss):
        print(f' - Epoch finished, loss: {loss:.3f}.')

    #def on_train_batch_start(self, trainer, batch_idx, batch):
    #    print(f'  - Start processing batch {batch_idx+1}.')

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        print(f'  - Processed batch {batch_idx+1}, loss: {loss:.3f}')

def train(
        max_epochs: Annotated[
            int,
            typer.Option(
                help='Max number of epochs to train',
                rich_help_panel='Training options',
            )
        ] = 10,
        lr: Annotated[
            float,
            typer.Option(
                help='Leraning rate',
                rich_help_panel='Training options',
            )
        ] = 1e-3,
        batch_size: Annotated[
            int,
            typer.Option(
                help='Batch size',
                rich_help_panel='Training options',
            )
        ] = 256,
        num_workers: Annotated[
            int,
            typer.Option(
                help='Number of workers for data loading (per GPU)',
                rich_help_panel='Training options',
            )
        ] = 8,
        model_name: Annotated[
            str,
            typer.Option(
                help='PyTorch Image Models (timm) model name',
                rich_help_panel='Training options',
            )
        ] = 'resnet50',
        privacy: Annotated[
            bool,
            typer.Option(
                help='Enable privacy (Opacus)',
                rich_help_panel='Training options',
            )
        ] = True,
        num_classes: Annotated[
            Optional[int],
            typer.Option(
                help='Number of classes for a classification model',
                rich_help_panel='Classification model options',
            )
        ] = 10,
        accelerator: Annotated[
            Optional[str],
            typer.Option(
                help='Accelerator to use',
                rich_help_panel='Fabric options',
            )
        ] = 'gpu',
        strategy: Annotated[
            Optional[str],
            typer.Option(
                help='Distributed strategy',
                rich_help_panel='Fabric options',
            )
        ] = 'ddp',
        devices: Annotated[
            Optional[int],
            typer.Option(
                help='Number of devices to use',
                rich_help_panel='Fabric options',
            )
        ] = 0,
        precision: Annotated[
            Optional[int],
            typer.Option(
                help='Training precision',
                rich_help_panel='Fabric options',
            )
        ] = 32,
        noise_multiplier: Annotated[
            Optional[float],
            typer.Option(
                help='Noise multiplier',
                rich_help_panel='Opacus options',
            )
        ] = 1.0,
        max_grad_norm: Annotated[
            Optional[float],
            typer.Option(
                help='Maximum gradient norm (clipping)',
                rich_help_panel='Opacus options',
            )
        ] = 1.0,
        clipping: Annotated[
            Optional[str],
            typer.Option(
                help='Clipping mode',
                rich_help_panel='Opacus options',
            )
        ] = 'flat',
    ):

    devices = devices if devices else 'auto'

    # setup data, model, and optimizer
    datamodule = CIFAR10DataModule(num_workers=num_workers, batch_size=batch_size)
    model = ImageClassificationModel(model_name, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    callbacks = [
        PrintStateCallback(),
    ]

    loggers = [
        L.pytorch.loggers.TensorBoardLogger(),
    ]

    if privacy:
        trainer = DPTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            max_epochs=max_epochs,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            clipping=clipping,
            callbacks=callbacks,
            loggers=loggers,
        )
    else:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            max_epochs=max_epochs,
            callbacks=callbacks,
            loggers=loggers,
        )

    trainer.fit()

    print('Done.')

if __name__ == '__main__':
    typer.run(train)
