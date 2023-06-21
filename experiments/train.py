#!/usr/bin/env python3
import os
import warnings
from pathlib import Path

import typer
from typing import Any, List, Optional, Union
from typing_extensions import Annotated

import torch
import torchmetrics

import lightning as L
import opacus

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
        trainer = DifferentiallyPrivateTrainer(
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
