#!/usr/bin/env python3
import os
from pathlib import Path

import torch
import torchmetrics

from lightning.pytorch.cli import LightningCLI
import lightning as L

from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.lightning import DPLightningDataModule

# models
from timm import create_model

# use Huggingface datasets
from datasets import load_dataset, Dataset

# download Huggingface datasets to custom directory if requested
if DATA_DIR := os.environ.get('HUGGINGFACE_DATA_DIR'):
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(DATA_DIR)

class MyHuggingFaceCIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def setup(self, stage: str):
        train = load_dataset('cifar10', split='train').with_format('torch')

        #self.train, self.val = torch.utils.data.random_split(train, [45000, 5000])
        _, self.train, self.val = torch.utils.data.random_split(train, [35000, 10000, 5000])
        self.test = load_dataset('cifar10', split='test').with_format('torch')

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
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn)

class TorchvisionCIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def setup(self, stage: str):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        self.train, self.val = torch.utils.data.random_split(train, [45000, 5000])

        self.test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)

class CIFAR10ClassificationModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model('resnet18', num_classes=10)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

def main():
    data = MyHuggingFaceCIFAR10DataModule(batch_size=2)
    #dp_data = DPLightningDataModule(data)
    dp_data = data

    model = LitModel()
    # compiled_model = torch.compile(model)

    trainer = L.Trainer(
        max_epochs=10,
        enable_model_summary=False,
    )
    trainer.fit(model, dp_data)

    trainer.test(model, data)
    #trainer.test(model, dp_data)  # identical


def cli_main():
    cli = LightningCLI(
        CIFAR10ClassificationModel,
        MyHuggingFaceCIFAR10DataModule,
        trainer_defaults={
            'max_epochs': 10,
            'enable_model_summary': False,
        },
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path='best', datamodule=cli.datamodule)


if __name__ == '__main__':
    if os.environ.get('LIGHTNING_VANILLA') == 'true':
        main()
    else:
        cli_main()
