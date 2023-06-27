import opacus
import torch
import torchmetrics

# for models
import timm

# use Huggingface datasets
# NB: Set data cache directory with the environment variable HF_DATASETS_CACHE
import datasets

from functools import partial

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
    def __init__(self, batch_size: int = 64, num_workers: int = 4, image_size=None):
        super().__init__(batch_size, num_workers)

        self.num_classes = 10
        self.image_size = image_size

        self._criterion = torch.nn.CrossEntropyLoss()
        self._accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)

        self.setup()

    def setup(self):
        dataset = datasets.load_dataset('cifar10', split='train')

        if self.image_size:
            transform = partial(self._resize_transform, self.image_size)
            dataset = dataset.map(transform, batched=True)

        train_dataset, val_dataset = torch.utils.data.random_split(dataset.with_format('torch'), [45000, 5000])

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

        test_dataset = datasets.load_dataset('cifar10', split='test')
        if self.image_size:
            transform = partial(self._resize_transform, self.image_size)
            test_dataset = test_dataset.map(transform, batched=True)

        self._test_dataloader = torch.utils.data.DataLoader(
            test_dataset.with_format('torch'),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    @staticmethod
    def _resize_transform(image_size, examples):
        examples['img'] = [image.resize(image_size) for image in examples['img']]
        return examples

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

