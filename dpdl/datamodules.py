# NB: Set datasets cache directory with the environment variable HF_DATASETS_CACHE
import datasets
import opacus
import torch

from functools import partial

from dpdl.utils import seed_everything

class DataModule():
    def __init__(
        self,
        batch_size: int = 64,
        physical_batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.physical_batch_size = physical_batch_size
        self.num_workers = num_workers
        self.seed = seed

        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

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
    def __init__(self, *, image_size=None, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = 10
        self.image_size = image_size

        self.setup()

    def setup(self):
        self._initialize_datasets()
        self._initialize_dataloaders()

    def _initialize_dataloaders(self):
        generator = torch.Generator()
        if self.seed:
            generator.manual_seed(self.seed)

        def seed_worker(worker_id):
            seed_everything(self.seed)

        self._train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset.with_format('torch'),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker if self.seed else None,
        )

        self._val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset.with_format('torch'),
            batch_size=self.physical_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )

        self._test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset.with_format('torch'),
            batch_size=self.physical_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def _initialize_datasets(self):
        # only load the data in single process on each node
        if torch.distributed.get_rank() == 0:
            dataset = datasets.load_dataset('cifar10', split='train')
            test_dataset = datasets.load_dataset('cifar10', split='test')
            torch.distributed.barrier()
        else:
            # wait for local rank 0 to load the datasets
            torch.distributed.barrier()

            dataset = datasets.load_dataset('cifar10', split='train')
            test_dataset = datasets.load_dataset('cifar10', split='test')

        if self.image_size:
            transform = partial(self._resize_transform, self.image_size)
            dataset = dataset.map(transform, batched=True)

        # create train and validation splits
        split_dataset = dataset.train_test_split(test_size=0.1, shuffle=False)
        self.train_dataset = split_dataset['train']
        self.val_dataset = split_dataset['test']

        test_dataset = datasets.load_dataset('cifar10', split='test')
        if self.image_size:
            transform = partial(self._resize_transform, self.image_size)
            test_dataset = test_dataset.map(transform, batched=True)

        self.test_dataset = test_dataset

    @staticmethod
    def _resize_transform(image_size, examples):
        examples['img'] = [image.resize(image_size) for image in examples['img']]
        return examples

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

class DataModuleFactory():
    @staticmethod
    def get_datamodule(configuration: dict, hyperparams: dict) -> DataModule:
        datamodule = CIFAR10DataModule(
            num_workers=configuration['num_workers'],
            batch_size=hyperparams['batch_size'],
            physical_batch_size=configuration['physical_batch_size'],
            seed=configuration['seed'],
            image_size=(224, 224),
        )

        return datamodule

