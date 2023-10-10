# NB: Set datasets cache directory with the environment variable HF_DATASETS_CACHE
import logging
import datasets
import torch

from functools import partial
from typing import Tuple

from dpdl.utils import seed_everything
from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)

class DataModule:
    def __init__(
        self,
        batch_size: int = 64,
        physical_batch_size: int = 64,
        num_workers: int = 4,
        subset_size: float = None,
        seed: int = 0,
        privacy: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.physical_batch_size = physical_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.subset_size = subset_size
        self.privacy = privacy

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

class ImageDataModule(DataModule):
    def __init__(
        self,
        *,
        dataset_name: str = 'cifar10',
        num_classes: int = 10,
        image_size: Tuple[int, int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = 10
        self.image_size = image_size
        self.dataset_name = dataset_name

        self.dataset_label_fields = {
            'cifar100': 'fine_label',
        }

        self._initialize_datasets()
        self._initialize_dataloaders()

    def _initialize_dataloaders(self):
        generator = torch.Generator()
        if self.seed:
            generator = generator.manual_seed(self.seed)

        def seed_worker(worker_id):
            seed_everything(self.seed)

        # for the DP case, Opacus handles distributed for use.
        # otherwise, we need to use distributedsampler and divide
        # the batch size by number of replicas
        if not self.privacy:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset.with_format('torch')
            )

            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset.with_format('torch')
            )

            test_sampler = torch.utils.data.distributed.DistributedSampler(
                self.test_dataset.with_format('torch')
            )

            batch_size = self.batch_size // torch.distributed.get_world_size()
        else:
            train_sampler, val_sampler, test_sampler = None, None, None
            batch_size = self.batch_size

        self._train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset.with_format('torch'),
            sampler=train_sampler,
            batch_size=batch_size,
            collate_fn=partial(self._collate_fn, self._get_dataset_label_field()),
            num_workers=self.num_workers,
            pin_memory=True,
            generator=generator,
            worker_init_fn=seed_worker if self.seed else None,
        )

        self._val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset.with_format('torch'),
            sampler=val_sampler,
            batch_size=self.physical_batch_size,
            collate_fn=partial(self._collate_fn, self._get_dataset_label_field()),
            num_workers=self.num_workers,
        )

        self._test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset.with_format('torch'),
            sampler=test_sampler,
            batch_size=self.physical_batch_size,
            collate_fn=partial(self._collate_fn, self._get_dataset_label_field()),
            num_workers=self.num_workers,
        )

        self._train_and_valid_dataloader = torch.utils.data.DataLoader(
            self.train_and_valid_dataset.with_format('torch'),
            sampler=train_sampler,
            batch_size=batch_size,
            collate_fn=partial(self._collate_fn, self._get_dataset_label_field()),
            num_workers=self.num_workers,
            pin_memory=True,
            generator=generator,
            worker_init_fn=seed_worker if self.seed else None,
        )

    def _get_dataset_label_field(self):
        if self.dataset_name in self.dataset_label_fields:
            label_field = self.dataset_label_fields[self.dataset_name]
        else:
            label_field = 'label'

        return label_field

    def _get_stratified_subset(self, dataset):
        generator = torch.Generator()
        if self.seed:
            generator.manual_seed(self.seed)

        label_field = self._get_dataset_label_field()
        labels = torch.tensor(dataset[label_field])
        unique_labels = labels.unique()

        sampled_indices = []
        for label in unique_labels:
            # find the indices of the dataset where the current label is present
            label_indices = torch.where(labels == label)[0]

            # determine the number of samples needed for this label
            num_samples = int(len(label_indices) * self.subset_size)

            # generate a random permutation of the label indices
            random_indices = torch.randperm(len(label_indices), generator=generator)

            # select the first 'num_samples' indices
            chosen_indices = random_indices[:num_samples]

            # retrieve the dataset indices corresponding to the chosen label indices
            chosen_dataset_indices = label_indices[chosen_indices].tolist()

            # extend the final list of sampled indices
            sampled_indices.extend(chosen_dataset_indices)

        # generate a random permutation for the final list of sampled indices
        random_order = torch.randperm(len(sampled_indices), generator=generator)

        # reorder the sampled indices randomly
        sampled_indices = torch.tensor(sampled_indices)[random_order].tolist()

        return dataset.select(sampled_indices)

    def _initialize_datasets(self):
        # first load the data only in rank 0
        if torch.distributed.get_rank() == 0:
            self._load_and_preprocess_datasets()

        # other ranks will wait here
        torch.distributed.barrier()

        # then after all preprocessing is done, load the data
        # on other ranks also. Huggingface datasets has cached
        # everything on disk.
        self._load_and_preprocess_datasets()

    def _load_and_preprocess_datasets(self):
        # load datasets and cache to disk
        train_dataset = datasets.load_dataset(self.dataset_name, split='train')
        test_dataset = datasets.load_dataset(self.dataset_name, split='test')

        # apply stratified sampling if subset_size is set
        if self.subset_size:
            train_dataset = self._get_stratified_subset(train_dataset)
            test_dataset = self._get_stratified_subset(test_dataset)

        # apply image resizing if image_size is set
        if self.image_size:
            transform = partial(self._resize_transform, self.image_size)
            train_dataset = train_dataset.map(transform, num_proc=self.num_workers, batched=True)
            test_dataset = test_dataset.map(transform, num_proc=self.num_workers, batched=True)

        # split the train dataset into train and validation sets
        split_dataset = train_dataset.train_test_split(
            test_size=0.1,
            shuffle=False,
            seed=self.seed,
        )
        self.train_dataset = split_dataset['train']
        self.val_dataset = split_dataset['test']

        self.test_dataset = test_dataset

        # for the last training round, let's train with the training
        # and also also the validation dataset
        # NB: Implementing this this way, as I've been thinking about that
        #     maybe we should split the test dataset into training and validation,
        #     instead of splitting the train dataset into testing and validation.
        self.train_and_valid_dataset = datasets.concatenate_datasets([
            self.train_dataset,
            self.val_dataset,
        ])

    @staticmethod
    def _resize_transform(image_size, examples):
        examples['img'] = [image.resize(image_size) for image in examples['img']]
        return examples

    @staticmethod
    def _collate_fn(label_field, batch):
        B = len(batch)
        H, W, C = batch[0]['img'].shape

        images = torch.empty((B, C, H, W))
        labels = torch.empty(B, dtype=torch.long)

        for i in range(B):
            images[i] = batch[i]['img'].permute(2, 0, 1)
            labels[i] = batch[i][label_field]

        return images, labels

class DataModuleFactory:
    @staticmethod
    def get_datamodule(configuration: Configuration, hyperparams: Hyperparameters) -> DataModule:
        datamodule = ImageDataModule(
            dataset_name=configuration.dataset_name,
            num_workers=configuration.num_workers,
            physical_batch_size=configuration.physical_batch_size,
            subset_size=configuration.subset_size,
            seed=configuration.seed,
            batch_size=hyperparams.batch_size,
            privacy=configuration.privacy,
            image_size=(224, 224),
        )

        return datamodule

