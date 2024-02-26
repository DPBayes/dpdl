import datasets
import logging
import math
import torch
import torchvision

from collections import defaultdict
from functools import partial
from typing import Tuple

from dpdl.utils import seed_everything
from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)


class DataModule:
    def __init__(self,
        dataset_name: str = 'default-dataset',
        batch_size: int = 64,
        sample_rate: float = 0,
        physical_batch_size: int = 64,
        num_workers: int = 4,
        subset_size: int = None,
        shots: int = None,
        seed: int = 0,
        privacy: bool = True,
        test_size: float = 0.1,
        evaluation_mode: bool = False,
        transforms: torchvision.transforms.transforms.Compose = None,
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.physical_batch_size = physical_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.subset_size = subset_size
        self.shots = shots
        self.privacy = privacy
        self.test_size = test_size
        self.evaluation_mode = evaluation_mode
        self.transforms = transforms

        self._dataloaders = {
            'train': None,
            'valid': None,
            'test': None,
        }

        # for storing mapping from dataset to the dictionary
        # key where the labels are stored.
        self.dataset_label_fields = {}

        # let's not collate batches by default
        self.collate_fn = None

    def initialize(self):
        self._initialize_datasets()

        # we use batch size of -1 to signal full batch
        if self.batch_size == -1:
            self.batch_size = len(self.train_dataset)

        # if sample_rate is set, we set train batch size to int(sample_rate*N)
        if self.sample_rate and self.sample_rate > 0:
            batch_size = int(self.sample_rate * len(self.train_dataset))

            if torch.distributed.get_rank() == 0:
                log.info(f'Sample rate is {self.sample_rate}, setting batch size to: {batch_size}.')

            self.batch_size = batch_size

        self._initialize_dataloaders()

    def get_dataloader(self, name):
        return self._dataloaders.get(name)

    def set_dataloader(self, name, dataloader):
        self._dataloaders[name] = dataloader

    def _initialize_datasets(self):
        # first load the data only rank 0
        if torch.distributed.get_rank() == 0:
            self._load_datasets()
            torch.distributed.barrier()
        else:
            # other ranks will wait ehre
            torch.distributed.barrier()

            # then after all preprocessing is done, load the data
            # on other ranks also. Huggingface datasets has cached
            # everything on disk.
            self._load_datasets()

        if torch.distributed.get_rank() == 0:
            # apply possible tranformations
            self._apply_transforms_to_datasets()
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            self._apply_transforms_to_datasets()

        # if subset of dataset is requested, we'll do stratified sampling
        if self.subset_size is not None and self.subset_size < 1.0:
            self.train_dataset = self._get_stratified_subset(self.train_dataset)
            self.val_dataset = self._get_stratified_subset(self.val_dataset)
        elif self.shots is not None:
            self.train_dataset = self._get_few_shot_subset(self.train_dataset)
            self.val_dataset = self._get_few_shot_subset(self.val_dataset)

    def _load_datasets(self):
        self.train_dataset = datasets.load_dataset(self.dataset_name, split='train')
        self.val_dataset = datasets.load_dataset(self.dataset_name, split='test')
        self.test_dataset = None

        if self.evaluation_mode:
            return

        # otherwise, we load and split the train dataset into train and valid
        split_dataset = self.train_dataset.train_test_split(
            test_size=self.test_size,
            shuffle=True,
            seed=self.seed,
            stratify_by_column=self._get_dataset_label_field(),
        )

        self.test_dataset = self.val_dataset
        self.train_dataset = split_dataset['train']
        self.val_dataset = split_dataset['test']

    def _apply_transforms_to_datasets(self):
        return # no default transforms

    def _initialize_dataloaders(self):
        self._set_generators_and_seed_worker()
        self._set_samplers_and_batch_size()
        self._create_dataloaders()

    def _get_dataset_label_field(self):
        if self.dataset_label_fields is None:
            # this is name of the dictionary key that contains the dataset labels
            return 'label'

        return self.dataset_label_fields.get(self.dataset_name, 'label')

    def _set_generators_and_seed_worker(self):
        self.generator = torch.Generator()
        if self.seed:
            self.generator.manual_seed(self.seed)

        # each dataloader will get a different seed
        def seed_worker(worker_id):
            worker_seed = self.seed + worker_id
            torch.manual_seed(worker_seed)

        self.seed_worker = seed_worker if self.seed else None

    def _create_dataloaders(self):
        self._dataloaders['train'] = torch.utils.data.DataLoader(
            self.train_dataset.with_format('torch'),
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.generator,
            worker_init_fn=self.seed_worker
        )

        self._dataloaders['valid'] = torch.utils.data.DataLoader(
            self.val_dataset.with_format('torch'),
            sampler=self.val_sampler,
            batch_size=self.physical_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

        if self.test_dataset:
            self._dataloaders['test'] = torch.utils.data.DataLoader(
                self.test_dataset.with_format('torch'),
                sampler=self.test_sampler,
                batch_size=self.physical_batch_size,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers
            )

    def _set_samplers_and_batch_size(self):
        # for the DP case, Opacus handles distributed for us. otherwise, we need
        # to use distributedsampler and divide the batch size by number of replicas
        if not self.privacy:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset.with_format('torch')
            )
            self.batch_size //= torch.distributed.get_world_size()
        else:
            self.train_sampler = None

        # we will validate and test only on rank 0
        self.val_sampler, self.test_sampler = None, None


    def _get_stratified_subset(self, dataset):
        split_dataset = dataset.train_test_split(
            test_size=self.subset_size,
            seed=self.seed,
            stratify_by_column=self._get_dataset_label_field(),
        )

        return split_dataset['test']

    def _get_few_shot_subset(self, dataset):
        test_size = self.shots * self.num_classes

        split_dataset = dataset.train_test_split(
            test_size=test_size,
            seed=self.seed,
            stratify_by_column=self._get_dataset_label_field(),
        )

        return split_dataset['test']

class ImageDataModule(DataModule):
    def __init__(
        self,
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.dataset_label_fields = {
            'cifar100': 'fine_label',
        }
        self.collate_fn = partial(self._collate_fn, self._get_dataset_label_field())

    def _apply_transforms_to_datasets(self):
        def _apply_transforms(transforms, examples):
            examples['img'] = [transforms(image) for image in examples['img']]
            return examples

        if self.transforms:
            transforms_func = partial(_apply_transforms, self.transforms)

            # XXX: For some reason num_proc > 1 started causing hangs.
            #      Also default batch size (1000) seems to hang, even
            #      with one process.
            self.train_dataset = self.train_dataset.map(transforms_func, num_proc=1, batched=True, batch_size=256)
            self.val_dataset = self.val_dataset.map(transforms_func, num_proc=1, batched=True, batch_size=256)
            if self.test_dataset:
                self.test_dataset = self.test_dataset.map(transforms_func, num_proc=1, batched=True, batch_size=256)

    @staticmethod
    def _collate_fn(label_field, batch):
        B = len(batch)
        C, H, W = batch[0]['img'].shape

        images = torch.empty((B, C, H, W))
        labels = torch.empty(B, dtype=torch.long)

        for i in range(B):
            images[i] = batch[i]['img']
            labels[i] = batch[i][label_field]

        return images, labels

class DataModuleFactory:
    @staticmethod
    def get_datamodule(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        transforms: torchvision.transforms.transforms.Compose,
    ) -> DataModule:
        datamodule = ImageDataModule(
            dataset_name=configuration.dataset_name,
            num_workers=configuration.num_workers,
            physical_batch_size=configuration.physical_batch_size,
            subset_size=configuration.subset_size,
            shots=configuration.shots,
            seed=configuration.seed,
            batch_size=hyperparams.batch_size,
            sample_rate=hyperparams.sample_rate,
            privacy=configuration.privacy,
            transforms=transforms,
            evaluation_mode=configuration.evaluation_mode,
        )

        datamodule.initialize()

        return datamodule

