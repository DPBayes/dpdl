import logging
import datasets
import torch
import torchvision
from functools import partial
from typing import Tuple
from dpdl.utils import seed_everything
from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)

class DataModule:
    def __init__(self,
        dataset_name: str = 'default-dataset',
        batch_size: int = 64,
        physical_batch_size: int = 64,
        num_workers: int = 4,
        subset_size: float = None,
        seed: int = 0,
        privacy: bool = True,
        test_size: float = 0.1,
        evaluation_mode: bool = False,
        transforms: torchvision.transforms.transforms.Compose = None,
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.physical_batch_size = physical_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.subset_size = subset_size
        self.privacy = privacy
        self.test_size = test_size
        self.evaluation_mode = evaluation_mode
        self.transforms = transforms

        self._dataloaders = {
            'train': None,
            'valid': None,
        }

        # for storing mapping from dataset to the dictionary
        # key where the labels are stored.
        self.dataset_label_fields = {}

        # let's not collate batches by default
        self.collate_fn = None

    def get_dataloader(self, name):
        # are dataloaders initialized?
        if not self._dataloaders['train']:
            self._initialize_datasets()
            self._initialize_dataloaders()

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

        # if subset of datasetr is requested, we'll do stratified sampling
        if self.subset_size:
            self.train_dataset = self._get_stratified_subset(self.train_dataset)
            self.val_dataset = self._get_stratified_subset(self.val_dataset)

    def _load_datasets(self):
        self.train_dataset = datasets.load_dataset(self.dataset_name, split='train')
        self.val_dataset = datasets.load_dataset(self.dataset_name, split='test')

        # apply possible tranformations
        self._apply_transforms_to_datasets()

        if self.evaluation_mode:
            return

        # otherwise, we load and split the train dataset into train and valid
        split_dataset = self.train_dataset.train_test_split(
            test_size=self.test_size,
            shuffle=True,
            seed=self.seed,
        )
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
        if self.transforms:
            # XXX: For some reason num_proc > 1 started causing hangs.
            #      Also default batch size (1000) seems to hang, even
            #      with one process.
            self.train_dataset = self.train_dataset.map(self._apply_transforms, num_proc=1, batched=True, batch_size=256)
            self.val_dataset = self.val_dataset.map(self._apply_transforms, num_proc=1, batched=True, batch_size=256)

    def _apply_transforms(self, examples):
        # log an empty line to see progress in logs,
        # otherwise something buffer the output. flushing
        # of the log handlers did not seem to help.

        log.info('')

        examples['img'] = [self.transforms(image) for image in examples['img']]
        return examples

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
            seed=configuration.seed,
            batch_size=hyperparams.batch_size,
            privacy=configuration.privacy,
            transforms=transforms,
            evaluation_mode=configuration.evaluation_mode,
        )

        return datamodule

