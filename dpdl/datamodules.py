import datasets
import logging
import math
import os
import pathlib
import torch
import torchvision
import numpy as np
import tensorflow_datasets as tfds

from collections import defaultdict
from functools import partial
from typing import Tuple

from dpdl.utils import seed_everything
from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)


def load_tfds_dataset(dataset_name):
    builder = tfds.builder(dataset_name, file_format='array_record')
    tfds_info = builder.info
    tfds_dataset = tfds.data_source(dataset_name)
    return tfds_dataset, tfds_info

def convert_tfds_to_huggingface(dataset_name):
    tfds_dataset, tfds_info = load_tfds_dataset(dataset_name)
    hf_datasets = {}

    # Define the Hugging Face dataset features based on the TFDS features
    hf_features = datasets.Features({
        'image': datasets.Image(),
        'label': datasets.ClassLabel(
            num_classes=tfds_info.features['label'].num_classes,
            names=tfds_info.features['label'].names,
        ),
    })

    for split_name, split_dataset in tfds_dataset.items():
        data = {feature: [] for feature in hf_features.keys()}
        for item in split_dataset:
            for feature, value in item.items():
                if not feature in hf_features:
                    continue

                data[feature].append(value)

        # Create Hugging Face dataset for the current split, specifying the features
        hf_datasets[split_name] = datasets.Dataset.from_dict(data, features=hf_features)

    return datasets.DatasetDict(hf_datasets)

class DataModule:
    def __init__(self,
        dataset_name: str = 'default-dataset',
        dataset_source: str = 'huggingface',
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
        self.dataset_source = dataset_source
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

    def _default_collate_fn(batch):
        # default collate is a no-op
        return batch

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
            # NB: for few-shot, we'll keep the validation dataset intact
            self.train_dataset = self._get_few_shot_subset(self.train_dataset)

    def _load_datasets(self):
        if self.dataset_source == 'huggingface':
            if torch.distributed.get_rank() == 0:
                log.info(f'Loading dataset "{self.dataset_name}" from Huggingface datasets.')

            dataset_splits = datasets.load_dataset(self.dataset_name)

        elif self.dataset_source == 'tensorflow':
            if torch.distributed.get_rank() == 0:
                log.info(f'Loading dataset "{self.dataset_name}" from Tensorflow datasets.')

            tfds_cache_fpath = self._get_tfds_cache_fpath()
            if tfds_cache_fpath.exists():
                dataset_splits = datasets.DatasetDict.load_from_disk(tfds_cache_fpath)
            else:
                dataset_splits = convert_tfds_to_huggingface(self.dataset_name)
                dataset_splits.save_to_disk(tfds_cache_fpath)
        else:
            raise ValueError(f'Unsupported dataset source: {self.dataset_source}')

        # Set dataset label fields based on the training split
        self._set_dataset_label_fields(dataset_splits['train'])

        # Check if there's a validation split available
        has_validation_split = 'validation' in dataset_splits

        # Process datasets (split/train/validate as necessary)
        self._load_huggingface_dataset(dataset_splits, has_validation_split)

    def _get_tfds_cache_fpath(self):
        # Get the base cache directory
        hf_cache_base = os.environ.get('HF_DATASETS_CACHE', '~/.cache/huggingface/datasets')
        converted_tfds_cache = pathlib.Path(f'{hf_cache_base}/converted_tfds_datasets')

        # Ensure the cache directory exists
        converted_tfds_cache.mkdir(exist_ok=True)

        # Construct the cache file name based on dataset_source and dataset_name
        tfds_cache_fname = f'dpdl_{self.dataset_source}_{self.dataset_name}_cache'

        # Construct the full path to the cache file
        cache_file_path = converted_tfds_cache / tfds_cache_fname

        return cache_file_path

    def _load_huggingface_dataset(self, dataset_splits, has_validation_split):
        # Always use the test set as-is
        self.test_dataset = dataset_splits.get('test', None)

        if self.evaluation_mode:
            if has_validation_split:
                # Combine training and validation sets if we have a separate validation set
                self.train_dataset = datasets.concatenate_datasets([dataset_splits['train'], dataset_splits['validation']])
            else:
                # Use the full training set
                self.train_dataset = dataset_splits['train']

            # Validation during evaluation mode could be on the test set or another set if specified
            self.val_dataset = self.test_dataset
        else:
            # If not in evaluation mode, we train on the train set, and validate on the validation set
            if has_validation_split:
                # Use separate validation set if it exists
                self.train_dataset = dataset_splits['train']
                self.val_dataset = dataset_splits['validation']
            else:
                # Split the training dataset into training and validation
                self.train_dataset, self.val_dataset = dataset_splits['train'].train_test_split(
                    test_size=self.test_size,
                    seed=self.seed,
                    shuffle=True,
                ).values()

    def _set_dataset_label_fields(self, dataset):
        # extract the keys that contain the labels and images
        features = list(dataset.features.keys())

        if len(features) == 2:
            # easy case, it just contains the image and label field
            self._image_field, self._label_field = features
        elif len(features) == 3:
            self._image_field, label_fields = features[0], features[1:]

            # NB: For CIFAR-100 the `fine_label` is the first element in the list.
            #     This might need some adjusting for other datasets with multiple labels.
            self._label_field = label_fields[0]

            if torch.distributed.get_rank() == 0:
                log.info(f'Warning: Multiple dataset labels defined. Using `{self._label_field}`.')

        else:
            raise ValueError(f'Failed to get dataset fields. Unknown number of features: {len(features)}')

    def _apply_transforms_to_datasets(self):
        return # no default transforms

    def _initialize_dataloaders(self):
        self._set_generators_and_seed_worker()
        self._set_samplers_and_batch_size()
        self._create_dataloaders()

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
        if self._collate_fn:
            # NB: The collate_fn needs to know the label and image fields,
            #     so let's overwrite it with a function that has those.
            collate_fn = partial(self._collate_fn, self._label_field, self._image_field)
        else:
            collate_fn = self._default_collate_fn

        self._dataloaders['train'] = torch.utils.data.DataLoader(
            self.train_dataset.with_format('torch'),
            sampler=self.train_sampler,
            batch_size=self.local_batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.generator,
            worker_init_fn=self.seed_worker
        )

        self._dataloaders['valid'] = torch.utils.data.DataLoader(
            self.val_dataset.with_format('torch'),
            sampler=self.val_sampler,
            batch_size=self.physical_batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers
        )

        if self.test_dataset:
            self._dataloaders['test'] = torch.utils.data.DataLoader(
                self.test_dataset.with_format('torch'),
                sampler=self.test_sampler,
                batch_size=self.physical_batch_size,
                collate_fn=collate_fn,
                num_workers=self.num_workers
            )

    def _set_samplers_and_batch_size(self):
        # for the DP case, Opacus handles distributed for us. otherwise, we need
        # to use distributedsampler and divide the batch size by number of replicas
        if not self.privacy:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset.with_format('torch')
            )

            # For distributed without Opacus, we need to divide the batch size
            # by the world size.
            self.local_batch_size = self.batch_size // torch.distributed.get_world_size()
        else:
            # For the DP case, Opacus handles these for us
            self.train_sampler = None
            self.local_batch_size = self.batch_size

        # we will validate and test only on rank 0
        self.val_sampler, self.test_sampler = None, None

    def _get_stratified_subset(self, dataset):
        split_dataset = dataset.train_test_split(
            test_size=self.subset_size,
            seed=self.seed,
            stratify_by_column=self._label_field,
        )

        return split_dataset['test']

    def _get_few_shot_subset(self, dataset):
        test_size = self.shots * self.num_classes

        # special case. train_test_split is unable to "split" if
        # the requested split size equals the dataset size.
        if test_size == len(dataset):
            return dataset

        split_dataset = dataset.train_test_split(
            test_size=test_size,
            seed=self.seed,
            stratify_by_column=self._label_field,
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


    def _apply_transforms_to_datasets(self):
        def _apply_transforms(transforms, label_field, image_field, examples):
            log.info('.')
            examples[image_field] = [transforms(image) for image in examples[image_field]]
            return examples

        if self.transforms:
            if self.dataset_name == 'imagenet-1k':
                # ImageNet contains also grayscale images
                self._add_rgb_transform()

            transforms_func = partial(_apply_transforms, self.transforms, self._label_field, self._image_field)

            self.train_dataset = self.train_dataset.map(
                transforms_func,
                num_proc=self.num_workers,
                batched=True,
                load_from_cache_file=True,
            )
            self.val_dataset = self.val_dataset.map(
                transforms_func,
                num_proc=self.num_workers,
                batched=True,
                load_from_cache_file=True,
            )
            if self.test_dataset:
                self.test_dataset = self.test_dataset.map(
                    transforms_func,
                    num_proc=self.num_workers,
                    batched=True,
                    load_from_cache_file=True,
                )

    def _add_rgb_transform(self):
        toRGB = torchvision.transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x)
        new_transforms = [toRGB] + self.transforms.transforms
        self.transforms = torchvision.transforms.Compose(new_transforms)

    @staticmethod
    def _collate_fn(label_field, image_field, batch):
        B = len(batch)
        C, H, W = batch[0][image_field].shape

        images = torch.empty((B, C, H, W))
        labels = torch.empty(B, dtype=torch.long)

        for i in range(B):
            images[i] = batch[i][image_field]
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
            dataset_source=configuration.dataset_source,
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

