import datasets
import logging
import math
import os
import pathlib
import torch
import torchvision
import numpy as np
import tensorflow_datasets as tfds

from collections import Counter
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
        max_test_examples: int = 0,
        sample_rate: float = 0,
        physical_batch_size: int = 64,
        num_workers: int = 4,
        subset_size: int = None,
        shots: int = None,
        stratify_shots: bool = True,
        seed: int = 0,
        privacy: bool = True,
        test_size: float = 0.1,
        validation_size: float = 0.1,
        split_seed: int = 42,
        evaluation_mode: bool = False,
        label_field: str = None,
        image_field: str = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_source = dataset_source
        self.batch_size = batch_size
        self.max_test_examples = max_test_examples
        self.sample_rate = sample_rate
        self.physical_batch_size = physical_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.subset_size = subset_size
        self.shots = shots
        self.privacy = privacy
        self.test_size = test_size
        self.val_size = validation_size
        self.split_seed = split_seed
        self.evaluation_mode = evaluation_mode
        self._image_field = image_field
        self._label_field = label_field
        self._stratify_shots = stratify_shots

        self._dataloaders = {
            'train': None,
            'valid': None,
            'test': None,
        }

        # The _load_datasets method will fill this
        self.num_classes = None

        # Load datasets to memory
        if torch.distributed.get_rank() == 0:
            # First load the data only rank 0. This is because, the datasets
            # might need to be loaded over the network, and rank 0 can cache
            # them to disk.
            self._load_datasets()
            torch.distributed.barrier()
        else:
            # Other ranks wait here for rank 0 to do its job.
            torch.distributed.barrier()

            # Now other ranks can load them to memory directly from disk
            self._load_datasets()

    def initialize(self, transforms: torchvision.transforms.transforms.Compose):
        self.transforms = transforms

        # Again, first do the initialization on rank 0, so it can cache everything
        # on disk without race conditions.
        # NB: There _might_ be some methods to speed this up using multiple GPUs.
        if torch.distributed.get_rank() == 0:
            self._initialize_datasets()
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
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

    def get_num_classes(self):
        return self.num_classes

    def get_dataloader(self, name):
        return self._dataloaders.get(name)

    def get_dataset_size(self, which='train_dataset'):
        dataset = getattr(self, which)
        return len(dataset)

    def set_dataloader(self, name, dataloader):
        self._dataloaders[name] = dataloader

    def _default_collate_fn(batch):
        # default collate is a no-op
        return batch

    def _initialize_datasets(self):
        # Create datasets train/validation/test splits if they do not yet exists
        self._create_dataset_splits()

        if self.subset_size and self.shots:
            raise ValueError('Subset size and shots are exlusive.')

        # if subset of dataset is requested, we'll do stratified sampling
        if self.subset_size is not None and self.subset_size < 1.0:
            self.train_dataset = self._get_stratified_subset(self.train_dataset)
            self.val_dataset = self._get_stratified_subset(self.val_dataset)

        if self.shots is not None:
            self.train_dataset = self._get_few_shot_subset(self.train_dataset)

        if self.max_test_examples:
            if len(self.val_dataset) > self.max_test_examples:
                log.info(f'Validation dataset has {len(self.val_dataset)} example which is more than the configured maximum ({self.max_test_examples}). Limiting dataset size.')
                _, self.val_dataset = self.val_dataset.train_test_split(
                    test_size=self.max_test_examples,
                    seed=self.split_seed,
                    shuffle=True,
                    stratify_by_column=self._label_field,
                ).values()

            if len(self.test_dataset) > self.max_test_examples:
                log.info(f'Test dataset has {len(self.test_dataset)} example which is more than the configured maximum ({self.max_test_examples}). Limiting dataset size.')

                _, self.test_dataset = self.test_dataset.train_test_split(
                    test_size=self.max_test_examples,
                    seed=self.split_seed,
                    shuffle=True,
                    stratify_by_column=self._label_field,
                ).values()

        self._apply_transforms_to_datasets()

    def _load_datasets(self):
        """Load the datasets to memory."""
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
        self._set_dataset_label_fields(dataset_splits)

        # Make sure the dataset label field is of type ClassLabel
        self._dataset_splits = self._enforce_label_field_type(dataset_splits)

        # Automatically determine the number of classes
        # NB: This can be done if the label is of type ClassLabel
        self.num_classes = dataset_splits['train'].features[self._label_field].num_classes

        if torch.distributed.get_rank() == 0:
            log.info(f'Determined the number of classes to be {self.num_classes}.')

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

    def _create_dataset_splits(self):
        # Check if there's a validation split available
        has_validation_split = 'validation' in self._dataset_splits
        has_test_split = 'test' in self._dataset_splits

        # We have all the splits, just use them as they are
        if has_validation_split and has_test_split:
            # Use separate validation set if it exists
            self.train_dataset = self._dataset_splits['train']
            self.val_dataset = self._dataset_splits['validation']
            self.test_dataset = self._dataset_splits['test']

        # No validation or test splist, create both
        if not has_validation_split and not has_test_split:
            # Split the training dataset into training and validation
            self.train_dataset, val_and_test_split = self._dataset_splits['train'].train_test_split(
                test_size=(self.test_size + self.val_size),
                seed=self.split_seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

            self.val_dataset, self.test_dataset = val_and_test_split.train_test_split(
                test_size=0.5,
                seed=self.split_seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

        # We have only test split, create validation split from train
        if not has_validation_split and has_test_split:
            # Split the training dataset into training and validation
            self.train_dataset, self.val_dataset = self._dataset_splits['train'].train_test_split(
                test_size=self.test_size,
                seed=self.split_seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

            self.test_dataset = self._dataset_splits['test']

        if has_validation_split and not has_test_split:
            raise ValueError('Splitting not implemented: Dataset has validation split but no test split.')

        if self.evaluation_mode:
            if has_validation_split:
                # Combine training and validation sets if we have a separate validation set
                self.train_dataset = datasets.concatenate_datasets([
                    self._dataset_splits['train'],
                    self._dataset_splits['validation']
                ])
            else:
                # Use the full training set
                self.train_dataset = self._dataset_splits['train']

            # In evaluation mode, we validate on the test dataset
            self.val_dataset = self.test_dataset

    def _enforce_label_field_type(self, dataset_splits):
        # Iterate through all dataset splits, and make the label field ClassLabel
        for key in dataset_splits.keys():
            dataset = dataset_splits[key]

            # If it already is a ClassLabel, HF dataset will throw an error, so check first
            if not isinstance(dataset.features[self._label_field], datasets.ClassLabel):
                dataset = dataset.class_encode_column(self._label_field)

            dataset_splits[key] = dataset

        return dataset_splits

    def _set_dataset_label_fields(self, dataset_splits):
        # extract the keys that contain the labels and images
        if torch.distributed.get_rank() == 0:
            log.info('Setting dataset fields.')

        self._set_image_field(dataset_splits['train'])
        self._set_label_field(dataset_splits['train'])

    def _set_image_field(self, dataset):
        if self._image_field is None:
            for feature_name, feature in dataset.features.items():
                if isinstance(feature, datasets.Image):
                    self._image_field = feature_name
                    break

            if self._image_field:
                if torch.distributed.get_rank() == 0:
                    log.info(f' - Determined image field: {self._image_field}')
            else:
                features = dataset.features.keys()
                raise ValueError('Could not determine image field for dataset.')

    def _set_label_field(self, dataset):
        if self._label_field is None:
            for feature_name, feature in dataset.features.items():
                if isinstance(feature, datasets.ClassLabel) or feature_name == 'label':
                    self._label_field = feature_name
                    break

            if self._label_field:
                if torch.distributed.get_rank() == 0:
                    log.info(f' - Determined label field: {self._label_field}')
            else:
                features = dataset.features.keys()
                raise ValueError('Could not determine label field for dataset. Available features: {features}')

    def _apply_transforms_to_datasets(self):
        return # no default transforms

    def _initialize_dataloaders(self):
        self._set_generators_and_seed_worker()
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
        # We might need initialize a DataModule without a batch size,
        # at least in the case of figuring out the maximum batch size
        # from the dataset length.
        if not self.batch_size:
            if torch.distributed.get_rank() == 0:
                log.info('Batch size not yet initialized, skipping dataloader creation.')
            return

        self._set_samplers_and_batch_size()

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
            if self.batch_size:
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
            seed=self.split_seed,
            stratify_by_column=self._label_field,
        )

        return split_dataset['test']

    def _get_few_shot_subset(self, dataset):
        if not self.num_classes:
            raise ValueError('Number of classes unknown, can not create few shot dataset.')

        test_size = self.shots * self.num_classes

        # Special case: `train_test_split` is unable to "split" if
        # the requested split size equals the dataset size. Also,
        # for small datasets we request more samples than exist.
        if test_size >= len(dataset):
            return dataset

        split_dataset = dataset.train_test_split(
            test_size=test_size,
            seed=self.split_seed,
            stratify_by_column=self._label_field if self._stratify_shots else None,
        )

        subset = split_dataset['test']

        if torch.distributed.get_rank() == 0:
            c = Counter(subset[self._label_field])
            n_examples = sum(c.values())
            log.info(f'Collected few shot dataset with {n_examples} examples: {c}')

        return subset

class ImageDataModule(DataModule):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def cache_features(self, model):
        """Cache features for the train, validation, and test datasets using the provided model."""

        if torch.distributed.get_rank() == 0:
            log.info('Feature caching enabled, caching features.')

        def _extract_features(model, image_field, examples):
            inputs = examples[image_field]

            with torch.no_grad():
                features = model.forward_features(inputs)

            examples['features'] = features.cpu()

            return examples

        model = model.cuda()
        _extract_features_fn = partial(_extract_features, model, self._image_field)

        if torch.distributed.get_rank() == 0:
            log.info(f' - Processing {len(self.train_dataset)} examples in the train dataset.')

        datasets_map_bs = 512

        self.train_dataset = self.train_dataset.with_format('torch', device='cuda').map(
            _extract_features_fn,
            batched=True,
            batch_size=datasets_map_bs,
            remove_columns=self._image_field,
            num_proc=self.num_workers,
        )

        if torch.distributed.get_rank() == 0:
            log.info(f' - Processing {len(self.val_dataset)} examples in the validation dataset.')

        self.val_dataset = self.val_dataset.with_format('torch', device='cuda').map(
            _extract_features_fn,
            batched=True,
            batch_size=datasets_map_bs,
            remove_columns=self._image_field,
            num_proc=self.num_workers,
        )

        if self.test_dataset:
            if torch.distributed.get_rank() == 0:
                log.info(f' - Processing {len(self.test_dataset)} examples in the test dataset.')

            self.test_dataset = self.test_dataset.with_format('torch', device='cuda').map(
                _extract_features_fn,
                batched=True,
                batch_size=datasets_map_bs,
                remove_columns=self._image_field,
            )

        # Update the collation function to the one that uses cached features
        self._collate_fn = self._collate_fn_with_cached_features

        self._create_dataloaders()

        if torch.distributed.get_rank() == 0:
            log.info('Feature caching finished.')

    def _apply_transforms_to_datasets(self):
        if torch.distributed.get_rank() == 0:
            log.info('Applying transformations to dataset.')

        def _apply_transforms(transforms, label_field, image_field, examples):
            log.info('.')
            examples[image_field] = [transforms(image) for image in examples[image_field]]
            return examples

        if self.transforms:
            # Make sure all images are RGB
            self._add_rgb_transform()

            transforms_func = partial(
                _apply_transforms,
                self.transforms,
                self._label_field,
                self._image_field,
            )

            log.info(f' - Processing {len(self.train_dataset)} examples in the train dataset.')
            self.train_dataset = self.train_dataset.map(
                transforms_func,
                num_proc=self.num_workers,
                batched=True,
                load_from_cache_file=True,
            )

            log.info(f' - Processing {len(self.val_dataset)} examples in the validation dataset.')
            self.val_dataset = self.val_dataset.map(
                transforms_func,
                num_proc=self.num_workers,
                batched=True,
                load_from_cache_file=True,
            )

            if self.test_dataset:
                log.info(f' - Processing {len(self.test_dataset)} examples in the test dataset.')
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

    @staticmethod
    def _collate_fn_with_cached_features(label_field, image_field, batch):
        features = torch.stack(
            [item['features'] for item in batch]
        )

        labels = torch.tensor(
            [item[label_field] for item in batch]
        )

        return features, labels

class DataModuleFactory:
    @staticmethod
    def get_datamodule(
        configuration: Configuration,
        hyperparams: Hyperparameters,
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
            evaluation_mode=configuration.evaluation_mode,
            label_field=configuration.dataset_label_field,
            max_test_examples=configuration.max_test_examples,
        )

        return datamodule

