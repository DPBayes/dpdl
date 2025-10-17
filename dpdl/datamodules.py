import datasets
import logging
import math
import os
import pathlib
import torch
import torchvision
import numpy as np
import random

from collections import Counter
from functools import partial
from PIL import Image
from typing import Tuple

from dpdl.utils import seed_everything
from .configurationmanager import Configuration, Hyperparameters

from transformers.data.data_collator import DataCollatorWithPadding

log = logging.getLogger(__name__)


class DataModule:

    def __init__(
        self,
        dataset_name: str = 'default-dataset',
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
        imbalance_factor: float = None,
        imbalance_reverse: bool = False,
        fairness_imbalance_class: int = None,
        cache_transforms: bool = False,
    ):

        self.dataset_name = dataset_name
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
        self._imbalance_factor = imbalance_factor
        self._imbalance_reverse = imbalance_reverse
        self._fairness_imbalance_class = fairness_imbalance_class
        self._cache_transforms = cache_transforms

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

        # Make sure all images are RGB
        self._add_rgb_transform()

        if self.transforms:
            # We need to remove the ToTensor transformation if not caching
            # transformations. The issue is that we are passing the dataset
            # to the dataloader using the `with_format('torch')` transformation
            # that already maps these to tensor and it's much faster to do it
            # in the dataset end, because this is a zero-copy operation with
            # Arrow datasets.
            if not self._cache_transforms:
                self._replace_to_tensor_with_to_float()

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

        # Imbalance before subsetting. If done in other order, we can get into
        # trouble with e.g. classes with zero examples
        # exponential distribution, not Fairness-style imbalance
        if self._imbalance_factor and not self._fairness_imbalance_class:
            if torch.distributed.get_rank() == 0:
                log.info('Creating imbalanced train set..')

            self.train_dataset = self._get_imbalanced_subset(self.train_dataset, self._imbalance_reverse)

            if torch.distributed.get_rank() == 0:
                log.info('Creating imbalanced validation set..')

            self.val_dataset = self._get_imbalanced_subset(self.val_dataset, self._imbalance_reverse)

            if torch.distributed.get_rank() == 0:
                log.info('Creating imbalanced test set..')

            self.test_dataset = self._get_imbalanced_subset(self.test_dataset, self._imbalance_reverse)

        # NOTE: we use full data for validation and test, but scale the metrics accordingly.
        if self._fairness_imbalance_class:

            if self._imbalance_reverse:
                raise ValueError('Cannot reverse imbalance for fairness style imbalanced dataset.')

            if torch.distributed.get_rank() == 0:
                log.info("Creating fairness imbalanced train set..")
                self.train_dataset = self._get_fairness_imbalanced_subset(
                    self.train_dataset
                )

            if torch.distributed.get_rank() == 0:
                log.info("Creating fairness imbalanced validation set..")
                self.val_dataset = self._get_fairness_imbalanced_subset(
                    self.val_dataset
                )

            if torch.distributed.get_rank() == 0:
                log.info(
                    f"We will not create fairness imbalanced test sets. Size of test set: {len(self.test_dataset)}"
                )

        # if subset of dataset is requested, we'll do stratified sampling
        if self.subset_size is not None and self.subset_size < 1.0:
            self.train_dataset = self._get_stratified_subset(self.train_dataset)

            if torch.distributed.get_rank() == 0:
                train_distribution = Counter(self.train_dataset[self._label_field])
                log.info(f'Training set (size: {len(self.train_dataset)}) class distribution after taking subset of size {self.subset_size}: {sorted(train_distribution.items())}')

            self.val_dataset = self._get_stratified_subset(self.val_dataset)

            if torch.distributed.get_rank() == 0:
                val_distribution = Counter(self.val_dataset[self._label_field])
                log.info(f'Validation set (size: {len(self.val_dataset)}) class distribution after taking subset of size {self.subset_size}: {sorted(val_distribution.items())}')

        if self.shots is not None:
            self.train_dataset = self._get_few_shot_subset(self.train_dataset)

        if self.max_test_examples:
            if len(self.val_dataset) > self.max_test_examples:
                if torch.distributed.get_rank() == 0:
                    log.info(f'Validation dataset has {len(self.val_dataset)} examples which is more than the configured maximum ({self.max_test_examples}). Limiting dataset size.')

                _, self.val_dataset = self.val_dataset.train_test_split(
                    test_size=self.max_test_examples,
                    seed=self.split_seed,
                    shuffle=True,
                    stratify_by_column=self._label_field,
                ).values()

            if len(self.test_dataset) > self.max_test_examples:
                if torch.distributed.get_rank() == 0:
                    log.info(f'Test dataset has {len(self.test_dataset)} examples which is more than the configured maximum ({self.max_test_examples}). Limiting dataset size.')

                _, self.test_dataset = self.test_dataset.train_test_split(
                    test_size=self.max_test_examples,
                    seed=self.split_seed,
                    shuffle=True,
                    stratify_by_column=self._label_field,
                ).values()

        if self._cache_transforms:
            # We need to apply transforms last. If we do it first, all examples in
            # a very large dataset will be transformed first and we probably won't
            # even use them all.
            self._apply_transforms_to_datasets()

    def _load_datasets(self):
        """Load the datasets to memory."""
        if torch.distributed.get_rank() == 0:
            log.info(f'Loading dataset "{self.dataset_name}" from Huggingface datasets.')

        dataset_splits = datasets.load_dataset(self.dataset_name)

        # Set dataset label fields based on the training split
        self._set_dataset_label_fields(dataset_splits)

        # Make sure the dataset label field is of type ClassLabel
        self._dataset_splits = self._enforce_label_field_type(dataset_splits)

        # Automatically determine the number of classes
        # NB: This can be done if the label is of type ClassLabel
        self.num_classes = dataset_splits['train'].features[self._label_field].num_classes

        if torch.distributed.get_rank() == 0:
            log.info(f'Determined the number of classes to be {self.num_classes}.')

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
                seed=self.seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

            self.val_dataset, self.test_dataset = val_and_test_split.train_test_split(
                test_size=0.5,
                seed=self.seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

        # We have only test split, create validation split from train
        if not has_validation_split and has_test_split:
            # Split the training dataset into training and validation
            self.train_dataset, self.val_dataset = self._dataset_splits['train'].train_test_split(
                test_size=self.test_size,
                seed=self.seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

            self.test_dataset = self._dataset_splits['test']

        if has_validation_split and not has_test_split:
            # Keep the original train split
            self.train_dataset = self._dataset_splits['train']

            # Split the validation into validation and test (50/50)
            self.val_dataset, self.test_dataset = self._dataset_splits['validation'].train_test_split(
                test_size=0.5,
                seed=self.seed,
                shuffle=True,
                stratify_by_column=self._label_field,
            ).values()

        if self.evaluation_mode:
            # Combine training and validation sets if we have a separate validation set
            self.train_dataset = datasets.concatenate_datasets([
                self.train_dataset,
                self.val_dataset,
            ])

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
                raise ValueError(f'Could not determine label field for dataset. Available features: {features}')

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
            #
            #     We also need to do the transformations in the collate function
            #     if we have not mapped the transformations to disk cache.
            if self._cache_transforms:
                collate_fn = partial(
                    self._collate_fn,
                    label_field=self._label_field,
                    image_field=self._image_field,
                )
            else:
                collate_fn = partial(
                    self._collate_fn,
                    label_field=self._label_field,
                    image_field=self._image_field,
                    transforms=self.transforms,
                )
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
        # Split the dataset using `split_seed`
        g = torch.Generator()
        g.manual_seed(self.split_seed)

        # Convert labels to a tensor
        labels = torch.tensor(dataset[self._label_field])

        # Get unique labels in the dataset
        unique_labels = labels.unique()

        sampled_indices = []

        # Iterate over each unique label
        for label in unique_labels:
            # Find the indices where the current label is present
            label_indices = torch.where(labels == label)[0]

            # Determine how many samples are needed for the given label based on the subset size
            num_samples_per_class = int(len(label_indices) * self.subset_size)

            # Make sure at least one sample is selected per class
            num_samples_per_class = max(1, num_samples_per_class)

            # Randomly choose the required number of indices for the current label
            chosen_indices = torch.randperm(len(label_indices), generator=g)[:num_samples_per_class]

            # Add the chosen indices to the sampled_indices list
            sampled_indices.extend(label_indices[chosen_indices].tolist())

        # Shuffle the sampled indices for randomization
        sampled_indices = torch.tensor(sampled_indices)[torch.randperm(len(sampled_indices), generator=g)].tolist()

        # Return the selected subset of the dataset
        return dataset.select(sampled_indices)

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

    def _get_imbalanced_subset(self, dataset, reverse=False):
        """
        Creates an imbalanced subset using an exponential distribution.

        https://github.com/richardaecn/class-balanced-loss/blob/1d7857208a2abc03d84e35a9d5383af8225d4b4d/src/data_utils.py#L93-L115

        If reverse is True, the class distribution is inverted:
          - The class that originally gets the most samples now gets the least,
          - And vice versa.
        """
        if not self._imbalance_factor:
            raise ValueError('Imbalance factor must be provided for creating an imbalanced dataset.')

        # Get the label counts and basic statistics.
        label_counts = Counter(dataset[self._label_field])
        num_classes = len(label_counts)
        max_count = max(label_counts.values())

        # Calculate the number of samples per class based on an exponential distribution.
        # For cls_idx = 0 we get max_count and for cls_idx = num_classes-1 we get fewer samples.
        img_num_per_cls = [
            max(1, int(max_count * (self._imbalance_factor ** (cls_idx / (num_classes - 1.0)))))
            for cls_idx in range(num_classes)
        ]

        # Do we want to flip the imbalance order?
        if reverse:
            img_num_per_cls.reverse()

        # Create a torch generator for reproducibility.
        g = torch.Generator()
        g.manual_seed(self.split_seed)

        # Collect indices for each class.
        class_indices = {cls: [] for cls in range(num_classes)}
        for idx, sample in enumerate(dataset):
            class_indices[sample[self._label_field]].append(idx)

        # Sample indices for each class using torch.randperm.
        sampled_indices = []
        for cls, indices in class_indices.items():
            indices_tensor = torch.tensor(indices)
            perm = torch.randperm(len(indices_tensor), generator=g)
            selected = indices_tensor[perm[:img_num_per_cls[cls]]]
            sampled_indices.extend(selected.tolist())

        sampled_dataset = dataset.select(sampled_indices)

        if torch.distributed.get_rank() == 0:
            distribution = Counter(sampled_dataset[self._label_field])
            log.info(f'Created imbalanced dataset (size: {len(sampled_dataset)}) with class distribution: {sorted(distribution.items())}')

        return sampled_dataset

    def _get_fairness_imbalanced_subset(self, dataset):
        """
        Creates an imbalanced subset with one class having less examples than the others.
        """
        if not self._fairness_imbalance_class:
            raise ValueError('Fairness imbalance class must be provided for creating an imbalanced dataset.')

        if self._imbalance_factor == 1.0:
            raise ValueError('Imbalance factor must be less than 1.0 for creating an imbalanced dataset.')

        # Get label counts as a list of counts
        label_counts = list(Counter(dataset[self._label_field]).values())
        num_classes = len(label_counts)

        # Determine number of samples per class: for the fairness-imbalanced class, reduce the count
        img_num_per_cls = [
            label_counts[i] if i != self._fairness_imbalance_class else int(label_counts[i] * self._imbalance_factor)
            for i in range(num_classes)
        ]

        # Create a torch generator for reproducibility
        g = torch.Generator()
        g.manual_seed(self.split_seed)

        # Collect indices for each class
        class_indices = {cls: [] for cls in range(num_classes)}
        for idx, sample in enumerate(dataset):
            class_indices[sample[self._label_field]].append(idx)

        # Sample indices for each class using torch.randperm
        sampled_indices = []
        for cls, indices in class_indices.items():
            num_samples = img_num_per_cls[cls]
            if num_samples > len(indices):
                log.warning(f'Requested {num_samples} samples for class {cls}, but only {len(indices)} available. Adjusting to {len(indices)}.')
                num_samples = len(indices)
            indices_tensor = torch.tensor(indices)
            perm = torch.randperm(len(indices_tensor), generator=g)
            selected = indices_tensor[perm[:num_samples]]
            sampled_indices.extend(selected.tolist())

        sampled_dataset = dataset.select(sampled_indices)

        if torch.distributed.get_rank() == 0:
            distribution = Counter(sampled_dataset[self._label_field])
            log.info(f'Created fairness imbalanced dataset (size: {len(sampled_dataset)}) with class distribution: {sorted(distribution.items())}')

        return sampled_dataset


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
            transforms_func = partial(
                _apply_transforms,
                self.transforms,
                self._label_field,
                self._image_field,
            )

            if torch.distributed.get_rank() == 0:
                log.info(f' - Processing {len(self.train_dataset)} examples in the train dataset.')

            self.train_dataset = self.train_dataset.map(
                transforms_func,
                num_proc=self.num_workers,
                batched=True,
                load_from_cache_file=True,
            )

            if torch.distributed.get_rank() == 0:
                log.info(f' - Processing {len(self.val_dataset)} examples in the validation dataset.')

            self.val_dataset = self.val_dataset.map(
                transforms_func,
                num_proc=self.num_workers,
                batched=True,
                load_from_cache_file=True,
            )

            if self.test_dataset:
                if torch.distributed.get_rank() == 0:
                    log.info(f' - Processing {len(self.test_dataset)} examples in the test dataset.')

                self.test_dataset = self.test_dataset.map(
                    transforms_func,
                    num_proc=self.num_workers,
                    batched=True,
                    load_from_cache_file=True,
                )

    def _add_rgb_transform(self):
        # Function for converting a PIL image to RGB
        def to_rgb_pil(x):
            if isinstance(x, Image.Image) and x.mode != 'RGB':
                return x.convert('RGB')
            return x

        # Function for converting a torch.Tensor to RGB
        def to_rgb_tensor(x):
            if isinstance(x, torch.Tensor):
                if len(x.shape) == 3 and x.shape[0] == 1:  # Grayscale tensor (C, H, W)
                    return x.repeat(3, 1, 1)  # Convert 1-channel to 3-channel (RGB)
                elif len(x.shape) == 3 and x.shape[0] == 3:
                    return x  # Already RGB
                else:
                    raise ValueError('Input tensor is not a valid image tensor.')
            return x

        # Select the appropriate transformation based on the type of input
        if self._cache_transforms:
            toRGB = torchvision.transforms.Lambda(to_rgb_pil)
        else:
            toRGB = torchvision.transforms.Lambda(to_rgb_tensor)

        # Update the transform pipeline
        new_transforms = [toRGB] + self.transforms.transforms
        self.transforms = torchvision.transforms.Compose(new_transforms)

    def _replace_to_tensor_with_to_float(self):
        # We need to convert the tensor to float and normalize to [0, 1]
        to_float = torchvision.transforms.Lambda(lambda x: x.float() / 255.0)

        # Filter out ToTensor and replace it with the new transformation
        new_transforms = []
        for t in self.transforms.transforms:
            if isinstance(t, torchvision.transforms.ToTensor):
                new_transforms.append(to_float)
            else:
                new_transforms.append(t)

        self.transforms.transforms = new_transforms

    @staticmethod
    def _collate_fn(batch, label_field=None, image_field=None, transforms=None):
        B = len(batch)

        # Apply transformation to the first image to determine the size after transformation
        if transforms:
            first_image = transforms(batch[0][image_field])
        else:
            first_image = batch[0][image_field]

        # Now that we know the transformed image size, we can initialize the `images` tensor
        C, H, W = first_image.shape
        images = torch.empty((B, C, H, W))
        labels = torch.empty(B, dtype=torch.long)

        # Now we are ready to process the batch
        for i in range(B):
            if transforms:
                images[i] = transforms(batch[i][image_field])
            else:
                images[i] = batch[i][image_field]

            labels[i] = batch[i][label_field]

        return images, labels

    @staticmethod
    def _collate_fn_with_cached_features(batch, label_field=None, image_field=None):
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

        if getattr(configuration, 'llm', False):
            # Use NLPDataModule for LLM tasks
            return NLPDataModule(
                max_length=hyperparams.max_length,
                task=configuration.task,
                dataset_name=configuration.dataset_name,
                num_workers=configuration.num_workers,
                physical_batch_size=configuration.physical_batch_size,
                subset_size=configuration.subset_size,
                validation_size=configuration.validation_size,
                test_size=configuration.test_size,
                shots=configuration.shots,
                seed=configuration.seed,
                batch_size=hyperparams.batch_size,
                sample_rate=hyperparams.sample_rate,
                privacy=configuration.privacy,
                evaluation_mode=configuration.evaluation_mode,
                label_field=configuration.dataset_label_field,
                max_test_examples=configuration.max_test_examples,
                imbalance_factor=configuration.imbalance_factor,
                imbalance_reverse=configuration.imbalance_reverse,
                fairness_imbalance_class=configuration.fairness_imbalance_class,
                cache_transforms=False,  # no image transforms for text
                split_seed=configuration.split_seed,
            )

        return ImageDataModule(
            dataset_name=configuration.dataset_name,
            num_workers=configuration.num_workers,
            physical_batch_size=configuration.physical_batch_size,
            subset_size=configuration.subset_size,
            validation_size=configuration.validation_size,
            test_size=configuration.test_size,
            shots=configuration.shots,
            seed=configuration.seed,
            batch_size=hyperparams.batch_size,
            sample_rate=hyperparams.sample_rate,
            privacy=configuration.privacy,
            evaluation_mode=configuration.evaluation_mode,
            label_field=configuration.dataset_label_field,
            max_test_examples=configuration.max_test_examples,
            imbalance_factor=configuration.imbalance_factor,
            imbalance_reverse=configuration.imbalance_reverse,
            fairness_imbalance_class=configuration.fairness_imbalance_class,
            cache_transforms=configuration.cache_dataset_transforms,
            split_seed=configuration.split_seed,
        )


class NLPDataModule(DataModule):
    """DataModule specialized for NLP tasks.
    - Detect text fields (string features) automatically if they are not specified.
    - Collate function returning (tokenized_batch_dict, labels_tensor). e.g., tokenized_batch_dict = {"input_ids": torch.Tensor(batch_size, seq_len),
                                                                        "attention_mask": torch.Tensor(batch_size, seq_len)}
    """

    def __init__(self, text_fields=None, max_length: int = 64, task: str = None, **kwargs):
        self._text_fields = text_fields  # list of text fields or None
        self.max_length = max_length
        self.task = task
        super().__init__(**kwargs)

    def _set_dataset_label_fields(self, dataset_splits):  
        print('an I using this method?')
        if torch.distributed.get_rank() == 0:
            log.info('Setting dataset label field (LLM mode).')
        print('self.task',self.task)
        if self.task != 'CasualLM':
            self._set_label_field(dataset_splits['train']) # find the label column
        self._detect_text_fields(dataset_splits['train']) # decide which text column(s) to use
        
        

    # def _set_label_field(self, dataset):

    #     if self._label_field is None and self.task == 'CausalLM':
    #         self._label_field = self._text_fields
        
    #     elif self._label_field is None and self.task != 'CausalLM':
    #         for feature_name, feature in dataset.features.items():
    #             if isinstance(feature, datasets.ClassLabel) or feature_name == 'label':
    #                 self._label_field = feature_name
    #                 break

    #         if self._label_field:
    #             if torch.distributed.get_rank() == 0:
    #                 log.info(f' - Determined label field: {self._label_field}')
    #         else:
    #             features = dataset.features.keys()
    #             raise ValueError(f'Could not determine label field for dataset. Available features: {features}')

    def _detect_text_fields(self, dataset):
        if self._text_fields is not None:
            if torch.distributed.get_rank() == 0:
                log.info(f"Using manually provided text fields: {self._text_fields}")
            return

        # Otherwise detect text fields (string features)
        candidates = []
        for feature_name, feature in dataset.features.items():
            # Skip label field
            if feature_name == self._label_field:
                continue
            # The datasets library uses Value('string') for raw text columns
            if getattr(feature, 'dtype', None) == 'string':
                candidates.append(feature_name)

        #if candidates:
        #    self._text_fields = candidates
        #    if len(candidates) > 1 and torch.distributed.get_rank() == 0:
        #        log.warning(f"Multiple text fields detected, using all: {candidates}")
        #else:
        #    self._text_fields = []
        
        self._text_fields = candidates[:1] if candidates else []  # use only the first text field if multiple found

        if torch.distributed.get_rank() == 0:
            log.info(f"Detected text fields: {self._text_fields}")
        if not self._text_fields:
            raise ValueError('Could not determine any text field for NLP dataset.')

    # skip image transforms and set custom dataloaders
    def initialize(self, tokenizer):
        self.tokenizer = tokenizer

        if torch.distributed.get_rank() == 0:
            log.info('Initializing NLPDataModule datasets...')
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

        self._initialize_text_dataloaders()


    def _initialize_text_dataloaders(self):
        self._set_generators_and_seed_worker()
        self._set_samplers_and_batch_size()

        collate_fn = self._make_text_collate()

        self._dataloaders['train'] = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.local_batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            generator=self.generator,
            worker_init_fn=self.seed_worker,
        )

        self._dataloaders['valid'] = torch.utils.data.DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.physical_batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

        if self.test_dataset:
            self._dataloaders['test'] = torch.utils.data.DataLoader(
                self.test_dataset,
                sampler=self.test_sampler,
                batch_size=self.physical_batch_size,
                collate_fn=collate_fn,
                num_workers=self.num_workers,
            )

    # use data collator from HF, e.g., DataCollatorWithPadding(tokenizer)?
    # or use custom collate function?
    def _make_text_collate(self):
        tokenizer = self.tokenizer
        text_fields = self._text_fields
        label_field = self._label_field
        max_len = self.max_length
        task = self.task

        def collate(batch):
            texts = []
            for sample in batch:
                # Concatenate multiple text fields if present
                parts = [str(sample[i]) for i in text_fields]
                texts.append(' '.join(parts))

            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            ) 

            if task == 'CausalLM':
                labels = tokenized['input_ids'].clone()
                labels[labels == tokenizer.pad_token_id] = -100
            elif task == 'SequenceClassification':
                labels = torch.tensor([sample[label_field] for sample in batch], dtype=torch.long)
            return tokenized, labels

        return collate

    def _add_rgb_transform(self):  
        return
    def _replace_to_tensor_with_to_float(self):  
        return
    def _apply_transforms_to_datasets(self):  
        return
