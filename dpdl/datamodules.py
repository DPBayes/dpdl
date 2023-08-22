# NB: Set datasets cache directory with the environment variable HF_DATASETS_CACHE
import datasets
import opacus
import torch

from functools import partial

from dpdl.utils import seed_everything

class DataModule:
    def __init__(
        self,
        batch_size: int = 64,
        physical_batch_size: int = 64,
        num_workers: int = 4,
        subset_size: float = None,
        seed: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.physical_batch_size = physical_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.subset_size = subset_size

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
    def __init__(self, *, dataset_name='cifar10', image_size=None, **kwargs):
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
            generator.manual_seed(self.seed)

        def seed_worker(worker_id):
            seed_everything(self.seed)

        self._train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset.with_format('torch'),
            batch_size=self.batch_size,
            collate_fn=partial(self._collate_fn, self._get_dataset_label_field()),
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker if self.seed else None,
        )

        self._val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset.with_format('torch'),
            batch_size=self.physical_batch_size,
            collate_fn=partial(self._collate_fn, self._get_dataset_label_field()),
            num_workers=self.num_workers,
            shuffle=False,
        )

        self._test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset.with_format('torch'),
            batch_size=self.physical_batch_size,
            collate_fn=partial(self._collate_fn, self._get_dataset_label_field()),
            num_workers=self.num_workers,
            shuffle=False,
        )

    def _get_dataset_label_field(self):
        if self.dataset_name in self.dataset_label_fields:
            label_field = self.dataset_label_fields[self.dataset_name]
        else:
            label_field = 'label'

        return label_field

    def _get_stratified_subset(self, dataset):
        label_field = self._get_dataset_label_field()
        labels = torch.tensor(dataset[label_field])
        unique_labels = labels.unique()

        sampled_indices = []
        for label in unique_labels:
            # find the indices of the dataset where the current label is present
            label_indices = torch.where(labels == label)[0]

            # determine how many samples are needed for the given label based on the subset size
            num_samples_per_class = int(len(label_indices) * self.subset_size)

            # randomly choose the required number of indices for the current label
            chosen_indices = torch.randperm(len(label_indices))[:num_samples_per_class]

            # add the chosen indices to the sampled_indices list
            sampled_indices.extend(label_indices[chosen_indices].tolist())

        # shuffle the sampled indices
        sampled_indices = torch.tensor(sampled_indices)[torch.randperm(len(sampled_indices))].tolist()

        return dataset.select(sampled_indices)

    def _initialize_datasets(self):
        # only load the data in single process
        if torch.distributed.get_rank() == 0:
            dataset = datasets.load_dataset(self.dataset_name, split='train')
            test_dataset = datasets.load_dataset(self.dataset_name, split='test')
            torch.distributed.barrier()
        else:
            # wait for local rank 0 to load the datasets
            torch.distributed.barrier()

            dataset = datasets.load_dataset(self.dataset_name, split='train')
            test_dataset = datasets.load_dataset(self.dataset_name, split='test')

        # stratified sampling for subset_size
        if self.subset_size:
            dataset = self._get_stratified_subset(dataset)
            test_dataset = self._get_stratified_subset(test_dataset)

        # do we need to scale the images?
        if self.image_size:
            transform = partial(self._resize_transform, self.image_size)
            dataset = dataset.map(transform, batched=True)
            test_dataset = test_dataset.map(transform, batched=True)

        # create train and validation splits
        split_dataset = dataset.train_test_split(test_size=0.1, shuffle=False)
        self.train_dataset = split_dataset['train']
        self.val_dataset = split_dataset['test']

        if self.image_size:
            transform = partial(self._resize_transform, self.image_size)
            test_dataset = test_dataset.map(transform, batched=True)

        self.test_dataset = test_dataset

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
    def get_datamodule(configuration: dict, hyperparams: dict) -> DataModule:
        datamodule = ImageDataModule(
            dataset_name=configuration['dataset_name'],
            num_workers=configuration['num_workers'],
            batch_size=hyperparams['batch_size'],
            physical_batch_size=configuration['physical_batch_size'],
            seed=configuration['seed'],
            subset_size=configuration['dataset_subset_size'],
            image_size=(224, 224),
        )

        return datamodule

