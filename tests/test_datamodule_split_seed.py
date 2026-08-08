import datasets

from dpdl.datamodules import DataModule


def _validation_ids(training_seed, split_seed):
    train = datasets.Dataset.from_dict(
        {
            'id': list(range(1000)),
            'label': [index % 10 for index in range(1000)],
        }
    ).class_encode_column('label')
    test = datasets.Dataset.from_dict(
        {
            'id': list(range(1000, 1200)),
            'label': [index % 10 for index in range(200)],
        }
    ).class_encode_column('label')

    datamodule = DataModule.__new__(DataModule)
    datamodule._dataset_splits = datasets.DatasetDict(train=train, test=test)
    datamodule._label_field = 'label'
    datamodule.test_size = 0.1
    datamodule.val_size = 0.1
    datamodule.seed = training_seed
    datamodule.split_seed = split_seed
    datamodule.evaluation_mode = False
    datamodule._create_dataset_splits()
    return datamodule.val_dataset['id']


def test_split_seed_keeps_validation_rows_fixed_across_training_seeds():
    assert _validation_ids(7101, 7000) == _validation_ids(7102, 7000)
    assert _validation_ids(7101, 7000) != _validation_ids(7101, 7001)
