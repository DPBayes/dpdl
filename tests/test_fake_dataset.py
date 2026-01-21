import os

import pytest

torch = pytest.importorskip('torch')

from dpdl.configurationmanager import Configuration, Hyperparameters
from dpdl.datamodules import DataModuleFactory


def test_fake_dataset_loads_without_network() -> None:
    old_env = os.environ.get('DPDL_FAKE_DATASET')
    os.environ['DPDL_FAKE_DATASET'] = '1'
    try:
        configuration = Configuration(
            command='train',
            task='ImageClassification',
            dataset_name='fake',
            privacy=True,
            model_name='resnet18',
            subset_size=1.0,
            shots=None,
            peft=None,
            device='cpu',
        )
        hyperparams = Hyperparameters(
            epochs=1,
            batch_size=None,
            noise_multiplier=None,
            max_grad_norm=None,
            target_epsilon=None,
            noise_batch_ratio=None,
        )
        datamodule = DataModuleFactory.get_datamodule(
            configuration,
            hyperparams,
            device=torch.device('cpu'),
        )
        datamodule._initialize_datasets()

        assert datamodule.get_num_classes() == 2
        assert len(datamodule.train_dataset) > 0
    finally:
        if old_env is None:
            os.environ.pop('DPDL_FAKE_DATASET', None)
        else:
            os.environ['DPDL_FAKE_DATASET'] = old_env
