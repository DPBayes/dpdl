import pytest

yaml = pytest.importorskip('yaml')
pytest.importorskip('opacus')

from dpdl.hyperparameteroptimizer import HyperparameterOptimizer


def test_read_optuna_config_types() -> None:
    config = HyperparameterOptimizer.read_optuna_config('conf/optuna_hypers.conf')

    assert config['learning_rate']['type'] == 'float'
    assert isinstance(config['learning_rate']['min'], float)
    assert isinstance(config['learning_rate']['max'], float)
    assert isinstance(config['learning_rate']['log_space'], bool)

    assert config['epochs']['type'] == 'int'
    assert isinstance(config['epochs']['min'], int)
    assert isinstance(config['epochs']['max'], int)
