import pytest

pydantic = pytest.importorskip('pydantic')

from dpdl.configurationmanager import Configuration, Hyperparameters


def test_hyperparameters_reject_batch_size_and_sample_rate() -> None:
    with pytest.raises(ValueError):
        Hyperparameters(batch_size=32, sample_rate=0.1, epochs=1)


def test_hyperparameters_reject_conflicting_privacy_params() -> None:
    with pytest.raises(ValueError):
        Hyperparameters(
            epochs=1,
            batch_size=32,
            target_epsilon=5.0,
            noise_multiplier=1.0,
        )


def test_configuration_rejects_unknown_command() -> None:
    with pytest.raises(pydantic.ValidationError):
        Configuration(command='invalid', task='ImageClassification')
