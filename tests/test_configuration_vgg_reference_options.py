import pytest
from pydantic import ValidationError

from dpdl.configurationmanager import Configuration


def _base_kwargs() -> dict:
    return {
        'command': 'train',
        'privacy': True,
        'noise_multiplier': 1.0,
        'max_grad_norm': 1.0,
        'epochs': 1,
        'batch_size': 4,
        'poisson_sampling': True,
        'noise_mechanism': 'gaussian',
        'accountant': 'prv',
    }


def test_vgg_reference_resolves_defaults() -> None:
    cfg = Configuration(
        **_base_kwargs(),
        model_name='vgg_bnb_reference',
    )
    assert cfg.vgg_ref_channels == [32, 64, 128]
    assert cfg.vgg_ref_dense_size == 128
    assert cfg.vgg_ref_activation == 'tanh'
    assert cfg.vgg_ref_input_size == 32


def test_vgg_reference_rejects_non_reference_model_overrides() -> None:
    with pytest.raises(ValidationError, match='VGG reference parameters require --model-name vgg_bnb_reference'):
        Configuration(
            **_base_kwargs(),
            model_name='resnet18',
            vgg_ref_dense_size=64,
        )


def test_vgg_reference_rejects_invalid_channels() -> None:
    with pytest.raises(ValidationError, match='--vgg-ref-channels must provide exactly three integers'):
        Configuration(
            **_base_kwargs(),
            model_name='vgg_bnb_reference',
            vgg_ref_channels=[32, 64],
        )


def test_vgg_reference_rejects_invalid_input_size() -> None:
    with pytest.raises(ValidationError, match='--vgg-ref-input-size must be divisible by 8 and >= 8'):
        Configuration(
            **_base_kwargs(),
            model_name='vgg_bnb_reference',
            vgg_ref_input_size=30,
        )
