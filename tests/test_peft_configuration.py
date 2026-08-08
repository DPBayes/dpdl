import pytest
from pydantic import ValidationError

from dpdl.configurationmanager import Configuration
from dpdl.peft import LoRA


def test_lora_rank_is_required():
    with pytest.raises(ValidationError, match='lora_rank.*required'):
        Configuration(command='train', peft='lora')


def test_lora_rank_must_be_positive():
    with pytest.raises(ValidationError, match='lora_rank.*positive'):
        Configuration(command='train', peft='lora', lora_rank=0)


def test_lora_alpha_defaults_to_rank():
    configuration = Configuration(command='train', peft='lora', lora_rank=4)

    assert configuration.lora_alpha == 4


def test_lora_alpha_must_be_positive():
    with pytest.raises(ValidationError, match='lora_alpha.*positive'):
        Configuration(command='train', peft='lora', lora_rank=4, lora_alpha=0)


def test_vit_lora_config_uses_requested_rank_and_alpha():
    config = LoRA._get_config('vit_tiny_patch16_224.augreg_in21k', lora_rank=4, lora_alpha=8)

    assert config.r == 4
    assert config.lora_alpha == 8
