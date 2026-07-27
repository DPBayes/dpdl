import pytest

from dpdl.peft import LoRA


@pytest.mark.parametrize("rank", [1, 2])
def test_bert_lora_config_honors_requested_rank(rank):
    config = LoRA._get_config("bert-base-uncased", lora_rank=rank)

    assert config.r == rank
    assert config.lora_alpha == 2 * rank
