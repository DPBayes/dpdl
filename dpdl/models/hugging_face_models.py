import os
import json
import re
import typing as t
from huggingface_hub import snapshot_download

import torch
from safetensors.torch import load_file as safe_load_file

from transformers import AutoModel, AutoModelForCausalLM,AutoTokenizer, BitsAndBytesConfig

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"


"""
Let's make two versions of model loading. 
    - The first one is loading the tensors. This way, we can control better the architecture.
    Pros:
        - We can control better the architecture
        - We can introduce the LoRA parameters 
    Cons:
        - We need the architecture defined

    - Load directly from HuggingFace
    Pros:
        - Easy to load.
        - No need of knowing the original architecture
    Cons:
        - Less control over the model. We use it more as a black box. Maybe modify it is harder.
"""

"""
download_safetensors
#TODO 
"""
def download_safetensors(model_name):

    #folder = snapshot_download("microsoft/Phi-3-mini-128k-instruct", allow_patterns=["*.json", "*model*.safetensors"])
    folder = snapshot_download(model_name, allow_patterns=["*.json", "*model*.safetensors"])

    print(folder)

    # Load the index
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
    assert os.path.isfile(safe_index_file)
    with open(safe_index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))
    state_dict: t.Dict[str, torch.Tensor] = {}
    for shard_file in shard_files:
        state_dict |= safe_load_file(os.path.join(folder, shard_file), "cpu")

"""
download_generic_huggingface_model

Downloads a generic AutoModelForCausalLM.
model_name: str 
quantization: dict
trust_remote_code: bool
We should:
    - Create a method that is for other tasks
"""
def download_generic_huggingface_model(model_name, quantization, trust_remote_code = False, peft = False, checkpoint_dir = None):

    quantization_config = None

    if quantization:
        quantization_config = BitsAndBytesConfig(
            **quantization
        )

    load_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": trust_remote_code
        }
    
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    if checkpoint_dir is not None and not peft:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            **load_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, quantization_config


class ModelBaseLLM(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        quantization: dict,
        vocab_size: int = -1,
        ignore_index: int = -100,
        trust_remote_code: bool = False,
        peft: bool = False,
        checkpoint_dir: str = None
    ):

        super().__init__()

        
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.peft = peft
        self.model, self.tokenizer, self.quantization_config = download_generic_huggingface_model(model_name=model_name,quantization=quantization,trust_remote_code=trust_remote_code,peft=peft,checkpoint_dir=checkpoint_dir)

    def criterion(self, logits, targets):

        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        return self._criterion(
            shift_logits.view(-1,self.vocab_size), 
            shift_targets.view(-1),
            self.ignore_index
        )

    def get_classifier(self):
        return self.model.lm_head

    def get_body(self):
        return self.model.get_base_model().model

    def get_transforms(self):
        return self.tokenizer
    
    def save_model(self, path):
        #This saves the model, whether it has peft or not. If it has peft, it will save only the trainable parameters 
        self.model.save_pretrained(path)


        # # let's track the training accuracy
        # self.train_metrics = torchmetrics.MetricCollection(
        #     {
        #         "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
        #             num_classes=self.vocab_size,
        #             average="macro",
        #         ).cuda(),
        #         "Perplexity": torchmetrics.text.Perplexity().cuda()
        #     }
        # )

        # # we only validate on rank 0, so there's no need to
        # # synchronize when calculating the metrics.
        # # NB: If `sync_on_compute` is enabled, this breaks
        # # distributed training. If this needs to be enabled,
        # # then we also need to actually run the validation on
        # # all the GPUs.
        # self.valid_metrics = torchmetrics.MetricCollection(
        #     {
        #         "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
        #             num_classes=self.vocab_size,
        #             average="macro",
        #             sync_on_compute=False,
        #         ).cuda(),
        #         "Perplexity": torchmetrics.text.Perplexity().cuda(),
        #     }
        # )

        # self.test_metrics = torchmetrics.MetricCollection(
        #     {
        #         "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
        #             num_classes=self.vocab_size,
        #             average="macro",
        #             sync_on_compute=False,
        #         ).cuda(),
        #         "Perplexity": torchmetrics.text.Perplexity().cuda(),
        #         "ConfusionMatrix": torchmetrics.ConfusionMatrix(
        #             task="multiclass" if self.vocab_size > 2 else "binary",
        #             num_classes=self.vocab_size,
        #             sync_on_compute=False,
        #         ).cuda(),
        #     }
        # )