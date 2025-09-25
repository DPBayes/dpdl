import os
import json
import re
import typing as t
from huggingface_hub import snapshot_download

import torch
from safetensors.torch import load_file as safe_load_file

from transformers import AutoModel, AutoModelForCausalLM,AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification

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
def download_generic_huggingface_model(
    model_name,
    quantization,
    trust_remote_code=False,
    num_labels: int | None = None,
):
    
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

    # For encoder/sequence classification models (roberta/bert), they are under AutoModelForSequenceClassification.
    is_seq_classification = any(x in model_name.lower() for x in ['roberta', 'bert', 'distilbert'])
    if is_seq_classification:
        if num_labels is not None:
            load_kwargs["num_labels"] = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **load_kwargs
        )
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer


class HF_llm (torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        quantization: dict,
        vocab_size: int = -1,
        ignore_index: int = -100,
        trust_remote_code: bool = False,
        num_labels: int | None = None,
    ):

        super().__init__()

        self.model, self.tokenizer = download_generic_huggingface_model(
            model_name=model_name,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
            num_labels=num_labels,
        )
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.is_seq_classification = any(x in model_name.lower() for x in ['roberta', 'bert', 'distilbert'])

    def forward(self, x):
        """
        Return logits given tokenized batch or tensor input.
        """
        if isinstance(x, dict):
            out = self.model(**x)
        else:
            out = self.model(x)
        if hasattr(out, 'logits'):
            return out.logits
        return out


    # use CrossEntropyLoss from torch.nn? 
    # move to LLM_base?
    def criterion(self, logits, targets):

        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        return self._criterion(
            shift_logits.view(-1,self.vocab_size), 
            shift_targets.view(-1),
            self.ignore_index
        )

    # Encoder models (e.g., RoBERTa/BERT) commonly use classifier or score. Causal LMs use lm_head.
    def get_classifier(self):
        for attr in ['classifier', 'score', 'lm_head']:
            if hasattr(self.model, attr):
                head = getattr(self.model, attr)
                if isinstance(head, torch.nn.Module):
                    return head
        return None

    #def get_body(self):
    #    return self.model.get_base_model().model

    def get_transforms(self):
        return self.tokenizer
