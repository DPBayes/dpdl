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
def download_generic_huggingface_model(model_name, quantization, trust_remote_code = False, num_labels: int | None = None, peft = False, checkpoint_dir = None):

    quantization_config = None

    if quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit = quantization
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
    elif checkpoint_dir is not None and not peft:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            **load_kwargs
        )
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer, quantization_config


class HF_llm (torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        quantization: dict,
        vocab_size: int = -1,
        ignore_index: int = -100,
        trust_remote_code: bool = False,
        peft: bool = False,
        checkpoint_dir: str = None,
        num_labels: int | None = None,
    ):

        super().__init__()

        
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.peft = peft
        self.model, self.tokenizer, self.quantization_config = download_generic_huggingface_model(
            model_name=model_name,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
            num_labels=num_labels,
            peft=peft,
            checkpoint_dir=checkpoint_dir
        )
        
        self.is_seq_classification = any(x in model_name.lower() for x in ['roberta', 'bert', 'distilbert'])

    def forward(self, x):
        """
        Return logits given tokenized batch or tensor input.
        """
        if isinstance(x, dict):
            out = self.model(**x)
        else:
            #out = self.model(input_ids=x[:,0,:], token_type_ids=x[:,1,:], attention_mask=x[:,2,:])
            out = self.model(x)
        
        print('out', out)

        if hasattr(out, 'logits'):
            return out.logits
        return out

    def forward_features(self, x):
        """
        Extract hidden features prior to classification/LM head.

        - Sequence classification models:
            * Prefer pooler_output, e.g., BERT.
            * Otherwise fallback to CLS token of last_hidden_state.
        - Causal LMs:
            * Return last_hidden_state (sequence length, hidden_size).
        """
        if isinstance(x, dict):
            out = self.model(**x, return_dict=True, output_hidden_states=True)
        else:
            out = self.model(x, return_dict=True, output_hidden_states=True)

        if self.is_seq_classification:
            # BERT-style: has pooler_output
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output  # (batch_size, hidden_size)

            # Otherwise: fallback to CLS token of final hidden state
            return out.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Causal LM: return final hidden states (all tokens)
        return out.last_hidden_state  # (batch_size, seq_len, hidden_size)


    def forward_head(self, features):
        
        head = self.get_classifier()
        if head is None:
            raise RuntimeError("No classifier/LM head found for this model")
        return head(features)

    # use CrossEntropyLoss from torch.nn? 
    #TODO: For sentence classification
    def criterion(self, logits, targets):

        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        print('are we here?', shift_logits.shape, shift_targets.shape)

        print(self._criterion)

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
    
    def save_model(self, path):
        #This saves the model, whether it has peft or not. If it has peft, it will save only the trainable parameters 
        self.model.save_pretrained(path)
