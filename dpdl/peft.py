import logging
import re
import torch

from dataclasses import dataclass, field
from typing import List
from peft import get_peft_model, LoraConfig

from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)

def get_nb_trainable_parameters(model: torch.nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

class PeftFactory:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, configuration: Configuration):
        if configuration.lora:
            model = LoRA.get_peft_model(model, configuration.model_name)

        if configuration.film:
            model = FiLM.get_peft_model(model, configuration.model_name)

        return model

@dataclass
class FilmConfig:
    target_modules: str
    modules_to_save: List[str] = field(default_factory=list)

class FiLM:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, model_name: str):
        # get the FiLM configuration
        film_config = FiLM._get_config(model_name)

        # compile regex pattern for matching target modules
        pattern = re.compile(film_config.target_modules)

        # initially, we'll freeze all params
        for param in model.parameters():
            param.requires_grad = False

        # we enable the gradient for all parameters in matching modules
        for name, module in model.named_modules():
            if pattern.match(name):
                for param in module.parameters():
                    param.requires_grad = True

        # if there are any modules (such as head) that we want to train,
        # let's enable gradients here
        for name, param in model.named_parameters():
            if name in film_config.modules_to_save:
                param.requires_grad = True

        trainable_params, all_params = get_nb_trainable_parameters(model)

        if torch.distributed.get_rank() == 0:
            log.info(f'FiLM setup done - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}')

        return model

    @staticmethod
    def _get_config(model_name: str):
        if model_name.startswith('vit_base_patch16_224'):
            return FilmConfig(
                target_modules=r'.*\.norm\d?',
                modules_to_save=['head'],
            )

        if model_name.startswith('resnetv2_50x1_bitm_in21k'):
            return FilmConfig(
                target_modules=r'.*\.norm\d$',
                modules_to_save=['head.fc'],
            )

        raise RuntimeError(f'No known FiLM configuration for model: {configuration.model_name}')

class LoRA:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, model_name: str):
        lora_config = LoRA._get_config(model_name)
        lora_model = get_peft_model(model, lora_config)

        trainable_params, all_params = get_nb_trainable_parameters(lora_model)

        if torch.distributed.get_rank() == 0:
            log.info(f'LoRA setup done - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}')

        return lora_model

    @staticmethod
    def _get_config(model_name: str):
        if model_name.startswith('vit_base_patch16_224'):
            return LoraConfig(
                r=4,
                bias='none',
                target_modules=r'patched_embed\.proj|.*\.attn\.qkv|.*\.attn_proj|.*\.mlp\.fc\d',
                modules_to_save=['head'],
            )

        if model_name.startswith('resnetv2_50x1_bitm_in21k'):
            return LoraConfig(
                r=4,
                bias='none',
                target_modules=r'stem\.conv|.*\.downsample\.conv|.*\.conv\d|head.fc',
                modules_to_save=['head.fc'],
            )

        raise RuntimeError(f'No known LoRA configuration for model: {configuration.model_name}')
