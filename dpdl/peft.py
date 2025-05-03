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

def print_trainable_modules(model: torch.nn.Module):
    log.info('Trainable modules:')
    for module_name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):
            log.info(module_name)

class PeftFactory:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, configuration: Configuration):
        if configuration.peft == 'lora':
            return LoRA.get_peft_model(model, configuration.model_name)

        if configuration.peft == 'film':
            return FiLM.get_peft_model(model, configuration.model_name)

        if configuration.peft == 'head-only':
            return HeadOnly.get_peft_model(model, configuration.model_name)

        raise RuntimeError(f'Unkown PEFT method: {configuration.peft}')

class HeadOnly:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, model_name: str):
        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # enable gradients for the head
        model.get_classifier().requires_grad_(True)

        trainable_params, all_params = get_nb_trainable_parameters(model)

        if torch.distributed.get_rank() == 0:
            print_trainable_modules(model)

            log.info(f'Finetuning head only - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}')


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
        for module_name, module in model.named_modules():
            if pattern.match(module_name):
                for name, param in module.named_parameters():
                    param.requires_grad = True

            # also check if this module is explicitly set to be enabled
            for module_to_save in film_config.modules_to_save:
                if module_name.endswith(module_to_save):
                    for name, param in module.named_parameters():
                        param.requires_grad = True

        trainable_params, all_params = get_nb_trainable_parameters(model)

        if torch.distributed.get_rank() == 0:
            print_trainable_modules(model)

            log.info(f'FiLM setup done - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}')

        return model

    @staticmethod
    def _get_config(model_name: str):
        if model_name.startswith('vit_base_patch16_224'):
            return FilmConfig(
                target_modules=r'.*\.norm\d?',
                modules_to_save=['model.head'],
            )

        if model_name.startswith('vit_small_patch16_224'):
            return FilmConfig(
                target_modules=r'.*\.norm\d?',
                modules_to_save=['model.head'],
            )

        if model_name.startswith('vit_tiny_patch16_224'):
            return FilmConfig(
                target_modules=r'.*\.norm\d?',
                modules_to_save=['model.head'],
            )

        if model_name.startswith('resnetv2_50x1_bit'):
            return FilmConfig(
                target_modules=r'.*\.norm\d$',
                modules_to_save=['model.head.fc'],
            )

        raise RuntimeError(f'No known FiLM configuration for model: {model_name}')

class LoRA:
    @staticmethod
    def get_peft_model(model: torch.nn.Module, model_name: str):
        lora_config = LoRA._get_config(model_name)
        lora_model = get_peft_model(model, lora_config)

        trainable_params, all_params = get_nb_trainable_parameters(lora_model)

        if torch.distributed.get_rank() == 0:
            print_trainable_modules(model)
            log.info(f'LoRA setup done - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}')

        return lora_model

    @staticmethod
    def _get_config(model_name: str):
        # default rank
        lora_rank = 4

        # general recommendation for alpha is 2*rank
        lora_alpha = 2*lora_rank

        if model_name.startswith('vit_base_patch16_224'):
            return LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                bias='none',
                target_modules=r'patched_embed\.proj|.*\.attn\.qkv|.*\.attn_proj|.*\.mlp\.fc\d',
                modules_to_save=['head'],
            )

        if model_name.startswith('resnetv2_50x1_bit'):
            return LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                bias='none',
                target_modules=r'stem\.conv|.*\.downsample\.conv|.*\.conv\d',
                modules_to_save=['head.fc'],
            )

        raise RuntimeError(f'No known LoRA configuration for model: {model_name}')
