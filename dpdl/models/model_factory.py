import logging
import re
import timm
import torch

from .model_base import ModelBase
from .wide_resnet import WideResNet
from .koskela_model import KoskelaNet
from .hugging_face_models import HF_llm

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dpdl.configurationmanager import Configuration, Hyperparameters
from dpdl.peft import PeftFactory

log = logging.getLogger(__name__)

def add_noise_to_weights(model, noise_level):
    for name, param in model.named_parameters():
        if 'weight' in name:
            noise = torch.randn(param.size()) * noise_level
            param.data.add_(noise)

class ModelFactory:

    @staticmethod
    def get_model(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        num_classes: int,
        loss_fn: torch.nn.Module
    ):
        
        HF_LLM_PATTERNS = [
            r'.*gpt.*',
            r'.*roberta.*',
            r'.*bert.*',
            r'.*distilbert.*',
            r'.*llama.*',
            r'.*mistral.*',
            r'.*phi.*',
            r'.*gemma.*',
            r'.*opt.*',
            r'.*t5.*',
            r'.*flan.*',
            r'.*mpt.*',
            r'.*codellama.*',
            r'.*llama2.*',
            r'.*llama3.*',
            r'microsoft/.*',
            r'meta-llama/.*',
            r'mistralai/.*',
            r'huggingface/.*',
            r'.*\-chat.*',
            r'.*\-instruct.*',
        ]

        """
        Create a model instance based on the configuration, with support for PEFT and zeroing head weights.

        Parameters:
        - configuration: Configuration object containing model specs.
        - hyperparams: Optional hyperparameters, not directly used here.
        - num_classes: The number of classes for a classification problem

        Returns:
        - A tuple of (ModelBase instance, Data Transforms).
        """

        """
        TO DO: create LLM_base class, similar to ModelBase, but for LLMs?
        or just use ModelBase directly?
        """

        transforms = {}  # No default transforms
        model_instance = None


        # Flag to skip image model creation if we load HF
        loaded_hf = False

        # check if we want to experiment on LLMs
        if configuration.llm:
            model_instance = HF_llm(
                        configuration.model_name,
                        configuration.quantization_config,
                        num_labels=num_classes
                    )

            transforms = model_instance.get_transforms()
            loaded_hf = True

        if loaded_hf:
            pass  # model_instance & transforms already set
        elif configuration.model_name.startswith('wrn-'):
            # Parse depth and width from model_name, e.g., 'wrn-16-4'
            parts = configuration.model_name.split('-')
            depth, width = int(parts[1]), int(parts[2])
            model_instance = WideResNet(depth=depth, width=width, num_classes=num_classes)
            transforms = model_instance.get_transforms()
        elif configuration.model_name == 'koskela-net':
            model_instance = KoskelaNet()
            transforms = model_instance.get_transforms()
        else:
            model_instance = timm.create_model(
                configuration.model_name,
                pretrained=configuration.pretrained,
                num_classes=num_classes,
            )

            # Resolve data config and create transforms
            model_config = timm.data.resolve_data_config({}, model=model_instance)
            transforms = timm.data.transforms_factory.create_transform(**model_config)

        # Wrap the instantiated model with ModelBase
        model = ModelBase(
            model_instance=model_instance,
            num_classes=num_classes,
            use_feature_cache=configuration.cache_features,
            criterion=loss_fn,
        )

        # Add noise to (pretrained) weights?
        if configuration.weight_perturbation_level > 0:
            add_noise_to_weights(model, configuration.weight_perturbation_level)

        # zero the head weights?
        if configuration.zero_head:
            model.zero_head_weights()

        # should we do Parameter Efficient Fine-Tuning (PEFT)?
        if configuration.peft:
            model = PeftFactory.get_peft_model(model, configuration)

        return model, transforms

