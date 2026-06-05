import logging
import re
import timm
import torch
import os
import sys
import importlib

from typing import Any, Dict, Optional

from .model_base import ModelBase
from .wide_resnet import WideResNet
from .koskela_model import KoskelaNet
from .hugging_face_models import HuggingfaceLanguageModel

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dpdl.configurationmanager import Configuration, Hyperparameters
from dpdl.peft import PeftFactory

log = logging.getLogger(__name__)

def add_noise_to_weights(model, noise_level):
    for name, param in model.named_parameters():
        if 'weight' in name:
            noise = torch.randn(param.size()) * noise_level
            param.data.add_(noise)

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by modification time"""
    if checkpoint_dir is None:
        return None

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        d
        for d in os.listdir(checkpoint_dir)
        if d.startswith('checkpoint_step_') and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]

    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest)

def import_by_name(model_name, model_dir):
    spec = importlib.util.spec_from_file_location(model_name, f"{model_dir}/{model_name}.py")
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = model_module
    spec.loader.exec_module(model_module)

    return model_module

def instantiate(class_name, model_module, num_classes, parameters=None):
    parameters = [] if parameters is None else parameters

    if class_name not in vars(model_module):
        raise AttributeError(f"Class {class_name} not found in custom model file, please rename your model class to camel case!")

    try:
        model_instance = vars(model_module)[class_name](*parameters, num_classes)
    except TypeError:
        model_instance = vars(model_module)[class_name]()

    return model_instance

class ModelFactory:

    @staticmethod
    def get_model(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        num_classes: int,
        loss_fn: torch.nn,
        metrics: Optional[Dict[str, Any]] = None
    ):

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

        # reference to model module to be loaded
        model_module = None

        # Flag to see if we load a local model already fine tuned
        checkpoints_dir_latest = None

        # check if we want to experiment on LLMs
        if configuration.llm:
            checkpoints_dir_latest = get_latest_checkpoint(
                configuration.checkpoints_dir
            )
            model_instance = HuggingfaceLanguageModel(
                configuration.model_name,
                configuration.load_in_4bit,
                num_labels=num_classes,
                peft=configuration.peft,
                checkpoint_dir=checkpoints_dir_latest,
                task=configuration.task,
            )

            transforms = model_instance.get_transforms()
        else:
            # keep settings for provided models to use them with dpdl binary
            if configuration.model_name.startswith('wrn-'):
                # Parse depth and width from model_name, e.g., 'wrn-16-4'
                parts = configuration.model_name.split('-')
                depth, width = int(parts[1]), int(parts[2])
                model_instance = WideResNet(depth=depth, width=width, num_classes=num_classes)
                transforms = model_instance.get_transforms()
            elif configuration.model_name == 'koskela_net':
                model_instance = KoskelaNet()
                transforms = model_instance.get_transforms()
            else:
                # check if we can use the model folder
                model_dir = f"{sys.path[0]}/dpdl/models/"
                dir_available = os.path.isdir(model_dir)

                # check if we receive a name or a path
                model_name = configuration.model_name
                path_given = os.path.isfile(configuration.model_name)


                model_parameters = []

                if path_given:
                    model_dir = model_name[:model_name.rfind('/')+1]
                    model_name = model_name.split("/")[-1].split('.py')[0]
                    camel_case_name = model_name.replace('_', ' ').title().replace(' ', '')

                    model_module = import_by_name(model_name, model_dir)

                    # update config to not clutter the model name with a full path
                    configuration.model_name = model_name

                else:
                    if dir_available:

                        # look for custom model specification before consulting timm
                        # simple, fixed architecture custom model
                        if f"{model_name}.py" in os.listdir(model_dir):
                            camel_case_name = model_name.replace('_', ' ').title().replace(' ', '')
                            model_module = import_by_name(model_name, model_dir)

                        # custom model with architectural parameters
                        elif f"{model_name.split('-')[0]}.py" in os.listdir(model_dir):
                            camel_case_name = model_name.split('-')[0].replace('_', ' ').title().replace(' ', '')

                            # transform every digit-only string to int, leave all other casting to the model
                            model_parameters = model_name.split('-')[1:]
                            model_parameters = list(map(lambda x: int(x) if x.isdigit() else x, model_parameters))

                            model_module = import_by_name(model_name.split('-')[0], model_dir)

                    # search timm if no model created yet
                    if model_module is None:
                        model_instance = timm.create_model(
                            configuration.model_name,
                            pretrained=configuration.pretrained,
                            num_classes=num_classes,
                        )

                        # Resolve data config and create transforms
                        model_config = timm.data.resolve_data_config({}, model=model_instance)
                        transforms = timm.data.transforms_factory.create_transform(**model_config)

        # we have loaded a module and need to instantiate and check it
        if model_module is not None:
            model_instance = instantiate(camel_case_name, model_module, num_classes, parameters=model_parameters)
            transforms = model_instance.get_transforms()

            # check adherance of class functions
            assert hasattr(model_instance, "get_classifier"), "Custom model does not have a get_classifier function!"
            assert hasattr(model_instance, "get_transforms"), "Custom model does not have a get_transforms function!"

        # resolve num_classes if needed
        if num_classes is None:
            if hasattr(model_instance, 'config') and getattr(model_instance.config, 'vocab_size', None):
                num_classes = int(model_instance.config.vocab_size)
            elif getattr(model_instance, 'num_classes', None):
                num_classes = int(model_instance.num_classes)
            else:
                raise ValueError('Num classes not given and unable to infer it.')

        # Wrap the instantiated model with ModelBase
        model = ModelBase(
            model_instance=model_instance,
            num_classes=num_classes,
            use_feature_cache=configuration.cache_features,
            criterion=loss_fn,
            metrics=metrics
        )

        # Add noise to (pretrained) weights?
        if configuration.weight_perturbation_level > 0:
            add_noise_to_weights(model, configuration.weight_perturbation_level)

        # zero the head weights?
        if configuration.zero_head:
            model.zero_head_weights()

        # should we do Parameter Efficient Fine-Tuning (PEFT)?
        if configuration.peft:
            model = PeftFactory.get_peft_model(model, configuration, checkpoints_dir_latest)

        return model, transforms, num_classes

