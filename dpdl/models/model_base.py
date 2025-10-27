import logging
import os

from typing import Any, Dict, Optional
from collections.abc import Mapping

import torch
import torchmetrics

log = logging.getLogger(__name__)


class ModelBase(torch.nn.Module):
    def __init__(
        self,
        model_instance: torch.nn.Module = None,
        num_classes: int = 10,
        use_feature_cache: bool = False,
        criterion: torch.nn.Module = torch.nn.CrossEntropyLoss() ,
        metrics: Optional[Dict[str, Any]] = None
    ):

        super().__init__()

        self.model = model_instance
        self.num_classes = num_classes
        self.use_feature_cache = use_feature_cache

        self._criterion = criterion.cuda()

        if metrics is not None:
            self.train_metrics = metrics['train_metrics']
            self.valid_metrics = metrics['valid_metrics']
            self.test_metrics = metrics['test_metrics']
    
    @property
    def config(self):
        return self.model.config
    
    @property
    def prepare_inputs_for_generation(self):
        """Expose the underlying model's method."""
        return self.model.prepare_inputs_for_generation
    
    def set_metrics(self, metrics):
        self.train_metrics = metrics['train_metrics']
        self.valid_metrics = metrics['valid_metrics']
        self.test_metrics = metrics['test_metrics']
        
    def forward(self, *args, **kwargs):

        # If PEFT calls with keyword arguments, convert them to a dict and pass as x
        if kwargs and not args:
            if isinstance(kwargs.get('input_ids'), Mapping):
                x = kwargs['input_ids']
            else:
                x = kwargs
        elif args:
            x = args[0]
        else:
            x = None

        if self.use_feature_cache:
            return self.model.forward_head(x) 
        else:
            return self.model(x)

    def forward_head(self, x):
        return self.model.forward_head(x)

    def forward_features(self, x):
        return self.model.forward_features(x)

    def criterion(self, logits, targets):
        return self._criterion(logits, targets)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def show_layers(self):
        log.info("Layers:")

        for n, m in self.model.named_modules():
            log.info(f"{n}, {type(m)}")

    def zero_head_weights(self):
        classifier = self.model.get_classifier()
        torch.nn.init.zeros_(classifier.weight)
        if classifier.bias is not None:
            torch.nn.init.zeros_(classifier.bias)

    def get_classifier(self):
        return self.model.get_classifier()

    def get_body(self):
        return torch.nn.Sequential(*list(self.model.children())[:-1])

    def save_model(self, fpath):
        # Extract the directory from the path
        directory = os.path.dirname(fpath)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self.model.save_model(fpath)