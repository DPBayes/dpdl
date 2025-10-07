import logging
import os

from typing import Any, Dict, Optional

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

    def forward(self, x):
        if self.use_feature_cache:
            # self.model.forward_head(x) calls its classification head, x here are feature tensor not raw inputs
            # it take those features, possibly apply pooling/dropout, and then run the final linear (classifier) layer to produce logits.
            return self.model.forward_head(x) 
        else:
            return self.model(x)
    
    # def forward(self, *args, **kwargs):
    #     if self.use_feature_cache:
    #         x = args[0] if args else (kwargs.get("x") or kwargs.get("features"))
    #         return self.model.forward_head(x)

    #     if kwargs:
    #         print("are we here?")
    #         return self.model(**kwargs)
    #     elif args:
    #         return self.model(*args)
    #     else:
    #         raise TypeError("forward received no inputs")

    def forward_head(self, x):
        return self.model.forward_head(x)

    def forward_features(self, x):
        return self.model.forward_features(x)

    def criterion(self, logits, targets):
        return self._criterion(logits, targets)

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

    def save_model(self,fpath):
        
        directory = os.path.dirname(fpath)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self.model.save_model(fpath)

    # def save_model(self, fpath):
    #     # Extract the directory from the path
    #     directory = os.path.dirname(fpath)

    #     # Create the directory if it doesn't exist
    #     if not os.path.exists(directory):
    #         os.makedirs(directory, exist_ok=True)

    #     torch.save(self.model.state_dict(), fpath)