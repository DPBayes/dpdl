import logging
import os

import torch
import torchmetrics

log = logging.getLogger(__name__)


class ModelBase(torch.nn.Module):
    def __init__(
        self,
        model_instance: torch.nn.Module = None,
        num_classes: int = 10,
        use_feature_cache: bool = False,
    ):

        super().__init__()

        self.model = model_instance
        self.num_classes = num_classes
        self.use_feature_cache = use_feature_cache

        self._criterion = torch.nn.CrossEntropyLoss().cuda()

        # let's track the training accuracy
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="macro",
                ).cuda(),
                "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="micro",
                ).cuda(),
                "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="none",
                ).cuda(),
            }
        )

        # we only validate on rank 0, so there's no need to
        # synchronize when calculating the metrics.
        # NB: If `sync_on_compute` is enabled, this breaks
        # distributed training. If this needs to be enabled,
        # then we also need to actually run the validation on
        # all the GPUs.
        self.valid_metrics = torchmetrics.MetricCollection(
            {
                "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="macro",
                    sync_on_compute=False,
                ).cuda(),
                "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="micro",
                    sync_on_compute=False,
                ).cuda(),
                "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="none",
                    sync_on_compute=False,
                ).cuda(),
            }
        )

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="macro",
                    sync_on_compute=False,
                ).cuda(),
                "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="micro",
                    sync_on_compute=False,
                ).cuda(),
                "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="none",
                    sync_on_compute=False,
                ).cuda(),
                "ConfusionMatrix": torchmetrics.ConfusionMatrix(
                    task="multiclass" if self.num_classes > 2 else "binary",
                    num_classes=self.num_classes,
                    sync_on_compute=False,
                ).cuda(),
            }
        )

    def forward(self, x):
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

        torch.save(self.model.state_dict(), fpath)

class ModelBaseLLM(torch.nn.Module):
    def __init__(
        self,
        model_instance: torch.nn.Module = None,
        vocab_size: int = -1,
        ignore_index: int = -100
    ):

        super().__init__()

        self.model = model_instance
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

        self._criterion = torch.nn.CrossEntropyLoss().cuda()

        # let's track the training accuracy
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.vocab_size,
                    average="macro",
                ).cuda(),
                "Perplexity": torchmetrics.text.Perplexity().cuda()
            }
        )

        # we only validate on rank 0, so there's no need to
        # synchronize when calculating the metrics.
        # NB: If `sync_on_compute` is enabled, this breaks
        # distributed training. If this needs to be enabled,
        # then we also need to actually run the validation on
        # all the GPUs.
        self.valid_metrics = torchmetrics.MetricCollection(
            {
                "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.vocab_size,
                    average="macro",
                    sync_on_compute=False,
                ).cuda(),
                "Perplexity": torchmetrics.text.Perplexity().cuda(),
            }
        )

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.vocab_size,
                    average="macro",
                    sync_on_compute=False,
                ).cuda(),
                "Perplexity": torchmetrics.text.Perplexity().cuda(),
                "ConfusionMatrix": torchmetrics.ConfusionMatrix(
                    task="multiclass" if self.vocab_size > 2 else "binary",
                    num_classes=self.vocab_size,
                    sync_on_compute=False,
                ).cuda(),
            }
        )

    def forward(self, x):
        if self.use_feature_cache:
            return self.model.forward_head(x)
        else:
            return self.model(x)

    def forward_head(self, x):
        return self.model.forward_head(x)

    def forward_features(self, x):
        return self.model.forward_features(x)

    def criterion(self, logits, targets):

        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        return self._criterion(
            shift_logits.view(-1,self.vocab_size), 
            shift_targets.view(-1),
            self.ignore_index
        )

    def show_layers(self):
        log.info("Layers:")

        for n, m in self.model.named_modules():
            log.info(f"{n}, {type(m)}")

    def zero_head_weights(self):
        torch.nn.init.zeros_(self.model.lm_head.weight)
        if self.model.lm_head.bias is not None:
            torch.nn.init.zeros_(self.model.lm_head.bias)

    def get_classifier(self):
        return self.model.lm_head

    def get_body(self):
        return self.model.get_base_model().model

    def save_model(self, fpath):
        # Extract the directory from the path
        directory = os.path.dirname(fpath)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        torch.save(self.model.state_dict(), fpath)
