import os
import torch
import torchmetrics


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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # let's track the training accuracy
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="macro",
                ).to(self.device),
                "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="micro",
                ).to(self.device),
                "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="none",
                ).to(self.device),
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
                ).to(self.device),
                "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="micro",
                    sync_on_compute=False,
                ).to(self.device),
                "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_classes,
                    average="none",
                    sync_on_compute=False,
                ).to(self.device),
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

    def save_model(self, fpath):
        # Extract the directory from the path
        directory = os.path.dirname(fpath)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        torch.save(self.model.state_dict(), fpath)
