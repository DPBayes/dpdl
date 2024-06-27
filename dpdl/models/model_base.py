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

        self._criterion = torch.nn.CrossEntropyLoss().cuda()

        # let's track the training accuracy
        self.train_metrics = torchmetrics.MetricCollection({
            'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
                num_classes=self.num_classes,
                average='macro',
            ).cuda(),
            'MulticlassAccuracyWithMicro': torchmetrics.classification.MulticlassAccuracy(
                num_classes=self.num_classes,
                average='micro',
            ).cuda(),
        })

        # we only validate on rank 0, so there's no need to
        # synchronize when calculating the metrics. although,
        # the flag is probably redundant.
        self.valid_metrics = torchmetrics.MetricCollection({
            'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
                num_classes=self.num_classes,
                average='macro',
                sync_on_compute=False,
            ).cuda(),
            'MulticlassAccuracyWithMicro': torchmetrics.classification.MulticlassAccuracy(
                num_classes=self.num_classes,
                average='micro',
                sync_on_compute=False,
            ).cuda(),
        })

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
        log.info('Layers:')

        for n, m in self.model.named_modules():
            log.info(f'{n}, {type(m)}')

    def zero_head_weights(self):
        classifier = self.model.get_classifier()
        torch.nn.init.zeros_(classifier.weight)
        if classifier.bias is not None:
            torch.nn.init.zeros_(classifier.bias)

    def get_classifier(self):
        return self.model.get_classifier()

