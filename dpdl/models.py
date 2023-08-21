import opacus
import timm
import torch
import torchmetrics

class ModelFactory:
    @staticmethod
    def get_model(configuration, hyperparams):
        model = ImageClassificationModel(
            model_name=hyperparams['model_name'],
            num_classes=configuration['num_classes'],
            fix_model=configuration['modulevalidator_fix'],
        )
        return model

class TimmModel(torch.nn.Module):
    def __init__(
            self,
            *,
            model_name: str = 'resnet18',
            pretrained: bool = True,
            fix_model: bool = False,
            **kwargs,
        ):

        super().__init__(**kwargs)

        self.model_name = model_name
        self.pretrained = pretrained
        self.fix_model = fix_model

        # no default metrics
        self.train_metrics = torchmetrics.MetricCollection([])
        self.valid_metrics = torchmetrics.MetricCollection([])

    def forward(self, x):
        return self.model(x)

    def criterion(self, logits, y):
        raise NotImplementedError('Criterion not implemented for class: {self.__class__.__name__}')

class ImageClassificationModel(TimmModel):
    def __init__(
            self,
            *,
            num_classes: int = 10,
            **kwargs,
        ):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.model = timm.create_model(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained
        )

        if not opacus.validators.ModuleValidator.is_valid(self.model):
            if self.fix_model:
                self.model = opacus.validators.ModuleValidator.fix(self.model)
            else:
                raise RuntimeError("Model contains layers that are note compatible with DP-SGD. "
                                   "Use --modulevalidator-fix (with caution!) to automatically fix the model.")

        self._criterion = torch.nn.CrossEntropyLoss().cuda()

        # let's track the accuracy
        self.train_metrics = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes),
        ])

        self.valid_metrics = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes),
        ])

    def criterion(self, logits, y):
        return self._criterion(logits, y)
