import logging
import opacus
import timm
import torch
import torchmetrics

from .configurationmanager import Configuration, Hyperparameters
from .peft import PeftFactory

log = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def get_model(configuration: Configuration, hyperparams: Hyperparameters):
        model = ImageClassificationModel(
            model_name=configuration.model_name,
            num_classes=configuration.num_classes,
            fix_model=configuration.modulevalidator_fix,
        )

        # adjust the model if Parameter Efficient Fine Tuning (PEFT) is requested
        model = PeftFactory.get_peft_model(model, configuration)

        # finally, zero the head weights if requested
        if configuration.zero_head:
            model.zero_head_weights()

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

    def show_layers(self):
        log.info('Layers:')

        for n, m in self.model.named_modules():
            log.info(f'{n}, {type(m)}')

    def zero_head_weights(self):
        classifier = self.model.get_classifier()
        torch.nn.init.zeros_(classifier.weight)
        if classifier.bias is not None:
            torch.nn.init.zeros_(classifier.bias)

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
