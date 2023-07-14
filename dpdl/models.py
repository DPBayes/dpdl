import opacus
import timm
import torch
import torchmetrics

class ModelFactory():
    @staticmethod
    def get_model(configuration, hyperparams):
        model = ImageClassificationModel(
            model_name=hyperparams['model_name'],
            num_classes=configuration['num_classes']
        )
        return model

class TimmModel(torch.nn.Module):
    def __init__(
            self,
            *,
            model_name: str = 'resnet18',
            pretrained: bool = True,
            **kwargs,
        ):

        super().__init__(**kwargs)

        self.model_name = model_name
        self.pretrained = pretrained

    def forward(self, x):
        return self.model(x)

    def criterion(self, logits, y):
        raise(NotImplementedError('Criterion not implemented for class: {self.__class__.__name__}'))

    def accuracy(self, logits, y):
        raise(NotImplementedError('Accuracy not implemented for class: {self.__class__.__name__}'))

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
            self.model = opacus.validators.ModuleValidator.fix(self.model)

        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        self._accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)

    def criterion(self, logits, y):
        return self._criterion(logits, y)

    def accuracy(self, logits, y):
        preds = torch.argmax(logits, dim=1)
        return self._accuracy(preds, y)

