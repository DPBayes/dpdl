import opacus
import torch
import timm

class ImageClassificationModel(torch.nn.Module):
    def __init__(self,
            model_name: str ='resnet18',
            num_classes: int = 10,
            pretrained: bool = True,
        ):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )

        if not opacus.validators.ModuleValidator.is_valid(self.model):
            self.model = opacus.validators.ModuleValidator.fix(self.model)

    def forward(self, x):
        return self.model(x)

