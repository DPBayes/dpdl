import timm
import torch

class TimmModel(torch.nn.Module):
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True, **kwargs):
        super().__init__()

        # Initialize the TIMM model
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)

    def forward(self, x):
        return self.model(x)
