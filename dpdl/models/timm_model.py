import timm
import torch

class TimmModel(torch.nn.Module):
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True, **kwargs):
        super().__init__()

        # Initialize the TIMM model
        model_instance = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)

        if 'resnet' in model_name:
            model_instance.conv1 = torch.nn.Conv2d(1,
                                               model_instance.conv1.out_channels,
                                                 kernel_size=model_instance.conv1.kernel_size,
                                                 stride=model_instance.conv1.stride,
                                                 padding=model_instance.conv1.padding,
                                                 bias=model_instance.conv1.bias)
        self.model = model_instance

    def forward(self, x):
        return self.model(x)

    def forward_head(self, x):
        return self.model.forward_head(x)

    def forward_features(self, x):
        return self.model.forward_features(x)

    def get_classifier(self):
        return self.model.get_classifier()
