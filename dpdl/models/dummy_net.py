import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms


class DummyNet(nn.Module):
    """
    DummyNet for model loading testing

    """
    def __init__(self, parameter1=1, parameter2=2, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(1600, 500, bias=False)
        self.fc2 = nn.Linear(500, 500, bias=False)
        self.fc2 = nn.Linear(500, parameter2, bias=False)
        self.fc2 = nn.Linear(parameter2, parameter1, bias=False)
        self.final_fc = nn.Linear(parameter1, num_classes, bias=False)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final_fc(x)

        return x

    def get_classifier(self):
        return self.final_fc

    def get_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def save_model(self, fpath):
        torch.save(self.model.state_dict(), fpath)

