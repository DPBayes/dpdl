import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms


class VGGBnBReferenceModel(nn.Module):
    """
    PyTorch version of the JAX `build_vgg_model` used in multi-epoch MF,
    and more imporantly in many of the follow-up papers.

    Reference defaults:
    - channels: [32, 64, 128]
    - dense_size: 128
    - activation: tanh
    - expected input: NCHW, 3x32x32
    """

    def __init__(
        self,
        num_classes: int,
        channels: list[int] | tuple[int, int, int] = (32, 64, 128),
        dense_size: int = 128,
        activation: str = "tanh",
        input_size: int = 32,
    ):
        super().__init__()

        if len(channels) != 3:
            raise ValueError(
                f"VGGBnBReferenceModel requires exactly 3 channels, got {channels!r}."
            )

        self.channels = [int(c) for c in channels]
        self.dense_size = int(dense_size)
        self.activation_name = str(activation)
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)

        if self.input_size < 8 or self.input_size % 8 != 0:
            raise ValueError("--vgg-ref-input-size must be divisible by 8 and >= 8.")

        self.conv1 = nn.Conv2d(3, self.channels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            self.channels[0], self.channels[0], kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            self.channels[1], self.channels[1], kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            self.channels[2], self.channels[2], kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        spatial = self.input_size // 8
        self.fc1 = nn.Linear(self.channels[2] * spatial * spatial, self.dense_size)
        self.fc2 = nn.Linear(self.dense_size, self.num_classes)

        self._initialize_weights()

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "tanh":
            return torch.tanh(x)

        if self.activation_name == "relu":
            return F.relu(x)

        raise ValueError(
            f'Unsupported VGG reference activation "{self.activation_name}". '
            "Allowed: tanh, relu."
        )

    def _initialize_weights(self) -> None:
        # Initialize weights the same way as in DP-FTRL code.
        # conv: N(0, sqrt(1/fan_in)); linear: Xavier normal; all biases zero.
        # NB: I don't optimal for ReLU, but they use tanh.
        convs = [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
        ]
        for conv in convs:
            fan_in = int(conv.kernel_size[0] * conv.kernel_size[1] * conv.in_channels)
            std = float(torch.sqrt(torch.tensor(1.0 / fan_in)).item())

            with torch.no_grad():
                conv.weight.normal_(0.0, std)

            nn.init.zeros_(conv.bias)

        fcs = [self.fc1, self.fc2]
        for fc in fcs:
            nn.init.xavier_normal_(fc.weight)
            nn.init.zeros_(fc.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D NCHW input, got shape {tuple(x.shape)}.")

        if x.shape[1] != 3:
            raise ValueError(
                f"Expected RGB input with 3 channels, got input shape {tuple(x.shape)}."
            )

        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            raise ValueError(
                f"Expected input spatial shape {self.input_size}x{self.input_size}, "
                f"got {x.shape[2]}x{x.shape[3]}."
            )

        x = self._activation(self.conv1(x))
        x = self._activation(self.conv2(x))
        x = self.pool1(x)

        x = self._activation(self.conv3(x))
        x = self._activation(self.conv4(x))
        x = self.pool2(x)

        x = self._activation(self.conv5(x))
        x = self._activation(self.conv6(x))
        x = self.pool3(x)

        x = self.flatten(x)
        x = self._activation(self.fc1(x))
        x = self.fc2(x)

        return x

    def get_classifier(self):
        return self.fc2

    def get_transforms(self):
        # CIFAR preprocessing in JAX reference is x / 127.5 - 1.0.
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
