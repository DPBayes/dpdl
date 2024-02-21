import torch

"""

This is an implementation of the Wide ResNet used in the "Unlocking
high-accuracy..." (De et al., 2022).

With `depth=16` and `width=4` it creates the following network:

```
First_conv size=464 shapes={'bias': (16,), 'gain': (16,), 'w': (3, 3, 3, 16)}
Block_1Conv_0_0 size=9344 shapes={'bias': (64,), 'gain': (64,), 'w': (3, 3, 16, 64)}
Block_1Conv_0_1 size=36992 shapes={'bias': (64,), 'gain': (64,), 'w': (3, 3, 64, 64)}
Block_1Conv_1_0 size=36992 shapes={'bias': (64,), 'gain': (64,), 'w': (3, 3, 64, 64)}
Block_1Conv_1_1 size=36992 shapes={'bias': (64,), 'gain': (64,), 'w': (3, 3, 64, 64)}
Block_1_norm_0_0 size=32 shapes={'offset': (16,), 'scale': (16,)}
Block_1_norm_0_1 size=128 shapes={'offset': (64,), 'scale': (64,)}
Block_1_norm_1_0 size=128 shapes={'offset': (64,), 'scale': (64,)} Block_1_norm_1_1 size=128 shapes={'offset': (64,), 'scale': (64,)}
Block_1_skip_conv size=1152 shapes={'bias': (64,), 'gain': (64,), 'w': (1, 1, 16, 64)}
Block_1_skip_norm size=32 shapes={'offset': (16,), 'scale': (16,)}
Block_2Conv_0_0 size=73984 shapes={'bias': (128,), 'gain': (128,), 'w': (3, 3, 64, 128)}
Block_2Conv_0_1 size=147712 shapes={'bias': (128,), 'gain': (128,), 'w': (3, 3, 128, 128)}
Block_2Conv_1_0 size=147712 shapes={'bias': (128,), 'gain': (128,), 'w': (3, 3, 128, 128)}
Block_2Conv_1_1 size=147712 shapes={'bias': (128,), 'gain': (128,), 'w': (3, 3, 128, 128)}
Block_2_norm_0_0 size=128 shapes={'offset': (64,), 'scale': (64,)}
Block_2_norm_0_1 size=256 shapes={'offset': (128,), 'scale': (128,)}
Block_2_norm_1_0 size=256 shapes={'offset': (128,), 'scale': (128,)}
Block_2_norm_1_1 size=256 shapes={'offset': (128,), 'scale': (128,)}
Block_2_skip_conv size=8448 shapes={'bias': (128,), 'gain': (128,), 'w': (1, 1, 64, 128)}
Block_2_skip_norm size=128 shapes={'offset': (64,), 'scale': (64,)}
Block_3Conv_0_0 size=295424 shapes={'bias': (256,), 'gain': (256,), 'w': (3, 3, 128, 256)}
Block_3Conv_0_1 size=590336 shapes={'bias': (256,), 'gain': (256,), 'w': (3, 3, 256, 256)}
Block_3Conv_1_0 size=590336 shapes={'bias': (256,), 'gain': (256,), 'w': (3, 3, 256, 256)}
Block_3Conv_1_1 size=590336 shapes={'bias': (256,), 'gain': (256,), 'w': (3, 3, 256, 256)}
Block_3_norm_0_0 size=256 shapes={'offset': (128,), 'scale': (128,)}
Block_3_norm_0_1 size=512 shapes={'offset': (256,), 'scale': (256,)}
Block_3_norm_1_0 size=512 shapes={'offset': (256,), 'scale': (256,)}
Block_3_norm_1_1 size=512 shapes={'offset': (256,), 'scale': (256,)}
Block_3_skip_conv size=33280 shapes={'bias': (256,), 'gain': (256,), 'w': (1, 1, 128, 256)}
Block_3_skip_norm size=256 shapes={'offset': (128,), 'scale': (128,)}
Final_norm size=512 shapes={'offset': (256,), 'scale': (256,)}
Softmax size=2570 shapes={'b': (10,), 'w': (256, 10)}

Total number of parameters: 2753818
```

This output is taken from the `jax_privacy` code when running the experiments with WRN-16-4.

The JAX code produces 2753818 parameters.
"""

class WSConv2d(torch.nn.Module):
    """2D Convolutional layer with Weight Standardization and Gain parameter."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()

        # Standard convolutional layer without bias as bias is manually added later
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

        # Gain parameter for affine transformation, initialized to ones
        self.gain = torch.nn.Parameter(torch.ones(out_channels, 1, 1, 1))

        # Initialize weights using He initialization, suitable for ReLU activations
        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')

        if bias:
            torch.nn.init.zeros_(self.conv.bias)

    def standardize_weight(self, weight, eps=1e-5):
        """Applies Weight Standardization to the convolutional weights."""

        # Calculate mean and variance
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), keepdim=True)

        # Compute the scale factor using fan-in and gain
        fan_in = np.prod(weight.shape[1:])
        scale = torch.rsqrt(var * fan_in + eps) * self.gain

        # Standardize the weights
        return (weight - mean) * scale

    def forward(self, x):
        """Forward pass through the layer."""

        # Apply weight standardization to the weights
        weight = self.standardize_weight(self.conv.weight)

        # Perform the convolution operation without bias
        x = F.conv2d(x, weight, None, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        num_groups = 16

        self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.gn2 = torch.nn.GroupNorm(num_groups, out_channels)

        self.skip = torch.nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                WSConv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                torch.nn.GroupNorm(num_groups, out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        skip = self.skip(x)

        out += skip
        out = self.relu(out)
        return out

class Stage(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()

        self.blocks = torch.nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            *[ResidualBlock(out_channels, out_channels, 1) for _ in range(1, num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)

class WideResNet(torch.nn.Module):
    def __init__(self, depth, width, num_classes=10):
        super().__init__()

        self.in_channels = 16  # Initial in_channels

        num_blocks = (depth - 4) // 6

        self.first_conv = WSConv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = torch.nn.ReLU(inplace=True)

        self.stage1 = Stage(self.in_channels, 16 * width, stride=1, num_blocks=num_blocks)
        self.stage2 = Stage(16 * width, 32 * width, stride=2, num_blocks=num_blocks)
        self.stage3 = Stage(32 * width, 64 * width, stride=2, num_blocks=num_blocks)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64 * width, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
