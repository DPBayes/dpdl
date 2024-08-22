import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn


def create_data_loaders(data, target, batch_size, train_split=0.8):
    """
    Splits the data into training and validation sets and creates data loaders.

    Args:
        data (torch.Tensor): Input data.
        target (torch.Tensor): Target labels.
        batch_size (int): Batch size for data loaders.
        train_split (float): Ratio of data to use for training (the rest is for validation).

    Returns:
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
    """
    N = data.shape[0]
    train_size = int(train_split * N)
    val_size = N - train_size

    # Create dataset
    dataset = TensorDataset(data, target)

    # Split into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class MeanEstimator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mean = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        return self.mean.expand_as(x)
