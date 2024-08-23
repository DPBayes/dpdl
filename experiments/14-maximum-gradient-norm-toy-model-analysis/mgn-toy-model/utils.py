import torch
import numpy as np
import random
import os
from scipy.stats import skewnorm


def seed_everything(seed):
    """Set random seed for reproducibility across libraries."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_skew_normal_data(n, d, skewness=4):
    # Generate skew-normal distributed data
    alpha = skewness * np.ones(d)  # Same skewness for each dimension
    data = skewnorm.rvs(a=alpha, loc=0, scale=1, size=(n, d))

    # Make it centered by subtracting the mean
    data -= np.mean(data, axis=0)

    return torch.tensor(data, dtype=torch.float32)

def generate_mixture_data(
    n,
    d,
    mixture_dim=0,
    weight=0.9,
    mean1=0,
    mean2=10,
    cov1=1,
    cov2=0.5,
):
    """
    Generate a dataset where one dimension follows a mixture of two Gaussians,
    and the remaining dimensions follow standard Gaussian distributions.
    """
    # Generate standard Gaussian data for all dimensions
    data = torch.normal(mean=0, std=1, size=(n, d))

    # Generate the mixture component for the specified dimension
    n1 = int(n * weight)
    n2 = n - n1

    # Samples for the mixture dimension
    mixture_data1 = torch.normal(mean=mean1, std=cov1, size=(n1,))
    mixture_data2 = torch.normal(mean=mean2, std=cov2, size=(n2,))

    # Concatenate and shuffle the mixture data
    mixture_data = torch.cat([mixture_data1, mixture_data2], dim=0)
    mixture_data = mixture_data[torch.randperm(mixture_data.size(0))]

    # Replace the specified dimension with the mixture data
    data[:, mixture_dim] = mixture_data

    return data


def setup_directories(data_dir, image_dir):
    """Create required directories for saving results if they don't exist."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
