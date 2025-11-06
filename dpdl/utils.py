import logging
import numpy as np
import random
import torch

log = logging.getLogger(__name__)

def seed_everything(seed) -> None:
    if not seed:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tensor_to_python_type(data):
    """
    Recursively converts all tensors in a dictionary or list to Python native types.
    """
    if isinstance(data, torch.Tensor):
        return data.item() if data.numel() == 1 else data.tolist()
    elif isinstance(data, dict):
        return {key: tensor_to_python_type(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [tensor_to_python_type(item) for item in data]
    else:
        return data

def shift_and_flatten(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels_flat = shift_labels.view(-1)

    return shift_logits_flat, shift_labels_flat
