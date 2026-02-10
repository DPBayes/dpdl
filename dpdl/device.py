import torch


def resolve_device(device: str | None) -> torch.device:
    if not device:
        device = 'cuda'

    device = device.lower()

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device in ('cuda', 'gpu'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available.')
        return torch.device('cuda')

    if device == 'cpu':
        return torch.device('cpu')

    raise ValueError(f'Unsupported device: {device!r}')


def distributed_backend(device: torch.device) -> str:
    return 'nccl' if device.type == 'cuda' else 'gloo'


def set_cuda_device(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.set_device(0)
