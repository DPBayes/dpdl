import torch

from torch.optim import Optimizer

from .configurationmanager import Configuration, Hyperparameters


class BSRExampleSGD(Optimizer):
    '''
    PyTorch-like SGD, but with:
      - EMA momentum: m = beta*m + (1-beta)*grad
      - Multiplicative decay: p = alpha*p - lr*m

    This is to exactly match the BSR example code from Nikita.
    '''

    def __init__(self, params, lr=1e-3, beta=0.0, alpha=1.0):
        if lr <= 0:
            raise ValueError('lr must be > 0')

        if not (0.0 <= beta < 1.0):
            raise ValueError('beta must be in [0, 1)')

        if not (0.0 < alpha <= 1.0):
            raise ValueError('alpha must be in (0, 1]')

        defaults = dict(lr=lr, beta=beta, alpha=alpha)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                buf = state['momentum_buffer']
                buf.mul_(beta).add_(g, alpha=(1.0 - beta))  # EMA momentum

                p.mul_(alpha).add_(buf, alpha=-lr)  # multiplicative decay + step

        return loss


class OptimizerFactory:
    @staticmethod
    def get_optimizer(configuration: Configuration, hyperparams: Hyperparameters, model: torch.nn.Module):
        optimizer_name = str(configuration.optimizer)

        if optimizer_name == 'bsr-example-sgd':
            beta = configuration.bsr_beta if configuration.bsr_beta is not None else 0.0
            alpha = configuration.bsr_alpha if configuration.bsr_alpha is not None else 1.0

            return BSRExampleSGD(
                model.parameters(),
                lr=hyperparams.learning_rate,
                beta=beta,
                alpha=alpha,
            )

        optimizer_cls = getattr(torch.optim, configuration.optimizer)
        optimizer = optimizer_cls(model.parameters(), lr=hyperparams.learning_rate)

        return optimizer
