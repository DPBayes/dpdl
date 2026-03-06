import torch
from torch.optim import Optimizer

from .configurationmanager import Configuration, Hyperparameters


class PaperSGD(Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 < float(weight_decay) <= 1.0):
            raise ValueError(f"Invalid paper weight_decay: {weight_decay}")
        if not (0.0 <= float(momentum) < 1.0):
            raise ValueError(f"Invalid paper momentum: {momentum}")

        defaults = {
            'lr': float(lr),
            'paper_alpha': float(weight_decay),
            'momentum': float(momentum),
            'weight_decay': float(weight_decay),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group['lr'])
            alpha = float(group['paper_alpha'])
            beta = float(group['momentum'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                momentum_buffer = state.get('momentum_buffer')
                if momentum_buffer is None:
                    momentum_buffer = state['momentum_buffer'] = torch.zeros_like(p)

                momentum_buffer.mul_(beta).add_(grad)
                p.mul_(alpha).add_(momentum_buffer, alpha=-lr)

        return loss


class OptimizerFactory:
    @staticmethod
    def get_optimizer(configuration: Configuration, hyperparams: Hyperparameters, model: torch.nn.Module):
        optimizer_name = str(configuration.optimizer)

        if optimizer_name == 'paper-sgd':
            momentum = (
                float(configuration.optimizer_momentum)
                if configuration.optimizer_momentum is not None
                else 0.0
            )
            weight_decay = float(configuration.optimizer_weight_decay)
            return PaperSGD(
                model.parameters(),
                lr=hyperparams.learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )

        optimizer_cls = getattr(torch.optim, configuration.optimizer)
        optimizer_kwargs = {'lr': hyperparams.learning_rate}

        if optimizer_name.lower() == 'sgd':
            if configuration.optimizer_momentum is not None:
                optimizer_kwargs['momentum'] = float(configuration.optimizer_momentum)

            if configuration.optimizer_weight_decay is not None:
                optimizer_kwargs['weight_decay'] = float(configuration.optimizer_weight_decay)

        optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        return optimizer
