import torch

from .configurationmanager import Configuration, Hyperparameters

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
            weight_decay = float(configuration.optimizer_weight_decay or 0.0)

            return torch.optim.SGD(
                model.parameters(),
                lr=hyperparams.learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=False,
                dampening=0.0,
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
