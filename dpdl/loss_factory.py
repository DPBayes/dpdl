import torch

from .configurationmanager import Configuration, Hyperparameters


class LossFactory:
    @staticmethod
    def get_loss(configuration: Configuration):
        loss_cls = getattr(torch.nn, configuration.loss_function)
        return loss_cls()
