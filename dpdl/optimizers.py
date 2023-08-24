import torch

from .configurationmanager import Configuration, Hyperparameters

class OptimizerFactory:
    @staticmethod
    def get_optimizer(configuration: Configuration, hyperparams: Hyperparameters, model: torch.nn.Module):
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
        return optimizer

