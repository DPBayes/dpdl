import torch

class OptimizerFactory:
    @staticmethod
    def get_optimizer(configuration: dict, hyperparams: dict, model: torch.nn.Module):
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        return optimizer

