import torch

from .configurationmanager import Configuration, Hyperparameters

class LossFactory:
    @staticmethod
    def get_loss(configuration: Configuration):
        # if configuration.task == 'CausalLM':
        #     return CausalLMLoss()
        loss_cls = getattr(torch.nn, configuration.loss_function)
        return loss_cls()