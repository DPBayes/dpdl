import torch

from .configurationmanager import Configuration, Hyperparameters

# class CausalLMLoss(torch.nn.Module):
#     def __init__(self, ignore_index=-100):
#         super().__init__()
#         self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
#     def forward(self, logits, labels):
#         # Shift for next-token prediction
#         shift_logits = logits[:, :-1, :].contiguous()
#         shift_labels = labels[:, 1:].contiguous()
        
#         # Compute loss
#         loss = self.loss_fn(
#             shift_logits.view(-1, shift_logits.size(-1)),
#             shift_labels.view(-1)
#         )
#         return loss

class LossFactory:
    @staticmethod
    def get_loss(configuration: Configuration):
        # if configuration.task == 'CausalLM':
        #     return CausalLMLoss()
        loss_cls = getattr(torch.nn, configuration.loss_function)
        return loss_cls()