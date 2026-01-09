import torch

from .configurationmanager import Configuration, Hyperparameters

from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup

class OptimizerFactory:
    @staticmethod
    def get_optimizer(configuration: Configuration, hyperparams: Hyperparameters, model: torch.nn.Module):
        optimizer_cls = getattr(torch.optim, configuration.optimizer)
        optimizer = optimizer_cls(model.parameters(), lr=hyperparams.learning_rate)
        return optimizer

    @staticmethod
    def get_scheduler(configuration: Configuration, hyperparams: Hyperparameters, optimizer: torch.optim.Optimizer, total_steps:int):
        """
        Get learning rate scheduler with warmup and decay.
        
        Expects configuration to have:
        - use_scheduler: bool (whether to use a scheduler)
        - scheduler_type: str (e.g., 'cosine', 'linear')
        - warmup_steps: int
        - total_steps: int (or max_steps)
        """
        
        scheduler_type = configuration.scheduler_type
        warmup_steps = int(total_steps*0.15)

        if total_steps is None:
            raise ValueError("total_steps must be specified in configuration for scheduler")
        
        if scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        return scheduler