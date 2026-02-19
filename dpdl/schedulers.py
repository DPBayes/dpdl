import torch

from .configurationmanager import Configuration

def bnb_linear_decay_factor(
    *,
    step: int,
    total_steps: int,
    final_ratio: float = 1.0 / 20.0,
) -> float:
    """
    Piecewise LR schedule used in the balls-in-bins paper:
    - constant base LR for the first quarter of training
    - linear decay to final_ratio during the middle half
    - constant final_ratio for the last quarter
    """
    if total_steps <= 0:
        raise ValueError('total_steps must be > 0 for bnb_linear_decay.')

    decay_start = total_steps // 4
    decay_end = (3 * total_steps) // 4

    if step <= decay_start:
        return 1.0

    if step >= decay_end:
        return float(final_ratio)

    span = max(1, decay_end - decay_start)
    progress = float(step - decay_start) / float(span)

    return 1.0 - progress * (1.0 - float(final_ratio))


class SchedulerFactory:
    @staticmethod
    def get_scheduler(
        configuration: Configuration,
        optimizer: torch.optim.Optimizer,
        total_steps: int | None,
    ) -> torch.optim.lr_scheduler.LambdaLR | None:
        scheduler_name = str(configuration.lr_scheduler)

        if scheduler_name == 'none':
            return None

        if scheduler_name == 'bnb_linear_decay':
            if total_steps is None:
                raise ValueError('bnb_linear_decay requires a resolved total_steps value.')

            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: bnb_linear_decay_factor(
                    step=int(step),
                    total_steps=int(total_steps),
                ),
            )

        raise ValueError(f'Unknown lr scheduler: {scheduler_name}')
