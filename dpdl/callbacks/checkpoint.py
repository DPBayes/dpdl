import json
import logging
import os
import re

import torch
import torchmetrics

from ..utils import tensor_to_python_type
from .base_callback import Callback

log = logging.getLogger(__name__)

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by modification time"""

    if not os.path.exists(checkpoint_dir):
        return 0

    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint_step_')]

    if not checkpoints:
        return 0

    # Sort by modification time
    latest = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))

    # Extract step number
    match = re.search(r'checkpoint_step_(\d+)', latest)

    if match:
        return int(match.group(1))

    return 0


class CheckpointCallback(Callback):
    def __init__(self, log_dir: str, checkpoint_step_interval: int, device=None):
        super().__init__()

        self.log_dir = log_dir
        self.checkpoint_step_interval = checkpoint_step_interval
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')
        self.global_step = get_latest_checkpoint(self.checkpoints_dir)

        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Initialize mean metric for accumulating train loss over interval
        device = device or torch.device('cuda')
        self.interval_loss = torchmetrics.aggregation.MeanMetric(sync_on_compute=False).to(device)

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, **kwargs):
        is_rank_zero = self._is_global_zero()
        is_dist = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

        # Rank 0 owns the step counter and decides whether to checkpoint.
        # NOTE: non-zero ranks cannot return early — they must all reach the
        # trainer.validate() call below, because a generation-eval task (e.g.
        # DiseaseTask) issues an all_reduce inside validate(); any rank that
        # skipped would deadlock the collective.
        if is_rank_zero:
            self.interval_loss.update(loss)
            self.global_step += 1
            should_save = (self.global_step % self.checkpoint_step_interval == 0)
            current_step = self.global_step
        else:
            should_save = False
            current_step = 0

        # Broadcast trigger + step number so all ranks agree and branch the same.
        if is_dist:
            info = torch.tensor(
                [int(should_save), current_step], dtype=torch.int64, device=trainer.device,
            )
            torch.distributed.broadcast(info, src=0)
            should_save = bool(info[0].item())
            current_step = int(info[1].item())

        if not should_save:
            return

        checkpoint_path = os.path.join(
            self.checkpoints_dir, f'checkpoint_step_{current_step}.pt'
        )

        # Non-FSDP DDP: only rank 0 writes (other ranks hold identical weights).
        if is_rank_zero:
            self.save_checkpoint(trainer, checkpoint_path)

        # All ranks must call validate() — see note above.
        trainer.validate(enable_callbacks=False)

        if not is_rank_zero:
            return

        metrics = trainer._unwrap_model().valid_metrics.compute()
        trainer._unwrap_model().valid_metrics.reset()
        avg_train_loss = self.interval_loss.compute().item()
        self.interval_loss.reset()
        metrics = {
            'loss': loss,
            'avg_train_loss_since_last_checkpoint': avg_train_loss,
            **metrics,
        }
        metrics_path = os.path.join(
            self.checkpoints_dir, f'checkpoint_step_{current_step}_metrics.json'
        )
        self.save_metrics(metrics, metrics_path)

    def on_train_end(self, trainer, *args, **kwargs):
        is_rank_zero = self._is_global_zero()
        is_dist = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

        # Broadcast the step number so non-rank-0 ranks construct the same path.
        # Every rank must participate in validate() (see on_train_batch_end).
        if is_dist:
            step_tensor = torch.tensor(
                [self.global_step if is_rank_zero else 0],
                dtype=torch.int64, device=trainer.device,
            )
            torch.distributed.broadcast(step_tensor, src=0)
            final_step = int(step_tensor[0].item())
        else:
            final_step = self.global_step

        final_checkpoint_path = os.path.join(
            self.checkpoints_dir, f'final_checkpoint_step_{final_step}.pt'
        )

        if is_rank_zero:
            self.save_checkpoint(trainer, final_checkpoint_path)

        # All ranks must call validate() — see on_train_batch_end.
        trainer.validate(enable_callbacks=False)

        if not is_rank_zero:
            return

        metrics = trainer._unwrap_model().valid_metrics.compute()
        trainer._unwrap_model().valid_metrics.reset()
        avg_train_loss = self.interval_loss.compute().item()
        self.interval_loss.reset()
        metrics['avg_train_loss_since_last_checkpoint'] = avg_train_loss
        metrics_path = os.path.join(
            self.checkpoints_dir, f'final_checkpoint_step_{final_step}_metrics.json'
        )
        self.save_metrics(metrics, metrics_path)

    def save_checkpoint(self, trainer, checkpoint_path: str):
        trainer.save_model(checkpoint_path)
        log.info(f'Model checkpoint saved at {checkpoint_path}')

    def save_metrics(self, metrics, metrics_path: str):
        metrics = tensor_to_python_type(metrics)

        with open(metrics_path, 'w') as fh:
            json.dump(metrics, fh)

        log.info(f'Model checkpoint metrics saved at {metrics_path}')
