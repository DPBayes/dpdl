import json
import logging
import os
import re

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
    def __init__(self, log_dir: str, checkpoint_step_interval: int):
        super().__init__()

        self.log_dir = log_dir
        self.checkpoint_step_interval = checkpoint_step_interval
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')
        self.global_step = get_latest_checkpoint(self.checkpoints_dir)

        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Initialize mean metric for accumulating train loss over interval
        self.interval_loss = torchmetrics.aggregation.MeanMetric(sync_on_compute=False).cuda()

    def on_train_start(self, trainer):
        # Don't reset global_step - we want to keep the loaded checkpoint step
        print(f'on_train_start: self.global_step = {self.global_step}')


    def on_train_batch_end(self, trainer, batch_idx, batch, loss, **kwargs):

        if not self._is_global_zero():
            return

        # Update loss aggregator with current batch loss
        self.interval_loss.update(loss)
        self.global_step += 1

        if self.global_step % self.checkpoint_step_interval == 0:
            checkpoint_path = os.path.join(
                self.checkpoints_dir, f'checkpoint_step_{self.global_step}.pt'
            )

            self.save_checkpoint(trainer, checkpoint_path)

            trainer.validate(enable_callbacks=False)
            metrics = trainer._unwrap_model().valid_metrics.compute()
            trainer._unwrap_model().valid_metrics.reset()

            # Compute the average train loss since the last checkpoint
            avg_train_loss = self.interval_loss.compute().item()
            self.interval_loss.reset()

            # Add the average train loss to the metrics dictionary
            metrics = {
                'loss': loss,
                'avg_train_loss_since_last_checkpoint': avg_train_loss,
                **metrics,
            }

            metrics_path = os.path.join(
                self.checkpoints_dir, f'checkpoint_step_{self.global_step}_metrics.json'
            )
            self.save_metrics(metrics, metrics_path)

    def on_train_end(self, trainer, *args, **kwargs):
        if not self._is_global_zero():
            return

        final_checkpoint_path = os.path.join(
            self.checkpoints_dir, f'final_checkpoint_step_{self.global_step}.pt'
        )
        self.save_checkpoint(trainer, final_checkpoint_path)

        trainer.validate(enable_callbacks=False)
        metrics = trainer._unwrap_model().valid_metrics.compute()
        trainer._unwrap_model().valid_metrics.reset()

        # Compute avg loss since last checkpoint
        avg_train_loss = self.interval_loss.compute().item()
        self.interval_loss.reset()

        metrics['avg_train_loss_since_last_checkpoint'] = avg_train_loss

        metrics_path = os.path.join(
            self.checkpoints_dir, f'final_checkpoint_step_{self.global_step}_metrics.json'
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
