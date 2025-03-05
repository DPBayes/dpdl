import logging
import os
import torch

from .base_callback import Callback

log = logging.getLogger(__name__)


class CheckpointCallback(Callback):
    def __init__(self, log_dir: str, checkpoint_step_interval: int):
        self.log_dir = log_dir
        self.checkpoint_step_interval = checkpoint_step_interval
        self.global_step = 0
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')

        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save_checkpoint(self, trainer, checkpoint_path: str):
        torch.save(trainer._unwrap_model().state_dict(), checkpoint_path)

        log.info(f'Model checkpoint saved at {checkpoint_path}')

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, **kwargs):
        if not self._is_global_zero():
            return

        self.global_step += 1

        if self.global_step % self.checkpoint_step_interval == 0:
            checkpoint_path = os.path.join(
                self.checkpoints_dir, f'checkpoint_step_{self.global_step}.pt'
            )
            self.save_checkpoint(trainer, checkpoint_path)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            final_checkpoint_path = os.path.join(
                self.checkpoints_dir, 'final_checkpoint.pt'
            )
            self.save_checkpoint(trainer, final_checkpoint_path)
