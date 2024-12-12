import csv
import logging
import os

from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordTrainLossByStepCallback(Callback):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.train_losses = []

        os.makedirs(self.log_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, **kwargs):
        self.train_losses.append({'step': batch_idx, 'train_loss': loss})

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            train_loss_path = os.path.join(self.log_dir, 'train_loss_by_step.csv')

            with open(train_loss_path, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=['step', 'train_loss'])
                writer.writeheader()
                writer.writerows(self.train_losses)

            log.info(f'Training losses (by step) saved to {train_loss_path}')

