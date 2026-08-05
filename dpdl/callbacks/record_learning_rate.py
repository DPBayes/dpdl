import csv
import logging
import os

from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordLearningRateByEpochCallback(Callback):
    """
    Record the learning rate at the end of every training epoch and write the
    values to `<log_dir>/epoch_learning_rate.csv`. If the optimizer has
    multiple parameter groups, one column per group is emitted.
    """

    def __init__(self, log_dir: str):
        super().__init__()

        self.log_dir = log_dir
        self.csv_path = None
        self.learning_rates = []

    def on_train_start(self, trainer):
        if not self._is_global_zero():
            return

        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, 'epoch_learning_rate.csv')

    def on_train_epoch_end(self, trainer, epoch, metrics):
        if not self._is_global_zero():
            return

        if trainer.scheduler is not None:
            lrs = trainer.scheduler.get_last_lr()
        else:
            lrs = [group['lr'] for group in trainer.optimizer.param_groups]

        self.learning_rates.append(list(lrs))

    def on_train_end(self, trainer):
        if not self._is_global_zero() or not self.learning_rates:
            return

        num_groups = max(len(lrs) for lrs in self.learning_rates)

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            if num_groups == 1:
                writer.writerow(['epoch', 'learning_rate'])
            else:
                writer.writerow(
                    ['epoch'] + [f'learning_rate_group_{i}' for i in range(num_groups)]
                )

            for i, lrs in enumerate(self.learning_rates):
                writer.writerow([i + 1] + lrs)

        log.info(f'Per epoch learning rates written to {self.csv_path}')
