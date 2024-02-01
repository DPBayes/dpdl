import csv
import logging
import math
import os
import pathlib
import torch
import torchmetrics

from typing import List

from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)

class CallbackHandler:
    def __init__(self, callbacks: list = []):
        self.callbacks = callbacks

    def call(self, event, *args, **kwargs):
        for callback in self.callbacks:
            event_handler = getattr(callback, event)
            event_handler(*args, **kwargs)

class Callback:
    def _is_global_zero(self, trainer):
        return torch.distributed.get_rank() == 0
    def on_train_start(self, trainer):
        pass
    def on_train_end(self, trainer):
        pass
    def on_train_step(self, trainer):
        pass
    def on_train_epoch_start(self, trainer, epoch):
        pass
    def on_train_epoch_end(self, trainer, epoch, epoch_loss):
        pass
    def on_train_batch_start(self, trainer, batch_idx, batch):
        pass
    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        pass
    def on_validation_epoch_start(self, trainer, epoch):
        pass
    def on_validation_epoch_end(self, trainer, epoch, metrics):
        pass
    def on_validation_batch_start(self, trainer, batch_idx, batch):
        pass
    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        pass
    def on_test_epoch_start(self, trainer, epoch):
        pass
    def on_test_epoch_end(self, trainer, epoch, valid_loss, metrics):
        pass
    def on_test_batch_start(self, trainer, batch_idx, batch):
        pass
    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def _log_metrics(self, metrics, annotation='Metrics'):
        if not metrics:
            return

        log.info(annotation + ':')
        for key, value in metrics.items():
            log.info(f' - {key}: {value:.4f}.')

class RecordEpochStatsCallback(Callback):
    def __init__(self, use_steps=False):
        self.use_steps = use_steps

        self.train_loss = torchmetrics.aggregation.MeanMetric().cuda()
        self.evaluation_loss = torchmetrics.aggregation.MeanMetric(sync_on_compute=False).cuda()

    def on_train_start(self, trainer):
        if self._is_global_zero(trainer):
            if self.use_steps:
                batch_size = trainer.datamodule.batch_size
                data_size = len(trainer.get_dataloader('train').dataset)
                steps_per_epoch = data_size / batch_size
                epochs = math.ceil(trainer.total_steps / steps_per_epoch)

                log.info(f'!!! Starting training for approximately {epochs} epochs ({trainer.total_steps} steps).')
            else:
                log.info(f'!!! Starting training for {trainer.epochs} epochs.')

    def on_train_end(self, trainer):
        if self._is_global_zero(trainer):
            log.info('!!! Training finished.')

    def on_train_epoch_start(self, trainer, epoch):
        self.train_loss.reset()

        if self._is_global_zero(trainer):
            log.info(f'--------------------------------------------------')
            if not self.use_steps:
                log.info(f'Starting training epoch {epoch+1}.')
            else:
                log.info(f'Starting training approximate epoch {epoch+1}.')

    def on_train_epoch_end(self, trainer, epoch, metrics):
        loss = self.train_loss.compute()

        if self._is_global_zero(trainer):
            if not self.use_steps:
                log.info(f'Epoch {epoch+1} finished. Loss: {loss:.4f}.')
            else:
                log.info(f'Approximate epoch {epoch+1} finished. Loss: {loss:.4f}.')

            self._log_metrics(metrics, 'Train metrics')

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        self.train_loss.update(loss)

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if torch.distributed.get_rank() == 0:
            log.info(f'Validation finished. Loss: {loss:.4f}.')
            self._log_metrics(metrics, 'Validation metrics')

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)

    def on_test_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if torch.distributed.get_rank() == 0:
            log.info(f'Test finished. Loss: {loss:.4f}.')
            self._log_metrics(metrics, 'Test metrics')

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)

class RecordSNR(Callback):
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir
        self.grad_norms = []
        self.noise_norms = []
        self.snr_values = []

    def on_train_step(self, trainer):
        if torch.distributed.get_rank() == 0:
            grad_norm = trainer.optimizer._previous_grad.norm().item()
            noise_norm = trainer.optimizer._previous_noise.norm().item()
            snr = grad_norm / noise_norm if noise_norm != 0 else float('inf')  # NB: div by zero

            self.grad_norms.append(grad_norm)
            self.noise_norms.append(noise_norm)
            self.snr_values.append(snr)

            log.info(f'- Grad Mean: {grad_norm}, Noise Mean: {noise_norm}')

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            file_path = os.path.join(self.log_dir, 'signal-to-noise-ratio.csv')

            with open(file_path, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(['Step', 'Grad_Norm', 'Noise_Norm', 'SNR'])

                for step in range(len(self.grad_norms)):
                    writer.writerow([step, self.grad_norms[step], self.noise_norms[step], self.snr_values[step]])

            log.info(f'Signal-to-Noise ratio data saved at {file_path}')

class CallbackFactory:
    @staticmethod
    def get_callbacks(configuration: Configuration, hyperparams: Hyperparameters) -> List[Callback]:
        callbacks = [
            RecordEpochStatsCallback(use_steps=configuration.use_steps),
        ]

        if configuration.record_snr:
            log_dir = configuration.log_dir
            experiment_name = configuration.experiment_name
            full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

            callbacks.append(RecordSNR(log_dir=full_log_dir))

        return callbacks

