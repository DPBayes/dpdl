import logging
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
    def on_validation_epoch_end(self, trainer, epoch, valid_loss, metrics):
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

    def _log_metrics(self, metrics):
        if not metrics:
            return

        log.info('Metrics:')
        for key, value in metrics.items():
            log.info(f' - {key}: {value:.4f}.')

class RecordEpochLossCallback(Callback):
    def __init__(self):
        # use torchmetrics mean aggregation to track the losses
        self.train_loss = torchmetrics.aggregation.MeanMetric().cuda()

        # for validation and test sets
        self.evaluation_loss = torchmetrics.aggregation.MeanMetric().cuda()

    def on_train_start(self, trainer):
        if self._is_global_zero(trainer):
            log.info(f'!!! Starting training for {trainer.epochs} epochs.')

    def on_train_end(self, trainer):
        if self._is_global_zero(trainer):
            log.info('!!! Training finished.')

    def on_train_epoch_start(self, trainer, epoch):
        if self._is_global_zero(trainer):
            log.info(f'----------------------------------')
            log.info(f'Starting training epoch {epoch+1}.')

    def on_train_epoch_end(self, trainer, epoch, metrics):
        loss = self.train_loss.compute()
        self.train_loss.reset()

        if self._is_global_zero(trainer):
            log.info(f'Epoch {epoch+1} finished. Loss: {loss:.4f}.')
            self._log_metrics(metrics)

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if self._is_global_zero(trainer):
            if epoch:
                log.info(f'Validation epoch {epoch+1} finished. Loss: {loss:.4f}.')
            else:
                log.info(f'Validation finished. Loss: {loss:.4f}.')

            self._log_metrics(metrics)

    def on_test_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if self._is_global_zero(trainer):
            log.info(f'Test finished. Loss: {loss:.4f}.')
            self._log_metrics(metrics)

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        self.train_loss.update(loss)

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)

class CallbackFactory:
    @staticmethod
    def get_callbacks(configuration: Configuration, hyperparams: Hyperparameters) -> List[Callback]:
        callbacks = [
            RecordEpochLossCallback(),
        ]

        return callbacks

