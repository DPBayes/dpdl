import torch
import logging

from typing import List

log = logging.getLogger(__name__)

class CallbackHandler():
    def __init__(self, callbacks: list = []):
        self.callbacks = callbacks

    def call(self, event, *args, **kwargs):
        for callback in self.callbacks:
            event_handler = getattr(callback, event)
            event_handler(*args, **kwargs)

class Callback():
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
    def on_validation_epoch_end(self, trainer, epoch, valid_loss):
        pass
    def on_validation_batch_start(self, trainer, batch_idx, batch):
        pass
    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        pass

class PrintStateCallback(Callback):
    def on_train_start(self, trainer):
        if self._is_global_zero(trainer):
            log.info(f'!!! Starting training for {trainer.epochs} epochs.')

    def on_train_end(self, trainer):
        if self._is_global_zero(trainer):
            log.info('!!! Training finished.')

    def on_train_epoch_start(self, trainer, epoch):
        if self._is_global_zero(trainer):
            log.info(f'Starting training epoch {epoch+1}.')

    def on_train_epoch_end(self, trainer, epoch, loss):
        if self._is_global_zero(trainer):
            log.info(f'Epoch {epoch+1} finished. Loss: {loss:.4f}.')

#    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
#        if self._is_global_zero(trainer):
#            log.info(f'  - Processed batch {batch_idx+1}, loss: {loss:.4f}')

#    def on_validation_epoch_start(self, trainer, epoch):
#        if self._is_global_zero(trainer):
#            if epoch:
#                log.info(f' - Starting validation epoch {epoch+1}.')
#            else:
#                log.info(f' - Starting validation.')

    def on_validation_epoch_end(self, trainer, epoch, loss):
        if self._is_global_zero(trainer):
            if epoch:
                log.info(f'Validation epoch {epoch+1} finished. Loss: {loss:.4f}.')
            else:
                log.info(f'Validation finished. Loss: {loss:.4f}.')

class CallbackFactory():
    @staticmethod
    def get_callbacks(configuration: dict, hyperparams: dict) -> List[Callback]:
        callbacks = [
            PrintStateCallback(),
        ]

        return callbacks

