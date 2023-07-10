import torch

class Callback():
    def _is_global_zero(self, trainer):
        return trainer.fabric.is_global_zero
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
            print(f'!!! Starting training for {trainer.epochs} epochs.')

    def on_train_end(self, trainer):
        if self._is_global_zero(trainer):
            print('!!! Training finished.')

    def on_train_epoch_start(self, trainer, epoch):
        if self._is_global_zero(trainer):
            print(f' - Starting training epoch {epoch+1}.')

    def on_train_epoch_end(self, trainer, epoch, loss):
        if self._is_global_zero(trainer):
            print(f' - Epoch {epoch+1} finished. Loss: {loss:.4f}.')

#    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
#        if self._is_global_zero(trainer):
#            print(f'  - Processed batch {batch_idx+1}, loss: {loss:.4f}')

#    def on_validation_epoch_start(self, trainer, epoch):
#        if self._is_global_zero(trainer):
#            if epoch:
#                print(f' - Starting validation epoch {epoch+1}.')
#            else:
#                print(f' - Starting validation.')

    def on_validation_epoch_end(self, trainer, epoch, loss):
        if self._is_global_zero(trainer):
            if epoch:
                print(f' - Validation epoch {epoch+1} finished. Loss: {loss:.4f}.')
            else:
                print(f' - Validation finished. Loss: {loss:.4f}.')
