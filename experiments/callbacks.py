class PrintStateCallback():
    def on_train_start(self, trainer):
        print(f'Starting training for {trainer.max_epochs} epochs.')

    def on_train_end(self, trainer):
        print('Training finished.')

    def on_train_epoch_start(self, trainer, epoch):
        print(f' - Starting epoch {epoch+1}.')

    def on_train_epoch_end(self, trainer, epoch, loss):
        print(f' - Epoch finished, loss: {loss:.3f}.')

    #def on_train_batch_start(self, trainer, batch_idx, batch):
    #    print(f'  - Start processing batch {batch_idx+1}.')

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        print(f'  - Processed batch {batch_idx+1}, loss: {loss:.3f}')

