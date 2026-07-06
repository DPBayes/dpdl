import torch
import logging
from dpdl.callbacks.base_callback import Callback

log = logging.getLogger(__name__)


class DebugProbeCallback(Callback):
    def __init__(self):
        super().__init__()

    def _is_global_zero(self):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] Calling _is_global_zero")
        super().__is_global_zero()

    def on_train_start(self, trainer):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_start")

    def on_train_end(self, trainer):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_end")

    def on_train_epoch_start(self, trainer, epoch):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_epoch_start")

    def on_train_epoch_end(self, trainer, epoch, epoch_loss):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_epoch_end")

    def on_train_batch_start(self, trainer, batch_idx, batch):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_batch_start")

    def on_train_physical_batch_start(self, trainer, batch_idx, batch):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_physical_batch_start")

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_batch_end")

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, loss):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_physical_batch_end")

    def on_validation_epoch_start(self, trainer, epoch):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_validation_epoch_start")

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_validation_epoch_end")

    def on_validation_batch_start(self, trainer, batch_idx, batch):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_validation_batch_start")

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_validation_batch_end")

    def on_test_epoch_start(self, trainer, epoch):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_test_epoch_start")

    def on_test_epoch_end(self, trainer, epoch, metrics):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_test_epoch_end")

    def on_test_batch_start(self, trainer, batch_idx, batch):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_test_batch_start")

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        log.debug(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_test_batch_end")
