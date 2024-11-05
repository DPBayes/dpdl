import torch
import logging
from dpdl.callbacks.base_callback import Callback

log = logging.getLogger(__name__)

def is_global_zero():
    log.info(f"[DEBUG] Calling _is_global_zero")
    return torch.distributed.get_rank() == 0

def print_memory_usage(event):
    if is_global_zero():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved(i) / (1024**2)  # MB
            log.info(f"[DEBUG] {event} - GPU {i} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

class DebugProbeCallback(Callback):
    def on_train_start(self, trainer):
        print_memory_usage("on_train_start")
        log.info(f"[DEBUG] on_train_start")

    def on_train_end(self, trainer):
        print_memory_usage("on_train_end")
        log.info(f"[DEBUG] on_train_end")

    def on_train_epoch_start(self, trainer, epoch):
        print_memory_usage("on_train_epoch_start")
        log.info(f"[DEBUG] on_train_epoch_start")

    def on_train_epoch_end(self, trainer, epoch, epoch_loss):
        print_memory_usage("on_train_epoch_end")
        log.info(f"[DEBUG] on_train_epoch_end")

    def on_train_batch_start(self, trainer, batch_idx, batch):
        print_memory_usage("on_train_batch_start")
        log.info(f"[DEBUG] on_train_batch_start")

    def on_train_physical_batch_start(self, trainer, batch_idx, batch):
        print_memory_usage("on_train_physical_batch_start")
        log.info(f"[DEBUG] on_train_physical_batch_start")

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        print_memory_usage("on_train_batch_end")
        log.info(f"[DEBUG] on_train_batch_end")

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, loss):
        print_memory_usage("on_train_physical_batch_end")
        log.info(f"[DEBUG] on_train_physical_batch_end")

    def on_validation_epoch_start(self, trainer, epoch):
        print_memory_usage("on_validation_epoch_start")
        log.info(f"[DEBUG] on_validation_epoch_start")

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        print_memory_usage("on_validation_epoch_end")
        log.info(f"[DEBUG] on_validation_epoch_end")

    def on_validation_batch_start(self, trainer, batch_idx, batch):
        print_memory_usage("on_validation_batch_start")
        log.info(f"[DEBUG] on_validation_batch_start")

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        print_memory_usage("on_validation_batch_end")
        log.info(f"[DEBUG] on_validation_batch_end")

    def on_test_epoch_start(self, trainer, epoch):
        print_memory_usage("on_test_epoch_start")
        log.info(f"[DEBUG] on_test_epoch_start")

    def on_test_epoch_end(self, trainer, epoch, metrics):
        print_memory_usage("on_test_epoch_end")
        log.info(f"[DEBUG] on_test_epoch_end")

    def on_test_batch_start(self, trainer, batch_idx, batch):
        print_memory_usage("on_test_batch_start")
        log.info(f"[DEBUG] on_test_batch_start")

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        print_memory_usage("on_test_batch_end")
        log.info(f"[DEBUG] on_test_batch_end")
