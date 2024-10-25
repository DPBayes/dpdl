import csv
import os
import torch
from dpdl.callbacks.base_callback import Callback
import logging

log = logging.getLogger(__name__)


class RecordSNR(Callback):
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir
        self.grad_norms = []
        self.noise_norms = []
        self.snr_values = []

    def on_train_batch_end(self, trainer):
        grad_norm = trainer.optimizer._previous_grad.norm().item()
        noise_norm = trainer.optimizer._previous_noise.norm().item()
        snr = (
            grad_norm / noise_norm if noise_norm != 0 else float("inf")
        )  # NB: div by zero

        self.grad_norms.append(grad_norm)
        self.noise_norms.append(noise_norm)
        self.snr_values.append(snr)

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            file_path = os.path.join(self.log_dir, "signal-to-noise-ratio.csv")

            with open(file_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["Step", "Grad_Norm", "Noise_Norm", "SNR"])

                for step in range(len(self.grad_norms)):
                    writer.writerow(
                        [
                            step,
                            self.grad_norms[step],
                            self.noise_norms[step],
                            self.snr_values[step],
                        ]
                    )

            log.info(f"Signal-to-Noise ratio data saved at {file_path}")
