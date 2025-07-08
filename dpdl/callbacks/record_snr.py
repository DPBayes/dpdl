import csv
import logging
import os
from typing import List

import torch
from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordSNRCallback(Callback):
    """
    Log gradient-norm statistics and two SNR variants for every logical batch.
    step              – global step from the trainer
    mean_norm         – E[||g||] (pre-clip, original scale)
    q{pct}_norm       – chosen quantile of ||g||
    clip_fraction     – P(||g|| > C)
    raw_snr           – (mean_norm / C) / sigma
    eff_snr           – E[min(||g||/C, 1)] / sigma
    """

    def __init__(self, log_dir: str, max_grad_norm: float, quantile: float = 0.90):
        super().__init__()
        self.log_dir = log_dir
        self.C = float(max_grad_norm)
        self.quantile = float(quantile)

    def on_train_start(self, trainer, *args, **kwargs):
        super().on_train_start(trainer, *args, **kwargs)
        self._rows: List[List[float]] = []
        self._current_norms: List[torch.Tensor] = []

    def on_train_batch_start(self, trainer, *args, **kwargs):
        self._current_norms.clear()

    def on_train_physical_batch_end(self, trainer, *args, **kwargs):
        with torch.no_grad():
            sum_squares = None
            for p in trainer.optimizer.params:
                grad_sample_vals = getattr(p, 'grad_sample', None)
                if grad_sample_vals is None or grad_sample_vals.numel() == 0:
                    continue
                flat = grad_sample_vals.reshape(grad_sample_vals.size(0), -1)
                sq = flat.pow(2).sum(dim=1)
                sum_squares = sq if sum_squares is None else sum_squares + sq

            if sum_squares is not None:
                self._current_norms.append(sum_squares.sqrt())

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs):
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)
        if not self._current_norms:
            return

        with torch.no_grad():
            norms = torch.cat(self._current_norms)
            sigma = float(getattr(trainer.optimizer, 'noise_multiplier', 0.0))
            if sigma == 0:
                log.warning('noise_multiplier is zero – SNR will be inf/NaN')

            mean_norm = norms.mean().item()
            q_norm = torch.quantile(norms, self.quantile).item()
            clip_fraction = (norms > self.C).float().mean().item()

            mean_scaled = mean_norm / self.C
            clipped_scaled_mean = (norms / self.C).clamp(max=1).mean().item()

            raw_snr = float('inf') if sigma == 0 else mean_scaled / sigma
            eff_snr = float('inf') if sigma == 0 else clipped_scaled_mean / sigma

            step = self.global_step
            self._rows.append([
                step,
                mean_norm,
                q_norm,
                clip_fraction,
                raw_snr,
                eff_snr,
            ])

        self._current_norms.clear()

    def on_train_end(self, trainer, *args, **kwargs):
        if not self._is_global_zero():
            return

        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, 'snr_log.csv')
        header = [
            'step',
            'mean_norm',
            f'q{int(self.quantile*100)}_norm',
            'clip_fraction',
            'raw_snr',
            'eff_snr',
        ]
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerows([header, *self._rows])

        log.info(f'SNR log written to {path}')
