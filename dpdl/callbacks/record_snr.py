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

    def on_train_start(self, *_, **__):
        self._rows = []
        self._current_norms = []

    def on_train_batch_start(self, *_, **__):
        self._current_norms.clear()

    def on_train_physical_batch_end(self, trainer, *_, **__):
        """
        Accumulate per-sample gradient norms for *this* physical batcj only
        (the last entry in each param's grad_sample list).
        """
        with torch.no_grad():
            sq = None
            for p in trainer.optimizer.params:
                gs = getattr(p, 'grad_sample', None)

                if gs is None:
                    continue

                gs = gs[-1] if isinstance(gs, list) else gs  # last micro-batch

                if gs.numel() == 0:
                    continue

                g_flat = gs.reshape(gs.size(0), -1)
                s = g_flat.pow(2).sum(dim=1)  # ||g_i||^2
                sq = s if sq is None else sq + s

            if sq is not None:
                self._current_norms.append(sq.sqrt())  # shape (physical_bsz,)

    def on_train_batch_end(self, trainer, *_, **__):
        if not self._current_norms:  # empty batch (e.g. skipped)
            return

        with torch.no_grad():
            norms = torch.cat(self._current_norms)  # (logical_bsz,)
            sigma = float(getattr(trainer.optimizer, 'noise_multiplier', 0.0))

            if sigma == 0:
                log.warning('noise_multiplier is zero – SNR will be inf/NaN')

            mean_norm = norms.mean().item()
            q_norm = torch.quantile(norms, self.quantile).item()
            clip_fraction = (norms > self.C).float().mean().item()

            # scale by C once ⇒ same formula for standard & normalised modes
            norm_scaled = norms / self.C
            mean_scaled = mean_norm / self.C
            clipped_scaled = norm_scaled.clamp(max=1).mean().item()

            raw_snr = float('inf') if sigma == 0 else mean_scaled / sigma
            eff_snr = float('inf') if sigma == 0 else clipped_scaled / sigma

            step = int(getattr(trainer, 'global_step', len(self._rows)))
            self._rows.append(
                [
                    step,
                    mean_norm,
                    q_norm,
                    clip_fraction,
                    raw_snr,
                    eff_snr,
                ]
            )

        self._current_norms.clear()

    def on_train_end(self, *_, **__):
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

