import csv
import logging
from pathlib import Path
from typing import List

import torch
from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordSNRCallback(Callback):
    """
    Callback: record gradient-norm stats and SNR per logical batch.

    Columns logged:
      step           - global step index from base Callback
      mean_norm      - E[||g||] over all samples in the batch (pre-clip)
      q{pct}_norm    - empirical quantile of ||g|| at the requested pct
      clip_fraction  - fraction of samples with ||g|| > C
      raw_snr        - (mean_norm / C) / sigma
      eff_snr        - E[min(||g||/C, 1)] / sigma

    Notation:
      C       : max_grad_norm (clipping bound)
      sigma   : noise multiplier from the optimizer
      m       : number of samples in this logical batch
      d       : total number of trainable parameters
    """

    def __init__(self, log_dir: str | Path, max_grad_norm: float, quantile: float = 0.90):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.C = float(max_grad_norm)
        self.quantile = float(quantile)

        # Will hold rows [step, mean_norm, q_norm, clip_fraction, raw_snr, eff_snr]
        self._rows: List[List[float]] = []

        # Will accumulate per-physical-batch norm tensors
        self._current_norms: List[torch.Tensor] = []

    def on_train_start(self, trainer, *args, **kwargs):
        """
        Reset storage at the start of training.
        """
        super().on_train_start(trainer, *args, **kwargs)
        self._rows.clear()
        self._current_norms.clear()

    def on_train_batch_start(self, trainer, *args, **kwargs):
        """
        Begin a new logical batch: clear per-batch norms accumulator.
        """
        self._current_norms.clear()

    def on_train_physical_batch_end(self, trainer, *args, **kwargs):
        """
        After each micro-batch, collect its per-sample gradient norms.
        Assumes each p.grad_sample is shape [m_i, ...].
        """
        with torch.no_grad():
            sum_squares: torch.Tensor | None = None

            # Loop over all parameters in all param_groups
            for group in trainer.optimizer.param_groups:
                for p in group['params']:
                    grad_sample = getattr(p, 'grad_sample', None)
                    if grad_sample is None or grad_sample.numel() == 0:
                        continue

                    # Flatten per-sample
                    flat = grad_sample.view(grad_sample.size(0), -1)

                    # Sum of squares per sample
                    sq = flat.pow(2).sum(dim=1)

                    sum_squares = sq if sum_squares is None else sum_squares + sq

            if sum_squares is not None:
                # Store the Euclidean norms for this micro-batch
                self._current_norms.append(sum_squares.sqrt())

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs):
        """
        At end of logical batch, compute and log:
          - mean_norm, q_norm, clip_fraction
          - raw_snr and eff_snr using optimizer.noise_multiplier
        """
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        if not self._current_norms:
            return

        with torch.no_grad():
            # Concatenate all micro-batch norms into one vector of length m
            norms = torch.cat(self._current_norms)
            m = norms.numel()
            sigma = float(getattr(trainer.optimizer, 'noise_multiplier', 0.0))
            if sigma == 0.0:
                log.warning("noise_multiplier is zero; SNR may be infinite or NaN")

            # Pre-clip statistics
            mean_norm = norms.mean().item()
            q_norm = torch.quantile(norms, self.quantile).item()
            clip_fraction = (norms > self.C).float().mean().item()

            # Normalize by C, then by sigma
            raw_snr = (mean_norm / self.C) / sigma if sigma > 0 else float('inf')
            eff_snr = (norms.div(self.C).clamp(max=1.0).mean().item() / sigma) if sigma > 0 else float('inf')

            # Record row
            step = self.global_step
            self._rows.append([
                step,
                mean_norm,
                q_norm,
                clip_fraction,
                raw_snr,
                eff_snr,
            ])

        # Ready for next logical batch
        self._current_norms.clear()

    def on_train_end(self, trainer, *args, **kwargs):
        """
        Write the CSV on rank 0 (or if non-distributed).
        """
        if not self._rows or not self._is_global_zero():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.log_dir / 'snr_log.csv'
        header = [
            'step',
            'mean_norm',
            f'q{int(self.quantile * 100)}_norm',
            'clip_fraction',
            'raw_snr',
            'eff_snr',
        ]

        with out_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self._rows)

        log.info("SNR log written to %s", out_path)
