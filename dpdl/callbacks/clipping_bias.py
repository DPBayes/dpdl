"""
Callback: record clipping bias and MSE during DP training.

Logs one CSV row per logical batch with columns:
  - step      : global step index (from base Callback)
  - bias_sq   : squared clipping bias ||g_unclipped – g_clipped||^2
  - noise_var : DP-noise variance term sigma^2 * C^2 * d / m^2
  - mse_est   : bias_sq + noise_var

Notation:
  C   : per-sample clipping bound (max_grad_norm)
  sigma : noise multiplier used by the DP optimizer
  d   : total number of trainable parameters
  m   : number of samples in this logical batch
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional

import torch

from .base_callback import Callback

log = logging.getLogger(__name__)


class ClippingBiasCallback(Callback):
    """
    Collect clipping bias and estimated MSE during DP training.

    Parameters
    ----------
    log_dir : str or Path
        Directory where 'clip_stats.csv' will be written.
    max_grad_norm : float
        Clipping bound C used by DP-SGD or DP-Adam.

    Attributes
    ----------
    _rows : List[List[float]]
        Buffer of [step, bias_sq, noise_var, mse_est] entries.
    """

    def __init__(self, log_dir: str | Path, max_grad_norm: float) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)
        self.C = float(max_grad_norm)

        # Buffers for one logical batch (which may span multiple physical batches)
        self._unclipped_sum: Optional[List[torch.Tensor]] = None
        self._clipped_sum: Optional[List[torch.Tensor]] = None
        self._sample_count: int = 0

        # Total parameter count (d), computed once
        self._parameter_count: Optional[int] = None

        # Collected rows across training
        self._rows: List[List[float]] = []

    def on_train_start(self, trainer, *args, **kwargs) -> None:
        # Clear any leftover data
        self._rows.clear()
        super().on_train_start(trainer, *args, **kwargs)

    def on_train_batch_start(self, trainer, *args, **kwargs) -> None:
        # Begin a new logical batch
        self._unclipped_sum = []
        self._clipped_sum = []
        self._sample_count = 0

    def on_train_physical_batch_end(self, trainer, *args, **kwargs) -> None:
        """
        Accumulate per-sample unclipped vs clipped gradients.
        Assumes each param.grad_sample is a tensor of shape [m_i, ...].
        """
        with torch.no_grad():
            batch_size = None
            unclipped_parts: List[torch.Tensor] = []
            clipped_parts:   List[torch.Tensor] = []

            for group in trainer.optimizer.param_groups:
                for param in group['params']:
                    grad_sample = getattr(param, 'grad_sample', None)
                    if grad_sample is None or grad_sample.numel() == 0:
                        continue

                    # Ensure consistent m across parameters
                    m_i = grad_sample.size(0)
                    if batch_size is None:
                        batch_size = m_i
                    elif batch_size != m_i:
                        raise ValueError("Inconsistent batch sizes across params")

                    # Flatten per-sample gradients: [m, d_i]
                    flat = grad_sample.view(m_i, -1)

                    # Sum unclipped
                    unclipped_parts.append(flat.sum(dim=0))

                    # Compute per-sample clip factor and sum
                    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    scale = (self.C / norms).clamp(max=1.0)
                    clipped = flat * scale
                    clipped_parts.append(clipped.sum(dim=0))

            if batch_size is not None:
                # Concatenate sums from every param into one big vector of dim d
                self._unclipped_sum.append(torch.cat(unclipped_parts))
                self._clipped_sum.append(torch.cat(clipped_parts))
                self._sample_count += batch_size

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs) -> None:
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        # Skip if no data accumulated
        if not self._unclipped_sum or not self._clipped_sum or self._sample_count == 0:
            return

        # Sum across all physical batches in this logical batch
        total_unclipped = torch.stack(self._unclipped_sum).sum(dim=0)
        total_clipped   = torch.stack(self._clipped_sum).sum(dim=0)
        m = self._sample_count  # number of samples this batch

        # Compute means and bias^2
        mean_u = total_unclipped / m
        mean_c = total_clipped   / m
        bias_sq = (mean_u - mean_c).pow(2).sum().item()

        # Compute d = total trainable parameters count
        if self._parameter_count is None:
            self._parameter_count = sum(
                p.numel()
                for g in trainer.optimizer.param_groups
                for p in g['params']
                if p.requires_grad
            )
        d = self._parameter_count

        # Noise multiplier sigma from optimizer
        sigma = float(getattr(trainer.optimizer, 'noise_multiplier', 0.0))

        # variance term = sigma^2 * C^2 * d / m^2
        noise_var = sigma**2 * self.C**2 * d / (m**2)

        # Total MSE estimate
        mse_est = bias_sq + noise_var

        # Record row
        step = self.global_step
        self._rows.append([step, bias_sq, noise_var, mse_est])

    def on_train_end(self, trainer, *args, **kwargs) -> None:
        # Only write on rank 0 (or non-distributed)
        if not self._rows or not self._is_global_zero():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        out_path = self.log_dir / 'clip_stats.csv'
        header = ['step', 'bias_sq', 'noise_var', 'mse_est']

        with out_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self._rows)

        log.info("Clip statistics written to %s", out_path)
