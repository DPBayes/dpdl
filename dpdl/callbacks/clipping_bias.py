import csv
import logging
from pathlib import Path
from typing import List, Optional

import torch

from .base_callback import Callback

log = logging.getLogger(__name__)


class ClipMSEDecompositionCallback(Callback):
    def __init__(
        self,
        log_dir: str | Path,
        max_grad_norm: float,
        *,
        normalize_clipping: bool = False,
        store_on_cpu: bool = True,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        self.log_dir = Path(log_dir)
        self.C = float(max_grad_norm)

        self.normalize_clipping = bool(normalize_clipping)
        self.store_on_cpu = bool(store_on_cpu)
        self.dtype = dtype
        self.eps = float(eps)

        self._rows = []
        self._parameter_count = None

        # Logical-batch accumulators (across physical batches)
        self._m_total = 0
        self._params = None

        # Standard-form active-set sums: G = sum g_i, N = sum g_i / ||g_i||.
        self._G_acc = None
        self._N_acc = None

        # Normalized-form sums to reconstruct standard bias: C * Gbar - Graw.
        self._Gbar_acc = None
        self._Graw_acc = None

        # Sum_i ||gbar_i||^2 for the mini-batch sampling noise term.
        self._clip_norm_sq_sum = 0.0

    def on_train_start(self, trainer, *args, **kwargs) -> None:
        self._rows.clear()
        super().on_train_start(trainer, *args, **kwargs)

    def on_train_batch_start(self, trainer, *args, **kwargs) -> None:
        self._m_total = 0
        self._params = None
        self._G_acc = None
        self._N_acc = None
        self._Gbar_acc = None
        self._Graw_acc = None
        self._clip_norm_sq_sum = 0.0

    def _get_d(self, trainer) -> int:
        if self._parameter_count is None:
            self._parameter_count = sum(
                p.numel()
                for g in trainer.optimizer.param_groups
                for p in g['params']
                if p.requires_grad
            )

        return int(self._parameter_count)

    def _get_sigma(self, trainer) -> float:
        return float(getattr(trainer.optimizer, 'noise_multiplier', 0.0))

    def _get_q(self, trainer) -> float:
        if trainer.datamodule.sample_rate:
            return float(trainer.datamodule.sample_rate)

        N = trainer.datamodule.get_dataset_size()
        B = trainer.datamodule.batch_size
        q = B/N

        return q

    def _get_N(self, trainer) -> int:
        return int(trainer.datamodule.get_dataset_size('train_dataset'))

    def _init_accumulators(self, params: List[torch.nn.Parameter]) -> None:
        device = 'cpu' if self.store_on_cpu else params[0].device

        self._G_acc = []
        self._N_acc = []

        for p in params:
            zg = torch.zeros(p.numel(), device=device, dtype=self.dtype)
            zn = torch.zeros(p.numel(), device=device, dtype=self.dtype)

            self._G_acc.append(zg)
            self._N_acc.append(zn)

        if self.normalize_clipping:
            self._Gbar_acc = []
            self._Graw_acc = []

            for p in params:
                z_bar = torch.zeros(p.numel(), device=device, dtype=self.dtype)
                z_raw = torch.zeros(p.numel(), device=device, dtype=self.dtype)

                self._Gbar_acc.append(z_bar)
                self._Graw_acc.append(z_raw)

    def on_train_physical_batch_end(self, trainer, *args, **kwargs) -> None:
        with torch.no_grad():
            # Collect params that have grad_sample this step (order-stable)
            params = []
            batch_size = None

            for group in trainer.optimizer.param_groups:
                for p in group['params']:
                    gs = getattr(p, 'grad_sample', None)
                    if gs is None or gs.numel() == 0:
                        continue

                    m_i = gs.size(0)

                    if batch_size is None:
                        batch_size = m_i
                    elif batch_size != m_i:
                        raise ValueError('Inconsistent batch sizes across params')

                    params.append(p)

            if batch_size is None or not params:
                return

            if self._params is None:
                self._params = params
                self._init_accumulators(params)
            else:
                # Assumes same param order each time; if not, fail loudly.
                if len(self._params) != len(params) or any(a is not b for a, b in zip(self._params, params)):
                    raise RuntimeError('Parameter set/order with grad_sample changed across physical batches')

            m = int(batch_size)
            self._m_total += m

            # Compute per-sample norms without concatenating: ||g_i||^2 = sum_p ||g_{i,p}||^2
            norms_sq = None
            for p in self._params:
                gs = p.grad_sample.view(m, -1)

                part = (gs * gs).sum(dim=1)
                norms_sq = part if norms_sq is None else (norms_sq + part)

            norms = norms_sq.clamp_min(self.eps).sqrt()

            # Accumulate totals needed for this variant
            if not self.normalize_clipping:
                mask = norms > self.C

                # Sum of ||gbar_i||^2 in standard form.
                clip_norm_sq_sum_add = norms_sq[~mask].sum()
                if mask.any():
                    clip_norm_sq_sum_add = clip_norm_sq_sum_add + (mask.sum() * (self.C * self.C))

                self._clip_norm_sq_sum += float(clip_norm_sq_sum_add.item())

                if mask.any():
                    inv_norm = 1.0 / norms.clamp_min(self.eps)
                    inv_norm_masked = inv_norm[mask]

                    for idx, p in enumerate(self._params):
                        gs = p.grad_sample.view(m, -1)

                        g_clip = gs[mask]
                        G_add = g_clip.sum(dim=0)

                        # sum g_i / ||g_i||
                        N_add = (g_clip * inv_norm_masked.unsqueeze(1)).sum(dim=0)

                        if self.store_on_cpu:
                            self._G_acc[idx].add_(G_add.detach().to('cpu', dtype=self.dtype))
                            self._N_acc[idx].add_(N_add.detach().to('cpu', dtype=self.dtype))
                        else:
                            self._G_acc[idx].add_(G_add.to(dtype=self.dtype))
                            self._N_acc[idx].add_(N_add.to(dtype=self.dtype))

            else:
                invC = 1.0 / self.C
                inv_norm = 1.0 / norms.clamp_min(self.eps)

                # Normalized clipping scale: min(1/C, 1/||g||).
                scale = torch.minimum(torch.full_like(norms, invC), inv_norm)

                # Store ||gbar_i||^2 in normalized units (scaled to standard at logging).
                clip_norm_sq_sum_add = (scale * scale * norms_sq).sum()
                self._clip_norm_sq_sum += float(clip_norm_sq_sum_add.item())

                mask = norms > self.C

                for idx, p in enumerate(self._params):
                    gs = p.grad_sample.view(m, -1)
                    Gbar_add = (gs * scale.unsqueeze(1)).sum(dim=0)
                    Graw_add = gs.sum(dim=0)

                    if self.store_on_cpu:
                        self._Gbar_acc[idx].add_(Gbar_add.detach().to('cpu', dtype=self.dtype))
                        self._Graw_acc[idx].add_(Graw_add.detach().to('cpu', dtype=self.dtype))
                    else:
                        self._Gbar_acc[idx].add_(Gbar_add.to(dtype=self.dtype))
                        self._Graw_acc[idx].add_(Graw_add.to(dtype=self.dtype))

                    if mask.any():
                        g_clip = gs[mask]
                        inv_norm_masked = inv_norm[mask]
                        G_add = g_clip.sum(dim=0)
                        N_add = (g_clip * inv_norm_masked.unsqueeze(1)).sum(dim=0)

                        if self.store_on_cpu:
                            self._G_acc[idx].add_(G_add.detach().to('cpu', dtype=self.dtype))
                            self._N_acc[idx].add_(N_add.detach().to('cpu', dtype=self.dtype))
                        else:
                            self._G_acc[idx].add_(G_add.to(dtype=self.dtype))
                            self._N_acc[idx].add_(N_add.to(dtype=self.dtype))

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs) -> None:
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        if self._m_total == 0 or self._params is None:
            return

        d = self._get_d(trainer)
        sigma = self._get_sigma(trainer)

        # Use expected batch size q*N like Opacus.
        q = self._get_q(trainer)
        N = self._get_N(trainer)

        # Also colelct realized batch size for diagnostics.
        B = int(self._m_total)

        N_sq = 0.0
        G_sq = 0.0
        G_dot_N = 0.0
        for Gp, Np in zip(self._G_acc, self._N_acc):
            G_sq += float((Gp * Gp).sum().item())
            N_sq += float((Np * Np).sum().item())
            G_dot_N += float((Gp * Np).sum().item())

        if not self.normalize_clipping:
            clip_err_sq_sum = G_sq - 2.0 * self.C * G_dot_N + (self.C * self.C) * N_sq
        else:
            clip_err_sq_sum = 0.0
            for Gbarp, Grawp in zip(self._Gbar_acc, self._Graw_acc):
                bias_p = self.C * Gbarp - Grawp
                clip_err_sq_sum += float((bias_p * bias_p).sum().item())

        NdotG = G_dot_N
        NdotN = N_sq
        noise_term_sum = (self.C * self.C) * (sigma * sigma) * float(d)

        denom = float(q * N)
        denom *= denom
        clip_err_sq_mean = float(clip_err_sq_sum) / denom
        noise_term_mean = float(noise_term_sum) / denom
        clip_norm_sq_sum = float(self._clip_norm_sq_sum)

        if self.normalize_clipping:
            # Map normalized clipping stats to standard parameterization.
            clip_norm_sq_sum *= self.C**2

        minibatch_noise_sum = ((1.0 - q) / q) * clip_norm_sq_sum
        minibatch_noise_mean = float(minibatch_noise_sum) / denom
        mse_mean = clip_err_sq_mean + noise_term_mean + minibatch_noise_mean

        step = self.global_step
        self._rows.append([
            step,
            B,
            d,
            float(sigma),
            float(self.C),
            1 if self.normalize_clipping else 0,
            float(q),
            float(NdotG),
            float(NdotN),
            float(noise_term_mean),
            float(minibatch_noise_mean),
            float(clip_err_sq_mean),
            float(mse_mean),
        ])

    def on_train_end(self, trainer, *args, **kwargs) -> None:
        if not self._rows or not self._is_global_zero():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.log_dir / 'clipping_mse_decomposition.csv'

        header = [
            'step',
            'B',
            'd',
            'sigma',
            'C_train',
            'normalize_clipping',
            'q',
            'NdotG',
            'NdotN',
            'noise_term_mean',
            'minibatch_noise_mean',
            'clip_err_sq_mean',
            'mse_mean',
        ]

        with out_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self._rows)

        log.info(f'Clipping MSE decomposition written to {out_path}')
