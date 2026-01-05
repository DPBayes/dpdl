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
        debug_checks: bool = False,
    ) -> None:
        super().__init__()

        self.log_dir = Path(log_dir)
        self.C = float(max_grad_norm)

        self.normalize_clipping = bool(normalize_clipping)
        self.store_on_cpu = bool(store_on_cpu)
        self.dtype = dtype
        self.eps = float(eps)
        self.debug_checks = bool(debug_checks)

        self._rows = []
        self._parameter_count = None

        # Logical-batch accumulators (across physical batches)
        self._m_total = 0
        self._params = None
        self._G_acc = None
        self._N_acc = None
        self._S_acc = None

    def on_train_start(self, trainer, *args, **kwargs) -> None:
        self._rows.clear()
        super().on_train_start(trainer, *args, **kwargs)

    def on_train_batch_start(self, trainer, *args, **kwargs) -> None:
        self._m_total = 0
        self._params = None
        self._G_acc = None
        self._N_acc = None
        self._S_acc = None

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

    def _init_accumulators(self, params: List[torch.nn.Parameter]) -> None:
        device = 'cpu' if self.store_on_cpu else params[0].device

        if self.normalize_clipping:
            self._S_acc = []
            for p in params:
                z = torch.zeros(p.numel(), device=device, dtype=self.dtype)
                self._S_acc.append(z)
        else:
            self._G_acc = []
            self._N_acc = []
            for p in params:
                zg = torch.zeros(p.numel(), device=device, dtype=self.dtype)
                zn = torch.zeros(p.numel(), device=device, dtype=self.dtype)
                self._G_acc.append(zg)
                self._N_acc.append(zn)

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

            # 1) Compute per-sample norms without concatenating: ||g_i||^2 = sum_p ||g_{i,p}||^2
            norms_sq = None
            for p in self._params:
                gs = p.grad_sample.view(m, -1)
                part = (gs * gs).sum(dim=1)
                norms_sq = part if norms_sq is None else (norms_sq + part)

            norms = norms_sq.clamp_min(self.eps).sqrt()

            # 2) Accumulate totals needed for this variant
            if not self.normalize_clipping:
                mask = norms > self.C

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
                scale = torch.minimum(torch.full_like(norms, invC), inv_norm)
                w = scale - 1.0  # weight for (gbar - g) = w * g

                for idx, p in enumerate(self._params):
                    gs = p.grad_sample.view(m, -1)
                    S_add = (gs * w.unsqueeze(1)).sum(dim=0)

                    if self.store_on_cpu:
                        self._S_acc[idx].add_(S_add.detach().to('cpu', dtype=self.dtype))
                    else:
                        self._S_acc[idx].add_(S_add.to(dtype=self.dtype))

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs) -> None:
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        if self._m_total == 0 or self._params is None:
            return

        m = int(self._m_total)
        d = self._get_d(trainer)
        sigma = self._get_sigma(trainer)

        if not self.normalize_clipping:
            G_sq = 0.0
            N_sq = 0.0
            G_dot_N = 0.0

            for Gp, Np in zip(self._G_acc, self._N_acc):
                G_sq += float((Gp * Gp).sum().item())
                N_sq += float((Np * Np).sum().item())
                G_dot_N += float((Gp * Np).sum().item())

            clip_err_sq_sum = G_sq - 2.0 * self.C * G_dot_N + (self.C * self.C) * N_sq
            NdotG = G_dot_N
            NdotN = N_sq

            noise_term_sum = (self.C * self.C) * (sigma * sigma) * float(d)

        else:
            S_sq = 0.0
            for Sp in self._S_acc:
                S_sq += float((Sp * Sp).sum().item())

            clip_err_sq_sum = S_sq
            NdotG = float('nan')
            NdotN = float('nan')

            noise_term_sum = (sigma * sigma) * float(d)

        denom = float(m * m)
        clip_err_sq_mean = float(clip_err_sq_sum) / denom
        noise_term_mean = float(noise_term_sum) / denom
        mse_mean = clip_err_sq_mean + noise_term_mean

        step = self.global_step
        self._rows.append([
            step,
            m,
            d,
            float(sigma),
            float(self.C),
            1 if self.normalize_clipping else 0,
            float(NdotG),
            float(NdotN),
            float(noise_term_mean),
            float(clip_err_sq_mean),
            float(mse_mean),
        ])

    def on_train_end(self, trainer, *args, **kwargs) -> None:
        if not self._rows or not self._is_global_zero():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.log_dir / 'clip_mse_decomp.csv'

        header = [
            'step',
            'm',
            'd',
            'sigma',
            'C_train',
            'normalize_clipping',
            'NdotG',            # standard clipping only; NaN for normalized
            'NdotN',            # standard clipping only; NaN for normalized
            'noise_term_mean',
            'clip_err_sq_mean',
            'mse_mean',
        ]

        with out_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self._rows)

        log.info('Clip MSE decomposition written to %s', out_path)
