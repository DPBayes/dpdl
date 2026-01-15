import csv
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from .base_callback import Callback
from .grad_sample_utils import build_param_name_map, per_sample_grad_norms, select_grad_sample_params

log = logging.getLogger(__name__)


class GradNormTraceCallback(Callback):
    """
    Tracks per-sample gradient norm quantiles over training (q50/q90/q99 and mean log-norm).
    """

    def __init__(
        self,
        *,
        log_dir: str | Path,
        log_every_n_steps: int = 0,
        log_steps: Optional[Sequence[int]] = None,
        only_global_zero: bool = True,
        include_param_name_regex: Optional[str] = None,
        exclude_param_name_regex: Optional[str] = None,
        eps: float = 1e-12,
        log_eps: float = 1e-12,
    ) -> None:
        super().__init__()

        self.log_dir = Path(log_dir)
        self.log_every_n_steps = int(log_every_n_steps)
        self.log_steps = set(int(s) for s in log_steps) if log_steps is not None else None
        self.only_global_zero = bool(only_global_zero)

        self.include_param_name_regex = include_param_name_regex
        self.exclude_param_name_regex = exclude_param_name_regex

        self.eps = float(eps)
        self.log_eps = float(log_eps)

        self._name_by_id = None
        self._active = False
        self._target_step = None
        self._norm_parts: List[torch.Tensor] = []
        self._rows: List[dict] = []

    def _should_log_step(self, step: int) -> bool:
        if self.log_steps is not None:
            return step in self.log_steps
        if self.log_every_n_steps and self.log_every_n_steps > 0:
            return (step % self.log_every_n_steps) == 0
        return True

    def _metadata(self, trainer) -> dict:
        model = trainer._unwrap_model()
        return {
            "eps": getattr(trainer, "target_epsilon", None),
            "noise_multiplier": getattr(trainer, "noise_multiplier", None),
            "max_grad_norm": getattr(trainer, "max_grad_norm", None),
            "model": type(getattr(model, "model", model)).__name__,
            "task": getattr(trainer, "task", None),
            "peft": getattr(trainer, "peft", None),
        }

    def on_train_start(self, trainer, *args, **kwargs) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._name_by_id = build_param_name_map(trainer._unwrap_model())
        super().on_train_start(trainer, *args, **kwargs)

    def on_train_batch_start(self, trainer, batch_idx, batch, *args, **kwargs) -> None:
        next_step = int(self.global_step) + 1
        self._target_step = next_step
        self._active = self._should_log_step(next_step)
        if self.only_global_zero and not self._is_global_zero():
            self._active = False
        self._norm_parts.clear()

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs) -> None:
        if not self._active:
            return

        with torch.no_grad():
            sel = select_grad_sample_params(
                trainer.optimizer,
                name_by_param_id=self._name_by_id,
                include_param_name_regex=self.include_param_name_regex,
                exclude_param_name_regex=self.exclude_param_name_regex,
            )
            if sel.batch_size <= 0 or not sel.params:
                return

            norms = per_sample_grad_norms(sel.params, batch_size=sel.batch_size, eps=self.eps)
            if norms.numel() == 0:
                return

            self._norm_parts.append(norms.detach().to("cpu"))

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs) -> None:
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        if not self._active or not self._norm_parts:
            return

        norms = torch.cat(self._norm_parts, dim=0)
        meta = self._metadata(trainer)

        q50 = float(torch.quantile(norms, 0.50).item())
        q90 = float(torch.quantile(norms, 0.90).item())
        q99 = float(torch.quantile(norms, 0.99).item())
        meanlog = float(torch.log(norms + self.log_eps).mean().item())

        self._rows.append(
            {
                "step": int(self.global_step),
                **meta,
                "B": int(norms.numel()),
                "q50": q50,
                "q90": q90,
                "q99": q99,
                "meanlog": meanlog,
            }
        )

        self._norm_parts.clear()

    def on_train_end(self, trainer, *args, **kwargs) -> None:
        if not self._rows or not self._is_global_zero():
            return

        path = self.log_dir / "grad_norm_trace.csv"
        fieldnames = ["step", "eps", "model", "task", "peft", "noise_multiplier", "max_grad_norm", "B", "q50", "q90", "q99", "meanlog"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

        log.info("Saved gradient norm trace to %s", path)

