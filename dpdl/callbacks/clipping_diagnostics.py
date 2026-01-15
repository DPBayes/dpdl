import csv
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

from .base_callback import Callback
from .grad_sample_utils import build_param_name_map, per_sample_grad_norms, select_grad_sample_params

log = logging.getLogger(__name__)


def _as_float_list(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values]


def _default_cs_around(max_grad_norm: float) -> List[float]:
    # Covers a couple of orders of magnitude around the "real" clipping bound.
    base = float(max_grad_norm) if max_grad_norm is not None else 1.0
    factors = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    return [base * f for f in factors]


class _GradSampleStepGatedCallback(Callback):
    def __init__(
        self,
        *,
        log_dir: str | Path,
        log_every_n_steps: int = 0,
        log_steps: Optional[Sequence[int]] = None,
        only_global_zero: bool = True,
        include_param_name_regex: Optional[str] = None,
        exclude_param_name_regex: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.log_dir = Path(log_dir)
        self.log_every_n_steps = int(log_every_n_steps)
        self.log_steps = set(int(s) for s in log_steps) if log_steps is not None else None
        self.only_global_zero = bool(only_global_zero)

        self.include_param_name_regex = include_param_name_regex
        self.exclude_param_name_regex = exclude_param_name_regex

        self._active = False
        self._target_step = None
        self._name_by_id = None

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


class ClipSeverityCallback(_GradSampleStepGatedCallback):
    """
    Logs global clipping severity curves (r(C), m(C), wbar(C)) computed from per-sample norms.
    """

    def __init__(
        self,
        *,
        log_dir: str | Path,
        cs: Optional[Sequence[float]] = None,
        log_every_n_steps: int = 0,
        log_steps: Optional[Sequence[int]] = None,
        only_global_zero: bool = True,
        include_param_name_regex: Optional[str] = None,
        exclude_param_name_regex: Optional[str] = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            log_dir=log_dir,
            log_every_n_steps=log_every_n_steps,
            log_steps=log_steps,
            only_global_zero=only_global_zero,
            include_param_name_regex=include_param_name_regex,
            exclude_param_name_regex=exclude_param_name_regex,
        )

        self.cs = _as_float_list(cs) if cs is not None else None
        self.eps = float(eps)
        self._norm_parts: List[torch.Tensor] = []
        self._rows: List[dict] = []

    def on_train_batch_start(self, trainer, batch_idx, batch, *args, **kwargs) -> None:
        super().on_train_batch_start(trainer, batch_idx, batch, *args, **kwargs)
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
        total_norm = float(norms.sum().item())
        meta = self._metadata(trainer)

        cs = self.cs
        if cs is None:
            cs = _default_cs_around(getattr(trainer, "max_grad_norm", 1.0))

        for C in cs:
            C_t = float(C)
            clipped = norms > C_t
            r = float(clipped.float().mean().item())
            m_num = float((norms - C_t).clamp_min(0.0).sum().item())
            m = m_num / max(total_norm, self.eps)
            wbar = float(torch.minimum(torch.ones_like(norms), (C_t / norms.clamp_min(self.eps))).mean().item())

            self._rows.append(
                {
                    "step": int(self.global_step),
                    **meta,
                    "C": C_t,
                    "B": int(norms.numel()),
                    "r": r,
                    "m": m,
                    "wbar": wbar,
                }
            )

        self._norm_parts.clear()

    def on_train_end(self, trainer, *args, **kwargs) -> None:
        if not self._rows or not self._is_global_zero():
            return

        path = self.log_dir / "clip_severity.csv"
        fieldnames = ["step", "eps", "model", "task", "peft", "noise_multiplier", "max_grad_norm", "C", "B", "r", "m", "wbar"]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

        log.info("Saved clip severity diagnostics to %s", path)


class ClipDirectionDistortionCallback(_GradSampleStepGatedCallback):
    """
    Logs directional distortion metrics between unclipped batch gradient sum G and
    hypothetically clipped sum Gbar(C) for a grid of clipping bounds C.
    """

    def __init__(
        self,
        *,
        log_dir: str | Path,
        cs: Optional[Sequence[float]] = None,
        log_every_n_steps: int = 0,
        log_steps: Optional[Sequence[int]] = None,
        only_global_zero: bool = True,
        include_param_name_regex: Optional[str] = None,
        exclude_param_name_regex: Optional[str] = None,
        store_on_cpu: bool = True,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-12,
        max_total_numel: int = 10_000_000,
    ) -> None:
        super().__init__(
            log_dir=log_dir,
            log_every_n_steps=log_every_n_steps,
            log_steps=log_steps,
            only_global_zero=only_global_zero,
            include_param_name_regex=include_param_name_regex,
            exclude_param_name_regex=exclude_param_name_regex,
        )

        self.cs = _as_float_list(cs) if cs is not None else None
        self.store_on_cpu = bool(store_on_cpu)
        self.dtype = dtype
        self.eps = float(eps)
        self.max_total_numel = int(max_total_numel)

        self._rows: List[dict] = []

        self._params: Optional[List[torch.nn.Parameter]] = None
        self._G_acc: Optional[List[torch.Tensor]] = None
        self._Gbar_acc: Optional[List[torch.Tensor]] = None
        self._B_total = 0

    def on_train_batch_start(self, trainer, batch_idx, batch, *args, **kwargs) -> None:
        super().on_train_batch_start(trainer, batch_idx, batch, *args, **kwargs)
        self._params = None
        self._G_acc = None
        self._Gbar_acc = None
        self._B_total = 0

    def _init_accumulators(self, params: List[torch.nn.Parameter], nC: int) -> None:
        device = torch.device("cpu") if self.store_on_cpu else params[0].device
        self._G_acc = []
        self._Gbar_acc = []
        for p in params:
            d_p = int(p.numel())
            self._G_acc.append(torch.zeros((d_p,), device=device, dtype=self.dtype))
            self._Gbar_acc.append(torch.zeros((nC, d_p), device=device, dtype=self.dtype))

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs) -> None:
        if not self._active:
            return

        with torch.no_grad():
            cs = self.cs
            if cs is None:
                cs = _default_cs_around(getattr(trainer, "max_grad_norm", 1.0))

            sel = select_grad_sample_params(
                trainer.optimizer,
                name_by_param_id=self._name_by_id,
                include_param_name_regex=self.include_param_name_regex,
                exclude_param_name_regex=self.exclude_param_name_regex,
            )
            if sel.batch_size <= 0 or not sel.params:
                return

            total_numel = sum(int(p.numel()) for p in sel.params)
            if total_numel > self.max_total_numel:
                log.warning(
                    "Skipping ClipDirectionDistortionCallback at step %s: selected param numel=%s exceeds max_total_numel=%s",
                    self._target_step,
                    total_numel,
                    self.max_total_numel,
                )
                self._active = False
                return

            if self._params is None:
                self._params = list(sel.params)
                self._init_accumulators(self._params, nC=len(cs))
            else:
                if len(self._params) != len(sel.params) or any(a is not b for a, b in zip(self._params, sel.params)):
                    raise RuntimeError("Parameter set/order with grad_sample changed within a logical batch")

            m = int(sel.batch_size)
            self._B_total += m

            norms = per_sample_grad_norms(self._params, batch_size=m, eps=self.eps, dtype=self.dtype)
            if norms.numel() == 0:
                return

            Cs = torch.tensor(cs, device=norms.device, dtype=norms.dtype).view(1, -1)  # [1, nC]
            scale = torch.minimum(torch.ones_like(norms).view(-1, 1), Cs / norms.view(-1, 1).clamp_min(self.eps))  # [m, nC]

            for idx, p in enumerate(self._params):
                gs = p.grad_sample.view(m, -1).to(dtype=self.dtype)
                G_add = gs.sum(dim=0)
                Gbar_add = torch.matmul(scale.transpose(0, 1), gs)  # [nC, d_p]

                if self.store_on_cpu:
                    self._G_acc[idx].add_(G_add.detach().to("cpu"))
                    self._Gbar_acc[idx].add_(Gbar_add.detach().to("cpu"))
                else:
                    self._G_acc[idx].add_(G_add)
                    self._Gbar_acc[idx].add_(Gbar_add)

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs) -> None:
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        if not self._active or self._params is None or self._G_acc is None or self._Gbar_acc is None:
            return

        meta = self._metadata(trainer)
        cs = self.cs
        if cs is None:
            cs = _default_cs_around(getattr(trainer, "max_grad_norm", 1.0))

        G_norm_sq = 0.0
        for Gp in self._G_acc:
            G_norm_sq += float((Gp * Gp).sum().item())
        G_norm = (G_norm_sq + self.eps) ** 0.5

        nC = len(cs)
        for c_idx, C in enumerate(cs):
            Gbar_norm_sq = 0.0
            G_dot_Gbar = 0.0
            for Gp, Gbarp in zip(self._G_acc, self._Gbar_acc):
                v = Gbarp[c_idx]
                Gbar_norm_sq += float((v * v).sum().item())
                G_dot_Gbar += float((Gp * v).sum().item())

            Gbar_norm = (Gbar_norm_sq + self.eps) ** 0.5
            cos = G_dot_Gbar / (G_norm * Gbar_norm)
            shrink = Gbar_norm / G_norm

            self._rows.append(
                {
                    "step": int(self.global_step),
                    **meta,
                    "C": float(C),
                    "B": int(self._B_total),
                    "G_norm": float(G_norm),
                    "Gbar_norm": float(Gbar_norm),
                    "cos": float(cos),
                    "shrink": float(shrink),
                }
            )

        self._params = None
        self._G_acc = None
        self._Gbar_acc = None
        self._B_total = 0

    def on_train_end(self, trainer, *args, **kwargs) -> None:
        if not self._rows or not self._is_global_zero():
            return

        path = self.log_dir / "clip_direction_distortion.csv"
        fieldnames = [
            "step",
            "eps",
            "model",
            "task",
            "peft",
            "noise_multiplier",
            "max_grad_norm",
            "C",
            "B",
            "G_norm",
            "Gbar_norm",
            "cos",
            "shrink",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

        log.info("Saved clip direction distortion diagnostics to %s", path)


class ClassConditionalClippingCallback(_GradSampleStepGatedCallback):
    """
    Logs class-conditional clipping statistics for a grid of clipping bounds C.
    """

    def __init__(
        self,
        *,
        log_dir: str | Path,
        cs: Optional[Sequence[float]] = None,
        log_every_n_steps: int = 0,
        log_steps: Optional[Sequence[int]] = None,
        only_global_zero: bool = True,
        include_param_name_regex: Optional[str] = None,
        exclude_param_name_regex: Optional[str] = None,
        max_samples: Optional[int] = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            log_dir=log_dir,
            log_every_n_steps=log_every_n_steps,
            log_steps=log_steps,
            only_global_zero=only_global_zero,
            include_param_name_regex=include_param_name_regex,
            exclude_param_name_regex=exclude_param_name_regex,
        )
        self.cs = _as_float_list(cs) if cs is not None else None
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.eps = float(eps)

        self._num_classes = None
        self._norm_parts: List[torch.Tensor] = []
        self._label_parts: List[torch.Tensor] = []
        self._rows: List[dict] = []

    def on_train_start(self, trainer, *args, **kwargs) -> None:
        super().on_train_start(trainer, *args, **kwargs)
        if hasattr(trainer.datamodule, "get_num_classes"):
            self._num_classes = int(trainer.datamodule.get_num_classes())

    def on_train_batch_start(self, trainer, batch_idx, batch, *args, **kwargs) -> None:
        super().on_train_batch_start(trainer, batch_idx, batch, *args, **kwargs)
        self._norm_parts.clear()
        self._label_parts.clear()

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs) -> None:
        if not self._active:
            return

        _, labels = batch if isinstance(batch, (tuple, list)) and len(batch) >= 2 else (None, None)
        if labels is None:
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
            self._label_parts.append(labels.detach().to("cpu"))

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs) -> None:
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        if not self._active or not self._norm_parts or not self._label_parts:
            return

        norms = torch.cat(self._norm_parts, dim=0)
        labels = torch.cat(self._label_parts, dim=0)

        if norms.numel() != labels.numel():
            log.warning(
                "ClassConditionalClippingCallback: norms/labels size mismatch at step %s (%s vs %s); skipping",
                self.global_step,
                norms.numel(),
                labels.numel(),
            )
            return

        if self.max_samples is not None and norms.numel() > self.max_samples:
            idx = torch.randperm(norms.numel())[: self.max_samples]
            norms = norms[idx]
            labels = labels[idx]

        if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            return

        meta = self._metadata(trainer)
        cs = self.cs
        if cs is None:
            cs = _default_cs_around(getattr(trainer, "max_grad_norm", 1.0))

        if self._num_classes is None:
            # Best-effort fallback for classification tasks.
            self._num_classes = int(labels.max().item()) + 1 if labels.numel() else 0

        for class_id in range(self._num_classes):
            mask = labels == class_id
            n_class = int(mask.sum().item())
            if n_class == 0:
                continue

            class_norms = norms[mask]
            p50 = float(torch.quantile(class_norms, 0.50).item())
            p90 = float(torch.quantile(class_norms, 0.90).item())

            for C in cs:
                C_t = float(C)
                clipped = class_norms > C_t
                r = float(clipped.float().mean().item())
                w = torch.minimum(torch.ones_like(class_norms), (C_t / class_norms.clamp_min(self.eps)))
                wbar = float(w.mean().item())

                self._rows.append(
                    {
                        "step": int(self.global_step),
                        **meta,
                        "C": C_t,
                        "class_id": int(class_id),
                        "n_class": n_class,
                        "r_class": r,
                        "w_class": wbar,
                        "p50_norm": p50,
                        "p90_norm": p90,
                    }
                )

        self._norm_parts.clear()
        self._label_parts.clear()

    def on_train_end(self, trainer, *args, **kwargs) -> None:
        if not self._rows or not self._is_global_zero():
            return

        path = self.log_dir / "class_conditional_clipping.csv"
        fieldnames = [
            "step",
            "eps",
            "model",
            "task",
            "peft",
            "noise_multiplier",
            "max_grad_norm",
            "C",
            "class_id",
            "n_class",
            "r_class",
            "w_class",
            "p50_norm",
            "p90_norm",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

        log.info("Saved class-conditional clipping diagnostics to %s", path)

