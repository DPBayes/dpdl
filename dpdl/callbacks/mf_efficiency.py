import json
import math
from pathlib import Path
from typing import Any

import torch

from ..utils import safe_open
from .base_callback import Callback


class MFEfficiencyMetricsCallback(Callback):
    """
    Compute paper-aligned MF efficiency metrics at trial end.

    Formula:
      mse_prefix = ||A C^{-1}||_F^2 * sigma(C)^2
      rmse_prefix = sqrt(mse_prefix)
    where A is the lower-triangular all-ones prefix matrix.
    """

    FORMULA_ID = "prefix_mse_acinv_fro_sq_times_sigma_sq_v1"

    def __init__(self, log_dir: str | Path):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.latest_summary: dict[str, Any] | None = None

    @staticmethod
    def _resolve_horizon(trainer) -> int:
        total_steps = getattr(trainer, "total_steps", None)
        if total_steps is not None:
            return int(total_steps)

        epochs = getattr(trainer, "epochs", None)
        if epochs is not None:
            train_loader = trainer.datamodule.get_dataloader("train")
            return int(epochs) * int(len(train_loader))

        raise ValueError("cannot resolve horizon: expected trainer.total_steps or trainer.epochs")

    @staticmethod
    def _lower_toeplitz_from_coeffs(coeffs: list[float], horizon: int) -> torch.Tensor:
        c = torch.zeros((horizon, horizon), dtype=torch.float64)
        for i in range(horizon):
            max_lag = min(i, len(coeffs) - 1)
            for lag in range(max_lag + 1):
                c[i, i - lag] = float(coeffs[lag])

        return c

    @staticmethod
    def _prefix_matrix(horizon: int) -> torch.Tensor:
        return torch.tril(torch.ones((horizon, horizon), dtype=torch.float64))

    @staticmethod
    def _coerce_valid_sigma(value: Any) -> float | None:
        try:
            sigma = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(sigma) or sigma < 0.0:
            return None

        return sigma

    @staticmethod
    def _sigma_from_trainer(
        trainer, mechanism: str, mechanism_state: dict[str, Any]
    ) -> tuple[float | None, str | None]:
        if mechanism in ("bandmf", "bsr", "bnb"):
            if "z_std" in mechanism_state:
                sigma = MFEfficiencyMetricsCallback._coerce_valid_sigma(
                    mechanism_state.get("z_std")
                )
                if sigma is None:
                    return None, "mechanism_state.z_std_invalid"

                return sigma, "mechanism_state.z_std"

        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "noise_multiplier"):
            sigma = MFEfficiencyMetricsCallback._coerce_valid_sigma(
                getattr(optimizer, "noise_multiplier")
            )
            if sigma is not None:
                return sigma, "optimizer.noise_multiplier"

        if hasattr(trainer, "noise_multiplier") and getattr(trainer, "noise_multiplier") is not None:
            sigma = MFEfficiencyMetricsCallback._coerce_valid_sigma(
                getattr(trainer, "noise_multiplier")
            )
            if sigma is not None:
                return sigma, "trainer.noise_multiplier"

        pe = getattr(trainer, "privacy_engine", None)
        if pe is not None and hasattr(pe, "noise_multiplier"):
            sigma = MFEfficiencyMetricsCallback._coerce_valid_sigma(
                getattr(pe, "noise_multiplier")
            )
            if sigma is not None:
                return sigma, "privacy_engine.noise_multiplier"

        return None, None

    @staticmethod
    def _mechanism_context(trainer) -> tuple[str, str | None, dict[str, Any]]:
        mechanism = str(getattr(trainer, "noise_mechanism", "gaussian"))
        sampling_mode = getattr(trainer, "sampling_mode", None)
        mechanism_state: dict[str, Any] = {}

        pe = getattr(trainer, "privacy_engine", None)
        if pe is not None and hasattr(pe, "noise_mechanism_config"):
            mechanism_state = dict(getattr(pe.noise_mechanism_config, "mechanism_state", {}) or {})
            sem = getattr(pe, "sampling_semantics", None)
            if sampling_mode is None and sem is not None:
                sampling_mode = getattr(sem, "sampling_mode", None)

        return mechanism, sampling_mode, mechanism_state

    def _build_c_matrix(
        self, trainer, requested_horizon: int
    ) -> tuple[torch.Tensor | None, str | None]:
        mechanism, _, state = self._mechanism_context(trainer)

        if mechanism == "gaussian":
            return torch.eye(requested_horizon, dtype=torch.float64), "identity_from_gaussian"

        if "bnb_c_matrix" in state:
            c_matrix = torch.as_tensor(state["bnb_c_matrix"], dtype=torch.float64)
            if c_matrix.ndim != 2 or c_matrix.shape[0] != c_matrix.shape[1]:
                return None, "bnb_c_matrix_not_square"

            return c_matrix, "bnb_c_matrix_state"

        coeffs = state.get("coeffs")
        if isinstance(coeffs, (list, tuple)) and len(coeffs) > 0:
            coeffs_f = [float(c) for c in coeffs]
            if not all(math.isfinite(c) for c in coeffs_f):
                return None, "non_finite_coeffs"

            if coeffs_f[0] <= 1e-12:
                return None, "invalid_c0"

            return self._lower_toeplitz_from_coeffs(coeffs_f, requested_horizon), "toeplitz_from_coeffs"

        return None, "missing_matrix_inputs"

    def compute_summary(self, trainer) -> dict[str, Any]:
        mechanism, sampling_mode, state = self._mechanism_context(trainer)
        sigma, sigma_source = self._sigma_from_trainer(trainer, mechanism, state)
        trial_index = getattr(trainer, "trial_index", None)
        training_horizon = self._resolve_horizon(trainer)

        base = {
            "trial_index": int(trial_index) if trial_index is not None else None,
            "formula_id": self.FORMULA_ID,
            "mechanism": mechanism,
            "sampling_mode": sampling_mode,
            "horizon": int(training_horizon),
            "horizon_training": int(training_horizon),
            "sigma_c": sigma,
            "sigma_source": sigma_source,
            "mf_efficiency_status": "unavailable",
            "mf_efficiency_reason": None,
            "mse_prefix": None,
            "rmse_prefix": None,
            "matrix_source": None,
            "coeff_count": (
                len(state.get("coeffs", []))
                if isinstance(state.get("coeffs", []), (list, tuple))
                else None
            ),
            "bsr_bands": state.get("bsr_bands"),
            "bnb_bands": state.get("bnb_bands"),
        }

        if sigma is None:
            base["mf_efficiency_reason"] = "missing_or_invalid_sigma"
            return base

        c_matrix, matrix_source = self._build_c_matrix(trainer, requested_horizon=training_horizon)
        base["matrix_source"] = matrix_source
        if c_matrix is None:
            base["mf_efficiency_reason"] = matrix_source
            return base

        try:
            horizon = int(c_matrix.shape[0])
            base["horizon"] = horizon
            a = self._prefix_matrix(horizon)

            # Avoid explicit matrix inverse for better numerical behavior.
            ac_inv = torch.linalg.solve(c_matrix.T, a.T).T
            fro_sq = float(torch.sum(ac_inv * ac_inv).item())
            mse_prefix = float(fro_sq * float(sigma) * float(sigma))
            rmse_prefix = float(math.sqrt(max(mse_prefix, 0.0)))
        except Exception as exc:
            base["mf_efficiency_reason"] = f"compute_failed:{type(exc).__name__}"
            return base

        base["mf_efficiency_status"] = "computed"
        base["mf_efficiency_reason"] = None
        base["mse_prefix"] = mse_prefix
        base["rmse_prefix"] = rmse_prefix

        return base

    def on_train_end(self, trainer, *args, **kwargs):
        is_global_zero = self._is_global_zero()
        summary = self.compute_summary(trainer) if is_global_zero else None

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            shared = [summary]
            torch.distributed.broadcast_object_list(shared, src=0)
            summary = shared[0]

        if summary is None:
            return

        self.latest_summary = summary
        setattr(trainer, "mf_efficiency_summary", summary)

        if not is_global_zero:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        with safe_open(self.log_dir / "mf_efficiency.json", "w") as fh:
            json.dump(summary, fh, sort_keys=True, indent=2)

        history_path = self.log_dir / "mf_efficiency_trials.jsonl"
        with history_path.open("a") as fh:
            fh.write(json.dumps(summary, sort_keys=True) + "\n")
