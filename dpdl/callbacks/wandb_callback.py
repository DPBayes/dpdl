import logging
import wandb
from typing import Dict, Iterable, Optional

from ..utils import tensor_to_python_type
from .base_callback import Callback

log = logging.getLogger(__name__)


class WandbCallback(Callback):
    """
    A callback to log training/validation metrics to Weights & Biases (wandb).

    Features:
    - Log train loss at each logical step to wandb (with `step` and optional `epoch` fields).
    - Log epoch-level train/validation metrics.
    - Compute and log differences (train - val) for a list of metric keys when both are available.

    Usage:
        cb = WandbCallback(project='myproj', run_name='exp1', log_dir='runs')

    Parameters:
    - project, run_name, log_dir: passed to `wandb.init`.
        - log_train_loss_per_step: whether to log per-step train loss.
        - The callback will automatically compute differences for any metric keys
            present in both train and validation/test metric dicts passed by the
            trainer (no need to pass `metric_keys`).
    - Note: `step` and a numeric `epoch` field are always recorded for batch-level logs.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        log_train_loss_per_step: bool = True,
    ) -> None:
        super().__init__()

        self.project = project
        self.run_name = run_name
        self.log_dir = log_dir
        self.log_train_loss_per_step = log_train_loss_per_step

        self._initialized = False
        self._wandb_run = None
        self._current_epoch: Optional[int] = None

        self._last_train_metrics: Dict[str, object] = {}
        self._last_val_metrics: Dict[str, object] = {}

        # Metrics that are not scalar and should not be logged as single
        # numeric values (e.g. confusion matrices, per-class vectors).
        self._metrics_to_ignore = [
            "ConfusionMatrix",
            # "MulticlassAccuracyPerClass",
        ]

        self._last_train_loss: Optional[float] = None

    def _init_wandb(self):
        if self._initialized:
            return

        if not self._is_global_zero():
            return

        if wandb is None:
            log.info("wandb package not available; WandbCallback disabled.")
            self._initialized = True
            return

        init_kwargs = {}
        if self.project:
            init_kwargs["project"] = self.project
        if self.run_name:
            init_kwargs["name"] = self.run_name
        if self.log_dir:
            init_kwargs["dir"] = self.log_dir

        try:
            self._wandb_run = wandb.init(**init_kwargs)
            log.info(f"Initialized wandb run: {self._wandb_run}")
        except Exception as e:
            log.exception(f"Failed to initialize wandb; disabling WandbCallback: {e}")
            self._wandb_run = None

        self._initialized = True

    def _ensure_initialized(self):
        """Ensure wandb is initialized on rank 0. This is a cheap idempotent
        wrapper so other methods can call it without worrying about multiple
        initializations; it will only attempt to init once."""
        if self._initialized:
            return

        self._init_wandb()

    def on_train_start(self, trainer):
        self._ensure_initialized()

    def on_train_epoch_start(self, trainer, epoch):
        """
        Record the start of an epoch. We store the epoch locally so batch logs
        can include the epoch number, and we also emit an epoch-start event to
        wandb.
        """
        self._current_epoch = int(epoch + 1) if epoch is not None else None

        if not self._is_global_zero() or wandb is None:
            return

        self._ensure_initialized()
        if self._wandb_run is None:
            return

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, **kwargs):
        super().on_train_batch_end(trainer, batch_idx, batch, loss, **kwargs)

        try:
            self._last_train_loss = float(loss)
        except Exception:
            try:
                self._last_train_loss = float(loss.item())
            except Exception:
                self._last_train_loss = None

        if not self._is_global_zero() or wandb is None:
            return

        if not self.log_train_loss_per_step:
            return

        self._ensure_initialized()
        if self._wandb_run is None:
            return

        payload = {
            "train/loss": (
                float(self._last_train_loss)
                if self._last_train_loss is not None
                else None
            ),
            "step": int(self.global_step),
        }
        if self._current_epoch is not None:
            payload["epoch"] = int(self._current_epoch)

        try:
            # commit=False so subsequent metrics within the same step/epoch can be combined
            wandb.log(payload, step=self.global_step, commit=False)
        except Exception:
            log.exception("Failed to log train batch loss to wandb")

    def on_train_batch_start(self, trainer, batch_idx, batch):
        """
        Record and log a training step at the beginning of the logical batch.

        Note: the global_step counter defined on the base Callback is
        incremented in `on_train_batch_end`. To produce a human-friendly
        1-based step number for logs at batch-start we use `self.global_step +
        1`.
        """
        if not self._is_global_zero() or wandb is None:
            return

        self._ensure_initialized()
        if self._wandb_run is None:
            return

        step_to_log = int(self.global_step + 1)

        payload = {"step": step_to_log}
        if self._current_epoch is not None:
            payload["epoch"] = int(self._current_epoch)

        try:
            wandb.log(payload, step=step_to_log, commit=False)
        except Exception:
            log.exception("Failed to log train batch start to wandb")

    def on_train_epoch_end(self, trainer, epoch, metrics):
        if not self._is_global_zero() or wandb is None:
            return

        self._ensure_initialized()
        if self._wandb_run is None:
            return

        metrics_py = tensor_to_python_type(metrics) if metrics else {}
        self._last_train_metrics = metrics_py

        # Prepare scalar epoch-level metrics, excluding non-scalar keys
        to_log = {
            f"train/{k}": v
            for k, v in metrics_py.items()
            if k not in self._metrics_to_ignore and k != "MulticlassAccuracyPerClass"
        }

        # If per-class accuracy vector is present, expand it into per-class scalars
        percls = metrics_py.get("MulticlassAccuracyPerClass")
        if percls is not None:
            try:
                for i, val in enumerate(percls):
                    to_log[f"train_perclsacc/acc_{i}"] = float(val)
            except Exception:
                # if conversion fails, skip per-class logging
                log.exception("Failed to convert/train per-class accuracy values")

        to_log["train/epoch"] = int(epoch + 1)

        try:
            wandb.log(to_log, step=self.global_step, commit=True)
        except Exception:
            log.exception("Failed to log train epoch metrics to wandb")

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        if not self._is_global_zero() or wandb is None:
            return

        self._ensure_initialized()
        if self._wandb_run is None:
            return

        try:
            val_loss = float(loss)
        except Exception:
            try:
                val_loss = float(loss.item())
            except Exception:
                val_loss = None

        payload = {"val/loss_batch": val_loss, "step": int(self.global_step)}
        if self._current_epoch is not None:
            payload["epoch"] = int(self._current_epoch)

        # if we have a recent train loss, log the difference
        if self._last_train_loss is not None and val_loss is not None:
            payload["diff/loss_train_minus_val_batch"] = float(
                self._last_train_loss - val_loss
            )

        try:
            wandb.log(payload, step=self.global_step, commit=False)
        except Exception:
            log.exception("Failed to log validation batch info to wandb")

    def on_validation_epoch_end(self, trainer, epoch, metrics):

        if not self._is_global_zero() or wandb is None:
            return

        self._ensure_initialized()
        if self._wandb_run is None:
            return

        val_metrics_py = tensor_to_python_type(metrics) if metrics else {}
        self._last_val_metrics = val_metrics_py

        # Prepare scalar validation epoch-level metrics, excluding non-scalar keys
        to_log = {
            f"val/{k}": v
            for k, v in val_metrics_py.items()
            if k not in self._metrics_to_ignore and k != "MulticlassAccuracyPerClass"
        }

        # Expand per-class accuracy into scalars under val_perclsacc/acc_{i}
        percls_val = val_metrics_py.get("MulticlassAccuracyPerClass")
        if percls_val is not None:
            try:
                for i, val in enumerate(percls_val):
                    to_log[f"val_perclsacc/acc_{i}"] = float(val)
            except Exception:
                log.exception("Failed to convert/val per-class accuracy values")

        to_log["validation/epoch"] = int(epoch + 1)

        # compute and log differences for any scalar metrics present in both train and val
        train_keys = set(
            k
            for k in self._last_train_metrics.keys()
            if k not in self._metrics_to_ignore and k != "MulticlassAccuracyPerClass"
        )
        val_keys = set(
            k for k in self._last_val_metrics.keys() if k not in self._metrics_to_ignore and k != "MulticlassAccuracyPerClass"
        )
        common_keys = train_keys & val_keys
        for key in common_keys:
            try:
                train_v = float(self._last_train_metrics[key])
                val_v = float(self._last_val_metrics[key])
                to_log[f"diff/{key}"] = train_v - val_v
            except Exception:
                # if values aren't floats or conversion fails, skip
                pass

        try:
            # commit this to make the epoch-level logs flush
            wandb.log(to_log, step=self.global_step, commit=True)
        except Exception:
            log.exception("Failed to log validation epoch metrics to wandb")

    def on_test_epoch_end(self, trainer, epoch, metrics):
        """Log test epoch metrics to wandb with namespaced 'test/' keys."""
        if not self._is_global_zero() or wandb is None:
            return

        self._ensure_initialized()
        if self._wandb_run is None:
            return

        test_metrics_py = tensor_to_python_type(metrics) if metrics else {}

        # Prepare scalar test epoch-level metrics, excluding non-scalar keys
        to_log = {
            f"test/{k}": v
            for k, v in test_metrics_py.items()
            if k not in self._metrics_to_ignore and k != "MulticlassAccuracyPerClass"
        }

        percls_test = test_metrics_py.get("MulticlassAccuracyPerClass")
        if percls_test is not None:
            try:
                for i, val in enumerate(percls_test):
                    to_log[f"test_perclsacc/acc_{i}"] = float(val)
            except Exception:
                log.exception("Failed to convert/test per-class accuracy values")

        to_log["test/epoch"] = int(epoch + 1)

        try:
            wandb.log(to_log, step=self.global_step, commit=True)
        except Exception:
            log.exception("Failed to log test epoch metrics to wandb")

    def on_train_end(self, trainer):
        # close wandb run if initialized
        if not self._is_global_zero():
            return

        if wandb is None:
            return

        if getattr(self, "_wandb_run", None) is not None:
            try:
                wandb.finish()
            except Exception:
                log.exception("Failed to finish wandb run")


__all__ = ["WandbCallback"]

