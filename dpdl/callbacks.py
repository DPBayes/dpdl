import csv
import logging
import math
import json
import os
import pathlib
import torch
import torchmetrics

from collections import defaultdict
from typing import List

from .configurationmanager import Configuration, Hyperparameters
from .utils import tensor_to_python_type

log = logging.getLogger(__name__)


class CallbackHandler:
    def __init__(self, callbacks: list = []):
        self.callbacks = callbacks

    def call(self, event, *args, **kwargs):
        for callback in self.callbacks:
            event_handler = getattr(callback, event)
            event_handler(*args, **kwargs)


class Callback:
    def _is_global_zero(self, trainer):
        return torch.distributed.get_rank() == 0

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_train_epoch_start(self, trainer, epoch):
        pass

    def on_train_epoch_end(self, trainer, epoch, epoch_loss):
        pass

    def on_train_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def on_validation_epoch_start(self, trainer, epoch):
        pass

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_validation_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def on_test_epoch_start(self, trainer, epoch):
        pass

    def on_test_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_test_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def _log_metrics(self, metrics, annotation="Metrics"):
        if not metrics:
            return

        log.info(annotation + ":")
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    value = float(value)
                    log.info(f" - {key}: {value:.4f}.")
                else:
                    value = value.tolist()
                    log.info(f" - {key}: {value}.")
            else:
                log.info(f" - {key}: {value:.4f}.")


class RecordEpochStatsCallback(Callback):
    def __init__(self, use_steps=False):
        self.use_steps = use_steps

        self.train_loss = torchmetrics.aggregation.MeanMetric().cuda()
        self.evaluation_loss = torchmetrics.aggregation.MeanMetric(
            sync_on_compute=False
        ).cuda()

    def on_train_start(self, trainer):
        if self._is_global_zero(trainer):
            if self.use_steps:
                batch_size = trainer.datamodule.batch_size
                data_size = len(trainer.get_dataloader("train").dataset)
                steps_per_epoch = data_size / batch_size
                epochs = math.ceil(trainer.total_steps / steps_per_epoch)

                log.info(
                    f"!!! Starting training for approximately {epochs} epochs ({trainer.total_steps} steps)."
                )
            else:
                log.info(f"!!! Starting training for {trainer.epochs} epochs.")

    def on_train_end(self, trainer):
        if self._is_global_zero(trainer):
            log.info("!!! Training finished.")

    def on_train_epoch_start(self, trainer, epoch):
        self.train_loss.reset()

        if self._is_global_zero(trainer):
            log.info(f"--------------------------------------------------")
            if not self.use_steps:
                log.info(f"Starting training epoch {epoch+1}.")
            else:
                log.info(f"Starting training approximate epoch {epoch+1}.")

    def on_train_epoch_end(self, trainer, epoch, metrics):
        loss = self.train_loss.compute()

        if self._is_global_zero(trainer):
            if not self.use_steps:
                log.info(f"Epoch {epoch+1} finished. Loss: {loss:.4f}.")
            else:
                log.info(f"Approximate epoch {epoch+1} finished. Loss: {loss:.4f}.")

            self._log_metrics(metrics, "Train metrics")

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        self.train_loss.update(loss)

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if torch.distributed.get_rank() == 0:
            log.info(f"Validation finished. Loss: {loss:.4f}.")
            self._log_metrics(metrics, "Validation metrics")

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)

    def on_test_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if torch.distributed.get_rank() == 0:
            log.info(f"Test finished. Loss: {loss:.4f}.")
            self._log_metrics(metrics, "Test metrics")

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)


class RecordSNR(Callback):
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir
        self.grad_norms = []
        self.noise_norms = []
        self.snr_values = []

    def on_train_batch_end(self, trainer):
        grad_norm = trainer.optimizer._previous_grad.norm().item()
        noise_norm = trainer.optimizer._previous_noise.norm().item()
        snr = (
            grad_norm / noise_norm if noise_norm != 0 else float("inf")
        )  # NB: div by zero

        self.grad_norms.append(grad_norm)
        self.noise_norms.append(noise_norm)
        self.snr_values.append(snr)

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            file_path = os.path.join(self.log_dir, "signal-to-noise-ratio.csv")

            with open(file_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["Step", "Grad_Norm", "Noise_Norm", "SNR"])

                for step in range(len(self.grad_norms)):
                    writer.writerow(
                        [
                            step,
                            self.grad_norms[step],
                            self.noise_norms[step],
                            self.snr_values[step],
                        ]
                    )

            log.info(f"Signal-to-Noise ratio data saved at {file_path}")


class RecordGradientNormsCallback(Callback):
    def __init__(self, log_dir: str = None, max_grad_norm: float = 0.0):
        from collections import defaultdict

        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm

        # cache data at each epoch for mini-batch and/or multi-gpu
        self.norms_per_layer_sample = {}
        self.proportion_per_class = {}
        self.layer_mean_norms_per_class = {}

        # record the history
        self.grad_history = []

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs):
        # compute per sample norms from optimizer, which is a list contains NN's weights and bias
        per_layer_per_sample_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in trainer.optimizer.grad_samples
        ]
        _, class_labels = batch

        for class_label in class_labels.unique():
            class_mask = class_labels == class_label
            # preperation: concatenate the per-sample norm for each class
            if class_label.item() not in self.norms_per_layer_sample.keys():
                self.norms_per_layer_sample[class_label.item()] = list(
                    norms[class_mask] for norms in per_layer_per_sample_norms
                )
            else:
                for layer_index in range(len(per_layer_per_sample_norms)):
                    self.norms_per_layer_sample[class_label.item()][layer_index] = (
                        torch.cat(
                            (
                                self.norms_per_layer_sample[class_label.item()][
                                    layer_index
                                ],
                                per_layer_per_sample_norms[layer_index][class_mask],
                            ),
                            dim=0,
                        )
                    )
            # preperation: concatenate the per-sample norm for each class
            masked_per_layer_per_sample_norms = [
                norms[class_mask] for norms in per_layer_per_sample_norms
            ]
            # clipped proportion: compute the norm by sample
            per_sample_norms = torch.norm(
                torch.stack(masked_per_layer_per_sample_norms), p=2, dim=0
            )
            # clipped proportion: compute the proportion of clipped samples
            clipped_num_this_class = (
                (per_sample_norms > self.max_grad_norm).sum().item()
            )
            total_num_this_class = sum(class_mask)
            if total_num_this_class == 0:
                self.proportion_per_class[class_label] = 0
            else:
                self.proportion_per_class[class_label] = (
                    clipped_num_this_class / total_num_this_class
                )

            # layer norm: computing the mean norm over layer
            for (
                class_label,
                layer_norms_per_class,
            ) in self.norms_per_layer_sample.items():
                mean_layer_norms_per_class = []
                for layer_norms in layer_norms_per_class:
                    mean_layer_norms_per_class.append(
                        torch.mean(layer_norms, dim=0).item()
                    )
                self.layer_mean_norms_per_class[class_label] = (
                    mean_layer_norms_per_class
                )

        # save
        if self._is_global_zero(trainer):
            self.grad_history.append(
                {
                    "step": batch_idx,
                    "data": self.layer_mean_norms_per_class,
                    "clipped_proportion": self.proportion_per_class,
                }
            )

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero(trainer):
            file_path = os.path.join(self.log_dir, f"gradient_norms_last.json")
            converted_data = tensor_to_python_type(self.grad_history)
            with open(file_path, "w") as fh:
                json.dump(converted_data, fh)

            log.info(f"Gradient norm data saved to {file_path}")


class DebugProbeCallback(Callback):
    def _is_global_zero(self, trainer):
        log.info(f"[DEBUG] Calling _is_global_zero")
        return torch.distributed.get_rank() == 0

    def on_train_start(self, trainer):
        log.info(f"[DEBUG] on_train_start")

    def on_train_end(self, trainer):
        log.info(f"[DEBUG] on_train_end")

    def on_train_epoch_start(self, trainer, epoch):
        log.info(f"[DEBUG] on_train_epoch_start")

    def on_train_epoch_end(self, trainer, epoch, epoch_loss):
        log.info(f"[DEBUG] on_train_epoch_end")

    def on_train_batch_start(self, trainer, batch_idx, batch):
        log.info(f"[DEBUG] on_train_batch_start")

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG] on_train_batch_end")

    def on_validation_epoch_start(self, trainer, epoch):
        log.info(f"[DEBUG] on_validation_epoch_start")

    def on_validation_epoch_end(self, trainer, epoch, metrics):
        log.info(f"[DEBUG] on_validation_epoch_end")

    def on_validation_batch_start(self, trainer, batch_idx, batch):
        log.info(f"[DEBUG] on_validation_batch_start")

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG] on_validation_batch_end")

    def on_test_epoch_start(self, trainer, epoch):
        log.info(f"[DEBUG] on_test_epoch_start")

    def on_test_epoch_end(self, trainer, epoch, metrics):
        log.info(f"[DEBUG] on_test_epoch_end")

    def on_test_batch_start(self, trainer, batch_idx, batch):
        log.info(f"[DEBUG] on_test_batch_start")

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG] on_test_batch_end")


class CallbackFactory:
    @staticmethod
    def get_callbacks(
        configuration: Configuration, hyperparams: Hyperparameters
    ) -> List[Callback]:
        callbacks = [
            RecordEpochStatsCallback(use_steps=configuration.use_steps),
        ]

        if configuration.record_snr:
            log_dir = configuration.log_dir
            experiment_name = configuration.experiment_name
            full_log_dir = pathlib.Path(f"{log_dir}/{experiment_name}")

            callbacks.append(RecordSNR(log_dir=full_log_dir))

        if configuration.record_gradient_norms:
            log_dir = configuration.log_dir
            experiment_name = configuration.experiment_name
            full_log_dir = pathlib.Path(f"{log_dir}/{experiment_name}")
            max_grad_norm = hyperparams.max_grad_norm

            callbacks.append(
                RecordGradientNormsCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

        if configuration.verbose_callback:
            callbacks.append(DebugProbeCallback())

        return callbacks
