import csv
import logging
import math
import json
import os
import pathlib
import torch
import torchmetrics
import torch.nn.functional as F

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

    def on_train_physical_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, loss):
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

        metrics = tensor_to_python_type(metrics)

        log.info(annotation + ":")
        for key, value in metrics.items():
            log.info(f" - {key}: {value}.")

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


class RecordCosineSimilarityCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float):
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm
        self.cosine_similarities_history = []
        self.accumulated_gradients = []

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumulated gradients at the start of each logical batch
        self.accumulated_gradients = []

    def on_train_physical_batch_end(self, trainer, *args, **kwargs):
        with torch.no_grad():
            # Collect per-sample gradients for the current physical batch
            per_sample_gradients = [
                g.view(len(g), -1) for g in trainer.optimizer.grad_samples
            ]

            # Concatenate the gradients across layers for each example
            flattened_gradients = torch.cat(per_sample_gradients, dim=-1)

            # Accumulate gradients across physical batches
            self.accumulated_gradients.append(flattened_gradients)

    def on_train_batch_end(self, trainer, *args, **kwargs):
        with torch.no_grad():
            # Concatenate all accumulated gradients after logical batch completion
            all_gradients = torch.cat(self.accumulated_gradients, dim=0)

            # Clipping gradients based on norm threshold
            clipped_gradients = torch.clamp(all_gradients, max=self.max_grad_norm)

            # Mean and median aggregation
            mean_grad_unclipped = torch.mean(all_gradients, dim=0)
            mean_grad_clipped = torch.mean(clipped_gradients, dim=0)
            median_grad_unclipped = torch.median(all_gradients, dim=0).values

            # Cosine similarities
            clipped_mean_vs_unclipped_mean = F.cosine_similarity(
                mean_grad_clipped,
                mean_grad_unclipped,
                dim=0,
                eps=1e-8,
            )

            clipped_mean_vs_unclipped_median = F.cosine_similarity(
                mean_grad_clipped,
                median_grad_unclipped,
                dim=0,
                eps=1e-8,
            )

            unclipped_mean_vs_unclipped_median = F.cosine_similarity(
                mean_grad_unclipped,
                median_grad_unclipped,
                dim=0,
                eps=1e-8,
            )

            # Save cosine similarity per logical batch
            self.cosine_similarities_history.append({
                'clipped_mean_vs_unclipped_mean': clipped_mean_vs_unclipped_mean.item(),
                'clipped_mean_vs_unclipped_median': clipped_mean_vs_unclipped_median.item(),
                'unclipped_mean_vs_unclipped_median': unclipped_mean_vs_unclipped_median.item(),
            })

            # Reset for the next logical batch
            self.accumulated_gradients = []

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            file_path = os.path.join(self.log_dir, 'cosine-similarities.csv')

            with open(file_path, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    'Step',
                    'Clipped_Mean_vs_Unclipped_Mean',
                    'Clipped_Mean_vs_Unclipped_Median',
                    'Unclipped_Mean_vs_Unclipped_Median',
                ])

                for step, record in enumerate(self.cosine_similarities_history):
                    writer.writerow([
                        step,
                        record['clipped_mean_vs_unclipped_mean'],
                        record['clipped_mean_vs_unclipped_median'],
                        record['unclipped_mean_vs_unclipped_median'],
                    ])

            log.info(f'Cosine similarity data saved at {file_path}')


class RecordPerClassAccuracyCallback(Callback):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.per_class_accuracies_history = []

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # At the end of the logical batch, we compute and save per-class accuracies
        train_metrics = trainer._unwrap_model().train_metrics.compute()

        # Extract per-class accuracies
        per_class_accuracies = train_metrics.get('MulticlassAccuracyPerClass', None)

        if per_class_accuracies is not None:
            # Convert to a list for easier logging and saving
            per_class_accuracies_list = per_class_accuracies.tolist()

            self.per_class_accuracies_history.append(per_class_accuracies_list)

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            file_path = os.path.join(self.log_dir, 'per-class-accuracies.csv')

            # Save the per-class accuracies history to a CSV
            with open(file_path, 'w', newline='') as fh:
                writer = csv.writer(fh)

                # Construct header row for the CSV
                header = ['Step']

                for i in range(len(self.per_class_accuracies_history[0])):
                    header += [f'Class_{i}']

                writer.writerow(header)

                # Write each batch's per-class accuracies
                for i, accuracies in enumerate(self.per_class_accuracies_history):
                    writer.writerow([i] + accuracies)

            log.info(f'Per-class accuracy data saved at {file_path}')


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

    def on_train_physical_batch_start(self, trainer, batch_idx, batch):
        log.info(f"[DEBUG] on_train_physical_batch_start")

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG] on_train_batch_end")

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG] on_train_physical_batch_end")

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

            #callbacks.append(
            #    RecordGradientNormsCallback(
            #        log_dir=full_log_dir, max_grad_norm=max_grad_norm
            #    )
            #)

            callbacks.append(
                RecordCosineSimilarityCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordPerClassAccuracyCallback(
                    log_dir=full_log_dir,
                )
            )

        if configuration.verbose_callback:
            callbacks.append(DebugProbeCallback())

        return callbacks
