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
from torch_geometric_median import geometric_median

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


class RecordBodyAndHeadGradientNormsPerClassCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float):
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm
        self.grad_history = []

    def on_train_start(self, trainer, *args, **kwargs):
        # Separate body and head of the model
        self.body, self.head = self._get_body_and_head(trainer._unwrap_model())
        self.num_classes = trainer.datamodule.get_num_classes()

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumulated norms at the start of each logical batch
        self.body_norms_per_class = {cls: [] for cls in range(self.num_classes)}
        self.head_norms_per_class = {cls: [] for cls in range(self.num_classes)}
        self.accumulated_labels = []

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs):
        batch_data, class_labels = batch

        with torch.no_grad():
            # Track body and head gradients per class
            for cls in range(self.num_classes):
                cls_mask = (class_labels == cls)
                if cls_mask.sum() == 0:
                    continue

                # Body gradient norms
                body_grad_norms = []
                for module in self.body.modules():
                    if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                        grad_samples = module.weight.grad_sample.view(len(module.weight.grad_sample), -1)
                        body_grad_norms.append(grad_samples[cls_mask].norm(p=2, dim=-1))

                if body_grad_norms:
                    body_class_norm = torch.cat(body_grad_norms).norm(p=2)
                    self.body_norms_per_class[cls].append(body_class_norm.item())

                # Head gradient norms
                head_grad_norms = []
                for module in self.head.modules():
                    if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                        grad_samples = module.weight.grad_sample.view(len(module.weight.grad_sample), -1)
                        head_grad_norms.append(grad_samples[cls_mask].norm(p=2, dim=-1))

                if head_grad_norms:
                    head_class_norm = torch.cat(head_grad_norms).norm(p=2)
                    self.head_norms_per_class[cls].append(head_class_norm.item())

            # Accumulate labels for later use
            self.accumulated_labels.append(class_labels)

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        # Calculate the mean norm for each class, body, and head
        row_data = {'step': batch_idx}

        for cls in range(self.num_classes):
            if self.body_norms_per_class[cls]:
                body_mean_norm = sum(self.body_norms_per_class[cls]) / len(self.body_norms_per_class[cls])
            else:
                body_mean_norm = 0.0

            if self.head_norms_per_class[cls]:
                head_mean_norm = sum(self.head_norms_per_class[cls]) / len(self.head_norms_per_class[cls])
            else:
                head_mean_norm = 0.0

            # Record the mean norms for body and head for this class
            row_data[f'Class_{cls}_Body_Mean_Norm'] = body_mean_norm
            row_data[f'Class_{cls}_Head_Mean_Norm'] = head_mean_norm

        # Save the data for this batch
        self.grad_history.append(row_data)

    def on_train_end(self, trainer, *args, **kwargs):
        # Save the gradient norms to a CSV file
        if self._is_global_zero(trainer):
            file_path = os.path.join(self.log_dir, 'gradient_norms_body_head_per_class.csv')
            self._save_to_csv(file_path)

    def _get_body_and_head(self, model):
        # Retrieve the head from the timm model
        head = model.get_classifier()

        # Get all layers of the model
        body_layers = []
        head_layers = set()

        # Identify parameters in the head so we can exclude them from the body
        for name, param in head.named_parameters(recurse=True):
            head_layers.add(name)

        # Everything that is not part of the head is part of the body
        for name, module in model.named_children():
            if not any(head_layer in name for head_layer in head_layers):
                body_layers.append(module)

        body = torch.nn.Sequential(*body_layers)

        return body, head

    def _save_to_csv(self, file_path):
        # Prepare fieldnames
        fieldnames = ['step']
        fieldnames += [f'Class_{cls}_Body_Mean_Norm' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Head_Mean_Norm' for cls in range(self.num_classes)]

        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.grad_history)

        log.info(f'Gradient norms (body & head per class) saved to {file_path}')


class RecordCosineSimilarityCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float, quantiles: list = [0.25, 0.5, 0.75]):
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm
        self.quantiles = quantiles  # List of gradient quantiles for comparison
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
            all_gradients = torch.cat(self.accumulated_gradients, dim=0)  # Shape (B, D)

            # Compute per-sample norms
            per_sample_norms = all_gradients.norm(p=2, dim=1)  # Shape (B)

            # Compute per-sample clip factors
            per_sample_clip_factors = (self.max_grad_norm / (per_sample_norms + 1e-6)) # Shape (B)
            per_sample_clip_factors.clamp(max=1.0)
            per_sample_clip_factors = per_sample_clip_factors.unsqueeze(-1)  # Shape (B, 1)

            # Clip the gradients
            clipped_gradients = per_sample_clip_factors * all_gradients  # Shape: (B, D)

            # Mean and median aggregation -> Shape (D)
            mean_grad_unclipped = torch.mean(all_gradients, dim=0)
            mean_grad_clipped = torch.mean(clipped_gradients, dim=0)
            median_grad_unclipped = torch.median(all_gradients, dim=0).values

            # Compute cosine similarities
            clipped_mean_vs_unclipped_mean = torch.nn.functional.cosine_similarity(
                mean_grad_clipped,
                mean_grad_unclipped,
                dim=0,
                eps=1e-8,
            )

            clipped_mean_vs_unclipped_median = torch.nn.functional.cosine_similarity(
                mean_grad_clipped,
                median_grad_unclipped,
                dim=0,
                eps=1e-8,
            )

            unclipped_mean_vs_unclipped_median = torch.nn.functional.cosine_similarity(
                mean_grad_unclipped,
                median_grad_unclipped,
                dim=0,
                eps=1e-8,
            )

            cosine_similarities = {
                'clipped_mean_vs_unclipped_mean': clipped_mean_vs_unclipped_mean.item(),
                'clipped_mean_vs_unclipped_median': clipped_mean_vs_unclipped_median.item(),
                'unclipped_mean_vs_unclipped_median': unclipped_mean_vs_unclipped_median.item(),
            }

            # Perform clipping at quantile thresholds
            for quantile in self.quantiles:
                threshold = torch.quantile(per_sample_norms, quantile)
                quantile_clip_factors = (threshold / (per_sample_norms + 1e-6)).clamp(max=1.0)
                quantile_clip_factors = quantile_clip_factors.unsqueeze(-1)  # Shape (B, 1)

                quantile_clipped_gradients = quantile_clip_factors * all_gradients  # Shape (B, D)
                mean_grad_quantile_clipped = torch.mean(quantile_clipped_gradients, dim=0)

                # Compute cosine similarities for quantile-clipped gradients
                quantile_clipped_vs_clipped_mean = torch.nn.functional.cosine_similarity(
                    mean_grad_quantile_clipped, mean_grad_clipped, dim=0, eps=1e-8
                )

                quantile_clipped_vs_unclipped_mean = torch.nn.functional.cosine_similarity(
                    mean_grad_quantile_clipped, mean_grad_unclipped, dim=0, eps=1e-8
                )

                quantile_clipped_vs_unclipped_median = torch.nn.functional.cosine_similarity(
                    mean_grad_quantile_clipped, median_grad_unclipped, dim=0, eps=1e-8
                )

                cosine_similarities[f'clipped_{quantile}_vs_clipped_mean'] = quantile_clipped_vs_clipped_mean.item()
                cosine_similarities[f'clipped_{quantile}_vs_unclipped_mean'] = quantile_clipped_vs_unclipped_mean.item()
                cosine_similarities[f'clipped_{quantile}_vs_unclipped_median'] = quantile_clipped_vs_unclipped_median.item()

            # Save cosine similarity per logical batch
            self.cosine_similarities_history.append(cosine_similarities)

            # Reset for the next logical batch
            self.accumulated_gradients = []

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            file_path = os.path.join(self.log_dir, 'cosine-similarities.csv')

            with open(file_path, 'w', newline='') as fh:
                writer = csv.writer(fh)
                header = [
                    'Step',
                    'Clipped_Mean_vs_Unclipped_Mean',
                    'Clipped_Mean_vs_Unclipped_Median',
                    'Unclipped_Mean_vs_Unclipped_Median',
                ]
                header += [f'Clipped_{quantile}_vs_Clipped_Mean' for quantile in self.quantiles]
                header += [f'Clipped_{quantile}_vs_Unclipped_Mean' for quantile in self.quantiles]
                header += [f'Clipped_{quantile}_vs_Unclipped_Median' for quantile in self.quantiles]

                writer.writerow(header)

                for step, record in enumerate(self.cosine_similarities_history):
                    row = [step]
                    row += [
                        record['clipped_mean_vs_unclipped_mean'],
                        record['clipped_mean_vs_unclipped_median'],
                        record['unclipped_mean_vs_unclipped_median']
                    ]
                    for quantile in self.quantiles:
                        row.append(record[f'clipped_{quantile}_vs_clipped_mean'])
                        row.append(record[f'clipped_{quantile}_vs_unclipped_mean'])
                        row.append(record[f'clipped_{quantile}_vs_unclipped_median'])

                    writer.writerow(row)

            log.info(f'Cosine similarity data saved at {file_path}')


class RecordPerClassCosineSimilarityCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float):
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm

    def on_train_start(self, trainer, *args, **kwargs):
        # Get the number of classes from datamodule
        self.num_classes = trainer.datamodule.get_num_classes()

        # Container for the cosine similarities
        self.cosine_similarities_history = {cls: [] for cls in range(self.num_classes)}

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumulated gradients and labels when logical batch starts
        self.accumulated_gradients = []
        self.accumulated_labels = []

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs):
        batch_data, class_labels = batch

        with torch.no_grad():
            # Collect per-sample gradients for the current physical batch
            per_sample_gradients = [
                g.view(len(g), -1) for g in trainer.optimizer.grad_samples
            ]

            # Create a Torch tensor of the per sample gradients
            flattened_gradients = torch.cat(per_sample_gradients, dim=-1)

            # Accumulate gradients and labels over physical batches
            self.accumulated_gradients.append(flattened_gradients)
            self.accumulated_labels.append(class_labels)

    def on_train_batch_end(self, trainer, *args, **kwargs):
        with torch.no_grad():
            # Concatenate all accumulated gradients and labels for this logical batch
            all_gradients = torch.cat(self.accumulated_gradients, dim=0)
            all_labels = torch.cat(self.accumulated_labels, dim=0)

            # Handle per class cosine similarity calculation
            for cls in range(self.num_classes):
                mask = (all_labels == cls)
                if mask.sum() == 0:
                    continue

                # We want the gradients that belong to this specific class
                class_gradients = all_gradients[mask]
                class_norms = class_gradients.norm(p=2, dim=1)

                # Calculate clipping factors based on class norms
                class_clip_factors = (self.max_grad_norm / (class_norms + 1e-6)).clamp(max=1.0)
                class_clip_factors = class_clip_factors.unsqueeze(-1)  # Shape (B_class, 1)

                # Apply clipping to gradients
                clipped_gradients = class_clip_factors * class_gradients

                # Compute mean and median for unclipped and clipped gradients
                mean_grad_unclipped = torch.mean(class_gradients, dim=0)
                mean_grad_clipped = torch.mean(clipped_gradients, dim=0)
                median_grad_unclipped = torch.median(class_gradients, dim=0).values

                # Compute cosine similarities
                clipped_mean_vs_unclipped_mean = torch.nn.functional.cosine_similarity(
                    mean_grad_clipped, mean_grad_unclipped, dim=0, eps=1e-8
                )

                clipped_mean_vs_unclipped_median = torch.nn.functional.cosine_similarity(
                    mean_grad_clipped, median_grad_unclipped, dim=0, eps=1e-8
                )

                unclipped_mean_vs_unclipped_median = torch.nn.functional.cosine_similarity(
                    mean_grad_unclipped, median_grad_unclipped, dim=0, eps=1e-8
                )

                # Save all cosine similarities for this class
                self.cosine_similarities_history[cls].append({
                    'clipped_mean_vs_unclipped_mean': clipped_mean_vs_unclipped_mean.item(),
                    'clipped_mean_vs_unclipped_median': clipped_mean_vs_unclipped_median.item(),
                    'unclipped_mean_vs_unclipped_median': unclipped_mean_vs_unclipped_median.item(),
                })

        # Reset for the next logical batch
        self.accumulated_gradients = []
        self.accumulated_labels = []

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            for cls in range(self.num_classes):
                file_path = os.path.join(self.log_dir, f'cosine_similarity_class_{cls}.csv')

                self._save_to_csv(
                    file_path,
                    self.cosine_similarities_history[cls],
                    f'Cosine Similarities (Class {cls})',
                )

    def _save_to_csv(self, file_path, history, column_name):
        # Save the cosine similarity history to a CSV file
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Step',
                'Clipped_Mean_vs_Unclipped_Mean',
                'Clipped_Mean_vs_Unclipped_Median',
                'Unclipped_Mean_vs_Unclipped_Median',
            ])

            for step, record in enumerate(history):
                writer.writerow([
                    step,
                    record['clipped_mean_vs_unclipped_mean'],
                    record['clipped_mean_vs_unclipped_median'],
                    record['unclipped_mean_vs_unclipped_median'],
                ])

        log.info(f'{column_name} data saved at {file_path}')


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


class RecordClippedProportionsPerClassCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float):
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm

    def on_train_start(self, trainer, *args, **kwargs):
        # Get the number of classes from the datamodule
        self.num_classes = trainer.datamodule.get_num_classes()

        # Initialize container for clipped proportions
        self.clipped_proportions_history = []

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumulated gradients and labels at the start of each logical batch
        self.accumulated_gradients = []
        self.accumulated_labels = []

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs):
        batch_data, class_labels = batch

        with torch.no_grad():
            # Collect per-sample gradients for the current physical batch
            per_sample_gradients = [
                g.view(len(g), -1) for g in trainer.optimizer.grad_samples
            ]
            # Concatenate the gradients across layers for each example
            flattened_gradients = torch.cat(per_sample_gradients, dim=-1)

            # Accumulate gradients and labels across physical batches
            self.accumulated_gradients.append(flattened_gradients)
            self.accumulated_labels.append(class_labels)

    def on_train_batch_end(self, trainer, *args, **kwargs):
        with torch.no_grad():
            # Concatenate all accumulated gradients and labels after logical batch completion
            all_gradients = torch.cat(self.accumulated_gradients, dim=0)
            all_labels = torch.cat(self.accumulated_labels, dim=0)

            clipped_proportions = []

            for cls in range(self.num_classes):
                mask = (all_labels == cls)
                if mask.sum() == 0:
                    # If no gradients for this class, mark as 0 clipped proportion
                    clipped_proportions.append(0.0)
                    continue

                # Select the gradients corresponding to the current class
                class_gradients = all_gradients[mask]
                class_norms = class_gradients.norm(2, dim=1)

                # Calculate how many gradients are clipped
                num_clipped = (class_norms > self.max_grad_norm).sum().item()
                proportion_clipped = num_clipped / class_norms.size(0)

                # Append the proportion of clipped gradients for this class
                clipped_proportions.append(proportion_clipped)

            # Save the clipped proportions for this step
            self.clipped_proportions_history.append(clipped_proportions)

        # Reset for the next logical batch
        self.accumulated_gradients = []
        self.accumulated_labels = []

    def on_train_end(self, trainer, *args, **kwargs):
        # Only save on rank 0 to avoid duplicate files in distributed setups
        if torch.distributed.get_rank() == 0:
            file_path = os.path.join(self.log_dir, 'clipped_proportions_per_class.csv')
            self._save_to_csv(file_path, self.clipped_proportions_history)

    def _save_to_csv(self, file_path, history):
        # Save the clipped proportions history to a single CSV file
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header row
            header = ['Step'] + [f'Class_{i}' for i in range(self.num_classes)]
            writer.writerow(header)

            # Write the clipped proportions for each step
            for step, proportions in enumerate(history):
                writer.writerow([step] + proportions)

        log.info(f'Clipped proportions data saved at {file_path}')


class RecordGradientStatisticsCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float):
        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm
        self.grad_history_samples = []
        self.grad_history_features = []

    def on_train_start(self, trainer, *args, **kwargs):
        self.num_classes = trainer.datamodule.get_num_classes()

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumulated gradients at the start of each logical batch
        self.gradients_per_class = {cls: [] for cls in range(self.num_classes)}

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs):
        batch_data, class_labels = batch

        with torch.no_grad():
            # Collect gradients for all layers and flatten them for each class
            per_sample_gradients = [
                g.view(len(g), -1) for g in trainer.optimizer.grad_samples
            ]
            flattened_gradients = torch.cat(per_sample_gradients, dim=-1)

            # Accumulate flattened gradients per class
            for cls in range(self.num_classes):
                cls_mask = (class_labels == cls)
                if cls_mask.sum() == 0:
                    continue

                self.gradients_per_class[cls].append(flattened_gradients[cls_mask])

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        row_data_samples = {'step': batch_idx}
        row_data_features = {'step': batch_idx}

        for cls in range(self.num_classes):
            if self.gradients_per_class[cls]:
                all_gradients = torch.cat(self.gradients_per_class[cls], dim=0)

                # Nonparametric Skewness and Geometric Median-based Skewness over Samples
                row_data_samples.update(
                    self._calculate_nonparametric_skewness(cls, all_gradients, over='samples')
                )
                row_data_samples.update(
                    self._calculate_geo_median_based_skewness(cls, all_gradients, over='samples')
                )

                # Mean-based Kurtosis over Samples
                row_data_samples.update(
                    self._calculate_mean_based_kurtosis(cls, all_gradients, over='samples')
                )

                # Nonparametric Skewness and Geometric Median-based Skewness over Features
                row_data_features.update(
                    self._calculate_nonparametric_skewness(cls, all_gradients, over='features')
                )
                row_data_features.update(
                    self._calculate_geo_median_based_skewness(cls, all_gradients, over='features')
                )

                # Mean-based Kurtosis over Features
                row_data_features.update(
                    self._calculate_mean_based_kurtosis(cls, all_gradients, over='features')
                )

        # Save the data for this batch
        self.grad_history_samples.append(row_data_samples)
        self.grad_history_features.append(row_data_features)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero(trainer):
            self._save_to_csv(
                'gradient_statistics_per_class_over_samples.csv',
                self.grad_history_samples,
            )
            self._save_to_csv(
                'gradient_statistics_per_class_over_features.csv',
                self.grad_history_features,
            )

    def _calculate_nonparametric_skewness(self, cls, all_gradients, over='samples'):
        row_data = {}

        # Switch dimensions if calculating over features
        if over == 'features':
            all_gradients = all_gradients.T

        # Compute mean, median, and standard deviation
        mean = torch.mean(all_gradients, dim=0)
        median = torch.median(all_gradients, dim=0).values
        std = torch.std(all_gradients, dim=0)

        # Nonparametric skewness: (mean - median) / std
        nonparametric_skewness = ((mean - median) / (std + 1e-6))
        nonparametric_skewness_std = nonparametric_skewness.std().item()
        nonparametric_skewness = nonparametric_skewness.mean().item()

        row_data[f'Class_{cls}_Nonparametric_Skewness'] = nonparametric_skewness
        row_data[f'Class_{cls}_Nonparametric_Skewness_Std'] = nonparametric_skewness_std

        return row_data

    def _calculate_geo_median_based_skewness(self, cls, all_gradients, over='samples'):
        row_data = {}

        if over == 'features':
            all_gradients = all_gradients.T

        # Compute the geometric median
        geo_median = geometric_median(all_gradients).median

        # Compute mean and standard deviation
        mean = torch.mean(all_gradients, dim=0)  # Mean over samples or features depending on transpose
        std = torch.std(all_gradients, dim=0)

        # Geometric median-based skewness: (mean - geometric median) / std
        geo_median_skewness = ((mean - geo_median) / (std + 1e-6))
        geo_median_skewness_std = geo_median_skewness.std().item()
        geo_median_skewness = geo_median_skewness.mean().item()

        row_data[f'Class_{cls}_Geo_Skewness'] = geo_median_skewness
        row_data[f'Class_{cls}_Geo_Skewness_Std'] = geo_median_skewness_std

        return row_data

    def _calculate_mean_based_kurtosis(self, cls, all_gradients, over='samples'):
        row_data = {}

        if over == 'features':
            all_gradients = all_gradients.T

        # Compute the mean and deviations from the mean
        mean_per_sample = torch.mean(all_gradients, dim=0)

        # Correct the dimension of unsqueeze based on the target
        diffs_per_sample = all_gradients - mean_per_sample.unsqueeze(0)

        # Compute standard deviation and Z-scores
        std_per_sample = torch.sqrt(torch.mean(diffs_per_sample ** 2.0, dim=0))
        zscores_per_sample = diffs_per_sample / (std_per_sample.unsqueeze(0) + 1e-6)

        # Compute kurtosis (mean-based)
        mean_kurtosis_per_sample = torch.mean(zscores_per_sample ** 4.0, dim=0) - 3.0
        mean_kurtosis = torch.mean(mean_kurtosis_per_sample).item()

        row_data[f'Class_{cls}_Mean_Kurtosis'] = mean_kurtosis

        return row_data

    def _save_to_csv(self, file_name, grad_history):
        fieldnames = ['step']
        fieldnames += [f'Class_{cls}_Nonparametric_Skewness' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Nonparametric_Skewness_Std' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Geo_Skewness' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Geo_Skewness_Std' for cls in range(self.num_classes)]
        fieldnames += [f'Class_{cls}_Mean_Kurtosis' for cls in range(self.num_classes)]

        file_path = os.path.join(self.log_dir, file_name)

        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(grad_history)

        log.info(f'{file_name} saved successfully at {file_path}.')


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
        configuration: Configuration, hyperparams: Hyperparameters) -> List[Callback]:
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
                RecordBodyAndHeadGradientNormsPerClassCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordCosineSimilarityCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordPerClassCosineSimilarityCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordPerClassAccuracyCallback(
                    log_dir=full_log_dir,
                )
            )

            callbacks.append(
                RecordClippedProportionsPerClassCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordGradientStatisticsCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

        if configuration.verbose_callback:
            callbacks.append(DebugProbeCallback())

        return callbacks
