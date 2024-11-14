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
    def _is_global_zero(self):
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
        if self._is_global_zero():
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
        if self._is_global_zero():
            log.info("!!! Training finished.")

    def on_train_epoch_start(self, trainer, epoch):
        self.train_loss.reset()

        if self._is_global_zero():
            log.info(f"--------------------------------------------------")
            if not self.use_steps:
                log.info(f"Starting training epoch {epoch+1}.")
            else:
                log.info(f"Starting training approximate epoch {epoch+1}.")

    def on_train_epoch_end(self, trainer, epoch, metrics):
        loss = self.train_loss.compute()

        if self._is_global_zero():
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

        if self._is_global_zero():
            log.info(f"Validation finished. Loss: {loss:.4f}.")
            self._log_metrics(metrics, "Validation metrics")

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        self.evaluation_loss.update(loss)

    def on_test_epoch_end(self, trainer, epoch, metrics):
        loss = self.evaluation_loss.compute()
        self.evaluation_loss.reset()

        if self._is_global_zero():
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
        if self._is_global_zero():
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
        # Initialize accumulators for each class at the start of each logical batch
        self.body_norms_accumulator = {cls: [] for cls in range(self.num_classes)}
        self.head_norms_accumulator = {cls: [] for cls in range(self.num_classes)}

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs):
        batch_data, class_labels = batch

        with torch.no_grad():
            for cls in range(self.num_classes):
                cls_mask = (class_labels == cls)
                if cls_mask.sum() == 0:
                    continue

                # Body gradient norms per class
                body_grad_norms = [
                    module.weight.grad_sample[cls_mask].view(-1).norm(p=2)
                    for module in self.body.modules()
                    if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad
                ]
                if body_grad_norms:
                    mean_body_norm = torch.stack(body_grad_norms).mean().item()
                    self.body_norms_accumulator[cls].append(mean_body_norm)

                # Head gradient norms per class
                head_grad_norms = [
                    module.weight.grad_sample[cls_mask].view(-1).norm(p=2)
                    for module in self.head.modules()
                    if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad
                ]
                if head_grad_norms:
                    mean_head_norm = torch.stack(head_grad_norms).mean().item()
                    self.head_norms_accumulator[cls].append(mean_head_norm)

                # Clear memory after each physical batch
                del body_grad_norms, head_grad_norms
                torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        # Aggregate the per-class gradient norms for the logical batch
        row_data = {'step': batch_idx}

        for cls in range(self.num_classes):
            # Calculate mean of body and head norms across physical batches
            body_mean_norm = (
                sum(self.body_norms_accumulator[cls]) / len(self.body_norms_accumulator[cls])
                if self.body_norms_accumulator[cls] else 0.0
            )
            head_mean_norm = (
                sum(self.head_norms_accumulator[cls]) / len(self.head_norms_accumulator[cls])
                if self.head_norms_accumulator[cls] else 0.0
            )

            # Store the calculated norms
            row_data[f'Class_{cls}_Body_Mean_Norm'] = body_mean_norm
            row_data[f'Class_{cls}_Head_Mean_Norm'] = head_mean_norm

        # Save the data for this batch
        self.grad_history.append(row_data)

        # Clear accumulators for the next logical batch
        for cls in range(self.num_classes):
            self.body_norms_accumulator[cls].clear()
            self.head_norms_accumulator[cls].clear()

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            file_path = os.path.join(self.log_dir, 'gradient_norms_body_head_per_class.csv')
            self._save_to_csv(file_path)

    def _get_body_and_head(self, model):
        head = model.get_classifier()
        body = torch.nn.Sequential(*list(model.children())[:-1])

        return body, head

    def _save_to_csv(self, file_path):
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
        self.quantiles = quantiles
        self.cosine_similarities_history = []

    def on_train_start(self, *args, **kwargs):
        # Accumulate statistics at the physical batch level here
        self.mean_grad_clipped_accumulator = []
        self.mean_grad_unclipped_accumulator = []
        self.median_grad_unclipped_accumulator = []

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumlators at the start of logical batch
        self.mean_grad_clipped_accumulator.clear()
        self.mean_grad_unclipped_accumulator.clear()
        self.median_grad_unclipped_accumulator.clear()

    def on_train_physical_batch_end(self, trainer, *args, **kwargs):
        with torch.no_grad():
            per_sample_gradients = [g.view(len(g), -1) for g in trainer.optimizer.grad_samples]
            flattened_gradients = torch.cat(per_sample_gradients, dim=-1)

            # Compute per-sample norms and clipping factors
            per_sample_norms = flattened_gradients.norm(p=2, dim=1)
            clip_factors = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)
            clip_factors = clip_factors.unsqueeze(-1)

            # Clip gradients
            clipped_gradients = clip_factors * flattened_gradients

            # Compute and accumulate per-physical-batch statistics
            self.mean_grad_unclipped_accumulator.append(flattened_gradients.mean(dim=0))
            self.mean_grad_clipped_accumulator.append(clipped_gradients.mean(dim=0))
            self.median_grad_unclipped_accumulator.append(torch.median(flattened_gradients, dim=0).values)

        # Make sure memory is cleared
        del per_sample_gradients, flattened_gradients, per_sample_norms, clip_factors, clipped_gradients
        torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        with torch.no_grad():
            # Stack the accumulated means and medians of physical batches
            mean_unclipped_stack = torch.stack(self.mean_grad_unclipped_accumulator, dim=0)
            mean_clipped_stack = torch.stack(self.mean_grad_clipped_accumulator, dim=0)
            median_unclipped_stack = torch.stack(self.median_grad_unclipped_accumulator, dim=0)

            # Compute aggregated means and medians for the logical batch
            mean_grad_unclipped = mean_unclipped_stack.mean(dim=0)
            mean_grad_clipped = mean_clipped_stack.mean(dim=0)
            median_grad_unclipped = torch.median(median_unclipped_stack, dim=0).values

            # Compute cosine similarities
            cosine_similarities = {
                'clipped_mean_vs_unclipped_mean': torch.nn.functional.cosine_similarity(
                    mean_grad_clipped, mean_grad_unclipped, dim=0, eps=1e-8
                ).item(),
                'clipped_mean_vs_unclipped_median': torch.nn.functional.cosine_similarity(
                    mean_grad_clipped, median_grad_unclipped, dim=0, eps=1e-8
                ).item(),
                'unclipped_mean_vs_unclipped_median': torch.nn.functional.cosine_similarity(
                    mean_grad_unclipped, median_grad_unclipped, dim=0, eps=1e-8
                ).item(),
            }

            # Cosine similarities against the quantiles
            all_norms = torch.stack([g.norm(p=2) for g in self.mean_grad_unclipped_accumulator])
            for quantile in self.quantiles:
                threshold = torch.quantile(all_norms, quantile).item()
                quantile_clip_factor = (threshold / (all_norms.mean() + 1e-6)).clamp(max=1.0)
                quantile_clipped_mean_grad = quantile_clip_factor * mean_grad_unclipped

                cosine_similarities[f'clipped_{quantile}_vs_clipped_mean'] = torch.nn.functional.cosine_similarity(
                    quantile_clipped_mean_grad, mean_grad_clipped, dim=0, eps=1e-8
                ).item()
                cosine_similarities[f'clipped_{quantile}_vs_unclipped_mean'] = torch.nn.functional.cosine_similarity(
                    quantile_clipped_mean_grad, mean_grad_unclipped, dim=0, eps=1e-8
                ).item()
                cosine_similarities[f'clipped_{quantile}_vs_unclipped_median'] = torch.nn.functional.cosine_similarity(
                    quantile_clipped_mean_grad, median_grad_unclipped, dim=0, eps=1e-8
                ).item()

            # Store results for this logical batch
            self.cosine_similarities_history.append(cosine_similarities)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
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
        self.num_classes = trainer.datamodule.get_num_classes()
        self.cosine_similarities_history = {cls: [] for cls in range(self.num_classes)}

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumulators for each logical batch
        self.sum_mean_grad_unclipped = {cls: 0 for cls in range(self.num_classes)}
        self.sum_mean_grad_clipped = {cls: 0 for cls in range(self.num_classes)}
        self.num_batches = {cls: 0 for cls in range(self.num_classes)}

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs):
        with torch.no_grad():
            batch_data, class_labels = batch

            # Collect per-sample gradients for the current physical batch
            per_sample_gradients = [g.view(len(g), -1) for g in trainer.optimizer.grad_samples]
            flattened_gradients = torch.cat(per_sample_gradients, dim=-1)

            # Make sure ememory is cleared
            del per_sample_gradients
            torch.cuda.empty_cache()

            # Process gradients for each class
            for cls in range(self.num_classes):
                mask = (class_labels == cls)
                if mask.sum() == 0:
                    continue

                # Select the gradients of current class
                class_gradients = flattened_gradients[mask]
                class_norms = class_gradients.norm(p=2, dim=1)

                # Compute clipped gradients
                class_clip_factors = (self.max_grad_norm / (class_norms + 1e-6)).clamp(max=1.0).unsqueeze(-1)
                clipped_gradients = class_clip_factors * class_gradients

                # Compute mean gradients for this class in this physical batch
                mean_grad_unclipped = class_gradients.mean(dim=0)
                mean_grad_clipped = clipped_gradients.mean(dim=0)

                # Accumulate sums for means
                self.sum_mean_grad_unclipped[cls] += mean_grad_unclipped
                self.sum_mean_grad_clipped[cls] += mean_grad_clipped
                self.num_batches[cls] += 1

                # Make sure memory is cleared
                del mask, class_gradients, class_norms, class_clip_factors, clipped_gradients
                torch.cuda.empty_cache()

            # Make sure memory is cleared
            del batch_data, class_labels, flattened_gradients
            torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        with torch.no_grad():
            # Compute aggregate statistic for each class out of the running means
            for cls in range(self.num_classes):
                if self.num_batches[cls] == 0:
                    continue

                # Compute aggregated means for this class
                mean_grad_unclipped = self.sum_mean_grad_unclipped[cls] / self.num_batches[cls]
                mean_grad_clipped = self.sum_mean_grad_clipped[cls] / self.num_batches[cls]

                # Calculate cosine similarities from the aggregated means
                cosine_similarity = torch.nn.functional.cosine_similarity(
                    mean_grad_clipped, mean_grad_unclipped, dim=0, eps=1e-8
                ).item()

                # Store the cosine similarity results for this class
                self.cosine_similarities_history[cls].append({
                    'step': batch_idx,
                    'clipped_mean_vs_unclipped_mean': cosine_similarity
                })

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            # Save cosine similarities for each class separately
            for cls in range(self.num_classes):
                file_path = os.path.join(self.log_dir, f'cosine_similarity_class_{cls}.csv')
                self._save_to_csv(file_path, self.cosine_similarities_history[cls], f'Cosine Similarities (Class {cls})')

    def _save_to_csv(self, file_path, history, column_name):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Step',
                'Clipped_Mean_vs_Unclipped_Mean',
            ])

            for record in history:
                writer.writerow([
                    record['step'],
                    record['clipped_mean_vs_unclipped_mean'],
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
        if self._is_global_zero():
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

        # Initialize container for the clipped proportions
        self.clipped_proportions_history = []

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumlated gradients and labels at the start of logical batch
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

        # Make sure memory is cleared
        del per_sample_gradients, flattened_gradients
        torch.cuda.empty_cache()

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
        if self._is_global_zero():
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

    def on_train_batch_start(self, *args, **kwargs):
        # Reset accumulators at the start of each logical batch
        self.statistics_samples = {'mean': [], 'median': [], 'std': []}
        self.statistics_features = {'mean': [], 'median': [], 'std': []}

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, *args, **kwargs):
        batch_data, class_labels = batch

        with torch.no_grad():
            # Collect gradients for all layers and flatten them for each sample
            per_sample_gradients = [g.view(len(g), -1) for g in trainer.optimizer.grad_samples]
            flattened_gradients = torch.cat(per_sample_gradients, dim=-1)

            # Compute statistics over samples
            mean_samples = flattened_gradients.mean(dim=0)
            median_samples = torch.median(flattened_gradients, dim=0).values
            std_samples = flattened_gradients.std(dim=0)

            self.statistics_samples['mean'].append(mean_samples)
            self.statistics_samples['median'].append(median_samples)
            self.statistics_samples['std'].append(std_samples)

            # Compute statistics over features
            mean_features = flattened_gradients.mean(dim=1)
            median_features = torch.median(flattened_gradients, dim=1).values
            std_features = flattened_gradients.std(dim=1)

            self.statistics_features['mean'].append(mean_features)
            self.statistics_features['median'].append(median_features)
            self.statistics_features['std'].append(std_features)

        # Make sure memory is cleared
        del per_sample_gradients, flattened_gradients
        torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        row_data_samples = {'step': batch_idx}
        row_data_features = {'step': batch_idx}

        # Calculate the statistics for the logical batch, first over the samples ...
        if self.statistics_samples['mean']:
            # Stack accumulated statistics
            mean_samples_concat = torch.cat(self.statistics_samples['mean'], dim=0)
            median_samples_concat = torch.cat(self.statistics_samples['median'], dim=0)
            std_samples_concat = torch.cat(self.statistics_samples['std'], dim=0)

            # Compute final statistics over the logical batch
            mean_samples = mean_samples_concat.mean(dim=0)
            median_samples = torch.median(median_samples_concat, dim=0).values
            std_samples = std_samples_concat.mean(dim=0)

            # Nonparametric skewness over samples
            skewness_samples = ((mean_samples - median_samples) / (std_samples + 1e-6))
            skewness_mean_samples = skewness_samples.mean().item()
            skewness_std_samples = skewness_samples.std().item()

            row_data_samples['Nonparametric_Skewness'] = skewness_mean_samples
            row_data_samples['Nonparametric_Skewness_Std'] = skewness_std_samples

        # ... and then over the features
        if self.statistics_features['mean']:
            # Stack accumulated statistics
            mean_features_concat = torch.cat(self.statistics_features['mean'], dim=0)
            median_features_concat = torch.cat(self.statistics_features['median'], dim=0)
            std_features_concat = torch.cat(self.statistics_features['std'], dim=0)

            # Compute final statistics over the logical batch
            mean_features = mean_features_concat.mean(dim=0)
            median_features = torch.median(median_features_concat, dim=0).values
            std_features = std_features_concat.mean(dim=0)

            # Nonparametric skewness over features
            skewness_features = ((mean_features - median_features) / (std_features + 1e-6))
            skewness_mean_features = skewness_features.mean().item()
            skewness_std_features = skewness_features.std().item()

            row_data_features['Nonparametric_Skewness'] = skewness_mean_features
            row_data_features['Nonparametric_Skewness_Std'] = skewness_std_features

        # Save the data for this logical batch
        self.grad_history_samples.append(row_data_samples)
        self.grad_history_features.append(row_data_features)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            self._save_to_csv(
                'gradient_statistics_over_samples.csv',
                self.grad_history_samples,
            )
            self._save_to_csv(
                'gradient_statistics_over_features.csv',
                self.grad_history_features,
            )

    def _save_to_csv(self, file_name, grad_history):
        fieldnames = ['step', 'Nonparametric_Skewness', 'Nonparametric_Skewness_Std']

        file_path = os.path.join(self.log_dir, file_name)

        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(grad_history)

        log.info(f'{file_name} saved successfully at {file_path}.')


class DebugProbeCallback(Callback):
    def _is_global_zero(self):
        log.info(f"[DEBUG] Calling _is_global_zero")
        super().__is_global_zero()

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
