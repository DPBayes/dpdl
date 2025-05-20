import csv
import logging
import os

import torch

from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordCosineSimilarityCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float, quantiles: list = [0.25, 0.5, 0.75]):
        super().__init__()

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
            # Stack the accumulated means and medians of physical batches for easy processing
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
            all_per_sample_norms = torch.cat([
                g.view(len(g), -1).norm(p=2, dim=1) for g in trainer.optimizer.grad_samples
            ])

            for quantile in self.quantiles:
                threshold = torch.quantile(all_per_sample_norms, quantile)

                mean_grad_unclipped_norm = mean_grad_unclipped.norm(p=2)
                quantile_clip_factor = threshold / (mean_grad_unclipped_norm + 1e-6)
                quantile_clip_factor = quantile_clip_factor.clamp(max=1.0)

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
        super().__init__()

        self.log_dir = log_dir
        self.max_grad_norm = max_grad_norm

    def on_train_start(self, trainer, *args, **kwargs):
        self.num_classes = trainer.datamodule.get_num_classes()
        self.cosine_similarities_history = []

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

            del per_sample_gradients
            torch.cuda.empty_cache()

            for cls in range(self.num_classes):
                mask = (class_labels == cls)
                if mask.sum() == 0:
                    continue

                class_gradients = flattened_gradients[mask]
                class_norms = class_gradients.norm(p=2, dim=1)
                class_clip_factors = (self.max_grad_norm / (class_norms + 1e-6)).clamp(max=1.0).unsqueeze(-1)
                clipped_gradients = class_clip_factors * class_gradients

                mean_grad_unclipped = class_gradients.mean(dim=0)
                mean_grad_clipped = clipped_gradients.mean(dim=0)

                self.sum_mean_grad_unclipped[cls] += mean_grad_unclipped
                self.sum_mean_grad_clipped[cls] += mean_grad_clipped
                self.num_batches[cls] += 1

                del mask, class_gradients, class_norms, class_clip_factors, clipped_gradients
                torch.cuda.empty_cache()

            del batch_data, class_labels, flattened_gradients
            torch.cuda.empty_cache()

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        super().on_train_batch_end(trainer, batch_idx, *args, **kwargs)

        with torch.no_grad():
            # Create a row for this step with one column per class.
            row = {'Step': self.global_step}

            for cls in range(self.num_classes):
                if self.num_batches[cls] == 0:
                    # If no update for this class in the batch, record a placeholder
                    row[f'Clipped_Mean_vs_Unclipped_Mean_Class{cls}'] = None
                else:
                    mean_grad_unclipped = self.sum_mean_grad_unclipped[cls] / self.num_batches[cls]
                    mean_grad_clipped = self.sum_mean_grad_clipped[cls] / self.num_batches[cls]

                    cosine_similarity = torch.nn.functional.cosine_similarity(
                        mean_grad_clipped, mean_grad_unclipped, dim=0, eps=1e-8
                    ).item()

                    row[f'Clipped_Mean_vs_Unclipped_Mean_Class{cls}'] = cosine_similarity

            self.cosine_similarities_history.append(row)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            header = ['Step'] + [
                f'Clipped_Mean_vs_Unclipped_Mean_Class{cls}' for cls in range(self.num_classes)
            ]

            file_path = os.path.join(self.log_dir, 'cosine-similarity-per-class.csv')
            self._save_to_csv(file_path, self.cosine_similarities_history, header)

    def _save_to_csv(self, file_path, history, header):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for record in history:
                writer.writerow([record.get(col, None) for col in header])

        log.info(f'Cosine Similarities data saved at {file_path}')

