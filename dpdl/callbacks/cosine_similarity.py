import csv
import os
import torch
from .base_callback import Callback
import logging

log = logging.getLogger(__name__)


class RecordCosineSimilarityCallback(Callback):
    def __init__(
        self, log_dir: str, max_grad_norm: float, quantiles: list = [0.25, 0.5, 0.75]
    ):
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
            per_sample_clip_factors = self.max_grad_norm / (
                per_sample_norms + 1e-6
            )  # Shape (B)
            per_sample_clip_factors = per_sample_clip_factors.clamp(max=1.0)
            per_sample_clip_factors = per_sample_clip_factors.unsqueeze(
                -1
            )  # Shape (B, 1)

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
                "clipped_mean_vs_unclipped_mean": clipped_mean_vs_unclipped_mean.item(),
                "clipped_mean_vs_unclipped_median": clipped_mean_vs_unclipped_median.item(),
                "unclipped_mean_vs_unclipped_median": unclipped_mean_vs_unclipped_median.item(),
            }

            # Perform clipping at quantile thresholds
            for quantile in self.quantiles:
                threshold = torch.quantile(per_sample_norms, quantile)
                quantile_clip_factors = (threshold / (per_sample_norms + 1e-6)).clamp(
                    max=1.0
                )
                quantile_clip_factors = quantile_clip_factors.unsqueeze(
                    -1
                )  # Shape (B, 1)

                quantile_clipped_gradients = (
                    quantile_clip_factors * all_gradients
                )  # Shape (B, D)
                mean_grad_quantile_clipped = torch.mean(
                    quantile_clipped_gradients, dim=0
                )

                # Compute cosine similarities for quantile-clipped gradients
                quantile_clipped_vs_clipped_mean = (
                    torch.nn.functional.cosine_similarity(
                        mean_grad_quantile_clipped, mean_grad_clipped, dim=0, eps=1e-8
                    )
                )

                quantile_clipped_vs_unclipped_mean = (
                    torch.nn.functional.cosine_similarity(
                        mean_grad_quantile_clipped, mean_grad_unclipped, dim=0, eps=1e-8
                    )
                )

                quantile_clipped_vs_unclipped_median = (
                    torch.nn.functional.cosine_similarity(
                        mean_grad_quantile_clipped,
                        median_grad_unclipped,
                        dim=0,
                        eps=1e-8,
                    )
                )

                cosine_similarities[f"clipped_{quantile}_vs_clipped_mean"] = (
                    quantile_clipped_vs_clipped_mean.item()
                )
                cosine_similarities[f"clipped_{quantile}_vs_unclipped_mean"] = (
                    quantile_clipped_vs_unclipped_mean.item()
                )
                cosine_similarities[f"clipped_{quantile}_vs_unclipped_median"] = (
                    quantile_clipped_vs_unclipped_median.item()
                )

            # Save cosine similarity per logical batch
            self.cosine_similarities_history.append(cosine_similarities)

            # Reset for the next logical batch
            self.accumulated_gradients = []

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            file_path = os.path.join(self.log_dir, "cosine-similarities.csv")

            with open(file_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                header = [
                    "Step",
                    "Clipped_Mean_vs_Unclipped_Mean",
                    "Clipped_Mean_vs_Unclipped_Median",
                    "Unclipped_Mean_vs_Unclipped_Median",
                ]
                header += [
                    f"Clipped_{quantile}_vs_Clipped_Mean" for quantile in self.quantiles
                ]
                header += [
                    f"Clipped_{quantile}_vs_Unclipped_Mean"
                    for quantile in self.quantiles
                ]
                header += [
                    f"Clipped_{quantile}_vs_Unclipped_Median"
                    for quantile in self.quantiles
                ]

                writer.writerow(header)

                for step, record in enumerate(self.cosine_similarities_history):
                    row = [step]
                    row += [
                        record["clipped_mean_vs_unclipped_mean"],
                        record["clipped_mean_vs_unclipped_median"],
                        record["unclipped_mean_vs_unclipped_median"],
                    ]
                    for quantile in self.quantiles:
                        row.append(record[f"clipped_{quantile}_vs_clipped_mean"])
                        row.append(record[f"clipped_{quantile}_vs_unclipped_mean"])
                        row.append(record[f"clipped_{quantile}_vs_unclipped_median"])

                    writer.writerow(row)

            log.info(f"Cosine similarity data saved at {file_path}")


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
                mask = all_labels == cls
                if mask.sum() == 0:
                    continue

                # We want the gradients that belong to this specific class
                class_gradients = all_gradients[mask]
                class_norms = class_gradients.norm(p=2, dim=1)

                # Calculate clipping factors based on class norms
                class_clip_factors = (self.max_grad_norm / (class_norms + 1e-6)).clamp(
                    max=1.0
                )
                class_clip_factors = class_clip_factors.unsqueeze(
                    -1
                )  # Shape (B_class, 1)

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

                clipped_mean_vs_unclipped_median = (
                    torch.nn.functional.cosine_similarity(
                        mean_grad_clipped, median_grad_unclipped, dim=0, eps=1e-8
                    )
                )

                unclipped_mean_vs_unclipped_median = (
                    torch.nn.functional.cosine_similarity(
                        mean_grad_unclipped, median_grad_unclipped, dim=0, eps=1e-8
                    )
                )

                # Save all cosine similarities for this class
                self.cosine_similarities_history[cls].append(
                    {
                        "clipped_mean_vs_unclipped_mean": clipped_mean_vs_unclipped_mean.item(),
                        "clipped_mean_vs_unclipped_median": clipped_mean_vs_unclipped_median.item(),
                        "unclipped_mean_vs_unclipped_median": unclipped_mean_vs_unclipped_median.item(),
                    }
                )

        # Reset for the next logical batch
        self.accumulated_gradients = []
        self.accumulated_labels = []

    def on_train_end(self, trainer, *args, **kwargs):
        if torch.distributed.get_rank() == 0:
            for cls in range(self.num_classes):
                file_path = os.path.join(
                    self.log_dir, f"cosine_similarity_class_{cls}.csv"
                )

                self._save_to_csv(
                    file_path,
                    self.cosine_similarities_history[cls],
                    f"Cosine Similarities (Class {cls})",
                )

    def _save_to_csv(self, file_path, history, column_name):
        # Save the cosine similarity history to a CSV file
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Step",
                    "Clipped_Mean_vs_Unclipped_Mean",
                    "Clipped_Mean_vs_Unclipped_Median",
                    "Unclipped_Mean_vs_Unclipped_Median",
                ]
            )

            for step, record in enumerate(history):
                writer.writerow(
                    [
                        step,
                        record["clipped_mean_vs_unclipped_mean"],
                        record["clipped_mean_vs_unclipped_median"],
                        record["unclipped_mean_vs_unclipped_median"],
                    ]
                )

        log.info(f"{column_name} data saved at {file_path}")
