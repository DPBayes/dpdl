import csv
import os
import torch
from .base_callback import Callback
from dpdl.utils import geometric_median
import logging

log = logging.getLogger(__name__)


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
                cls_mask = class_labels == cls
                if cls_mask.sum() == 0:
                    continue

                self.gradients_per_class[cls].append(flattened_gradients[cls_mask])

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        row_data_samples = {"step": batch_idx}
        row_data_features = {"step": batch_idx}

        for cls in range(self.num_classes):
            if self.gradients_per_class[cls]:
                all_gradients = torch.cat(self.gradients_per_class[cls], dim=0)

                # Nonparametric Skewness and Geometric Median-based Skewness over Samples
                row_data_samples.update(
                    self._calculate_nonparametric_skewness(
                        cls, all_gradients, over="samples"
                    )
                )
                row_data_samples.update(
                    self._calculate_geo_median_based_skewness(
                        cls, all_gradients, over="samples"
                    )
                )

                # Mean-based Kurtosis over Samples
                row_data_samples.update(
                    self._calculate_mean_based_kurtosis(
                        cls, all_gradients, over="samples"
                    )
                )

                # Nonparametric Skewness and Geometric Median-based Skewness over Features
                row_data_features.update(
                    self._calculate_nonparametric_skewness(
                        cls, all_gradients, over="features"
                    )
                )
                row_data_features.update(
                    self._calculate_geo_median_based_skewness(
                        cls, all_gradients, over="features"
                    )
                )

                # Mean-based Kurtosis over Features
                row_data_features.update(
                    self._calculate_mean_based_kurtosis(
                        cls, all_gradients, over="features"
                    )
                )

        # Save the data for this batch
        self.grad_history_samples.append(row_data_samples)
        self.grad_history_features.append(row_data_features)

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero(trainer):
            self._save_to_csv(
                "gradient_statistics_per_class_over_samples.csv",
                self.grad_history_samples,
            )
            self._save_to_csv(
                "gradient_statistics_per_class_over_features.csv",
                self.grad_history_features,
            )

    def _calculate_nonparametric_skewness(self, cls, all_gradients, over="samples"):
        row_data = {}

        # Switch dimensions if calculating over features
        if over == "features":
            all_gradients = all_gradients.T

        # Compute mean, median, and standard deviation
        mean = torch.mean(all_gradients, dim=0)
        median = torch.median(all_gradients, dim=0).values
        std = torch.std(all_gradients, dim=0)

        # Nonparametric skewness: (mean - median) / std
        nonparametric_skewness = (mean - median) / (std + 1e-6)
        nonparametric_skewness_std = nonparametric_skewness.std().item()
        nonparametric_skewness = nonparametric_skewness.mean().item()

        row_data[f"Class_{cls}_Nonparametric_Skewness"] = nonparametric_skewness
        row_data[f"Class_{cls}_Nonparametric_Skewness_Std"] = nonparametric_skewness_std

        return row_data

    def _calculate_geo_median_based_skewness(self, cls, all_gradients, over="samples"):
        row_data = {}

        if over == "features":
            all_gradients = all_gradients.T

        # Compute the geometric median
        geo_median = geometric_median(all_gradients).median

        # Compute mean and standard deviation
        mean = torch.mean(
            all_gradients, dim=0
        )  # Mean over samples or features depending on transpose
        std = torch.std(all_gradients, dim=0)

        # Geometric median-based skewness: (mean - geometric median) / std
        geo_median_skewness = (mean - geo_median) / (std + 1e-6)
        geo_median_skewness_std = geo_median_skewness.std().item()
        geo_median_skewness = geo_median_skewness.mean().item()

        row_data[f"Class_{cls}_Geo_Skewness"] = geo_median_skewness
        row_data[f"Class_{cls}_Geo_Skewness_Std"] = geo_median_skewness_std

        return row_data

    def _calculate_mean_based_kurtosis(self, cls, all_gradients, over="samples"):
        row_data = {}

        if over == "features":
            all_gradients = all_gradients.T

        # Compute the mean and deviations from the mean
        mean_per_sample = torch.mean(all_gradients, dim=0)

        # Correct the dimension of unsqueeze based on the target
        diffs_per_sample = all_gradients - mean_per_sample.unsqueeze(0)

        # Compute standard deviation and Z-scores
        std_per_sample = torch.sqrt(torch.mean(diffs_per_sample**2.0, dim=0))
        zscores_per_sample = diffs_per_sample / (std_per_sample.unsqueeze(0) + 1e-6)

        # Compute kurtosis (mean-based)
        mean_kurtosis_per_sample = torch.mean(zscores_per_sample**4.0, dim=0) - 3.0
        mean_kurtosis = torch.mean(mean_kurtosis_per_sample).item()

        row_data[f"Class_{cls}_Mean_Kurtosis"] = mean_kurtosis

        return row_data

    def _save_to_csv(self, file_name, grad_history):
        fieldnames = ["step"]
        fieldnames += [
            f"Class_{cls}_Nonparametric_Skewness" for cls in range(self.num_classes)
        ]
        fieldnames += [
            f"Class_{cls}_Nonparametric_Skewness_Std" for cls in range(self.num_classes)
        ]
        fieldnames += [f"Class_{cls}_Geo_Skewness" for cls in range(self.num_classes)]
        fieldnames += [
            f"Class_{cls}_Geo_Skewness_Std" for cls in range(self.num_classes)
        ]
        fieldnames += [f"Class_{cls}_Mean_Kurtosis" for cls in range(self.num_classes)]

        file_path = os.path.join(self.log_dir, file_name)

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(grad_history)

        log.info(f"{file_name} saved successfully at {file_path}.")
