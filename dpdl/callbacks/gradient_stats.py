import csv
import os
import torch
from .base_callback import Callback
import logging

log = logging.getLogger(__name__)


class RecordGradientStatisticsCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float):
        super().__init__() # We'll get `global_step` from super

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
        super().on_train_batch_end(trainer, batch_idx, *args, **kwargs)

        row_data_samples = {'step': self.global_step}
        row_data_features = {'step': self.global_step}

        # Calculate the statistics for the logical batch, first over the samples ...
        if self.statistics_samples['mean']:
            # Stack accumulated statistics
            mean_samples = torch.cat(self.statistics_samples['mean'], dim=0)
            median_samples = torch.cat(self.statistics_samples['median'], dim=0)
            std_samples = torch.cat(self.statistics_samples['std'], dim=0)

            # Nonparametric skewness over samples
            skewness_samples = ((mean_samples - median_samples) / (std_samples + 1e-6))
            skewness_mean_samples = skewness_samples.mean().item()
            skewness_std_samples = skewness_samples.std().item()

            row_data_samples['Nonparametric_Skewness'] = skewness_mean_samples
            row_data_samples['Nonparametric_Skewness_Std'] = skewness_std_samples

        # ... and then over the features
        if self.statistics_features['mean']:
            # Stack accumulated statistics
            mean_features = torch.cat(self.statistics_features['mean'], dim=0)
            median_features = torch.cat(self.statistics_features['median'], dim=0)
            std_features = torch.cat(self.statistics_features['std'], dim=0)

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
