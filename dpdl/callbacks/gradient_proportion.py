import csv
import os
import torch
from .base_callback import Callback
import logging

log = logging.getLogger(__name__)


class RecordClippedProportionsPerClassCallback(Callback):
    def __init__(self, log_dir: str, max_grad_norm: float):
        super().__init__()

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
