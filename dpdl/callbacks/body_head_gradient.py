import csv
import os
import torch
from .base_callback import Callback
import logging

log = logging.getLogger(__name__)


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
                cls_mask = class_labels == cls
                if cls_mask.sum() == 0:
                    continue

                # Body gradient norms
                body_grad_norms = []
                for module in self.body.modules():
                    if (
                        hasattr(module, "weight")
                        and module.weight is not None
                        and module.weight.requires_grad
                    ):
                        grad_samples = module.weight.grad_sample.view(
                            len(module.weight.grad_sample), -1
                        )
                        body_grad_norms.append(grad_samples[cls_mask].norm(p=2, dim=-1))

                if body_grad_norms:
                    body_class_norm = torch.cat(body_grad_norms).norm(p=2)
                    self.body_norms_per_class[cls].append(body_class_norm.item())

                # Head gradient norms
                head_grad_norms = []
                for module in self.head.modules():
                    if (
                        hasattr(module, "weight")
                        and module.weight is not None
                        and module.weight.requires_grad
                    ):
                        grad_samples = module.weight.grad_sample.view(
                            len(module.weight.grad_sample), -1
                        )
                        head_grad_norms.append(grad_samples[cls_mask].norm(p=2, dim=-1))

                if head_grad_norms:
                    head_class_norm = torch.cat(head_grad_norms).norm(p=2)
                    self.head_norms_per_class[cls].append(head_class_norm.item())

            # Accumulate labels for later use
            self.accumulated_labels.append(class_labels)

    def on_train_batch_end(self, trainer, batch_idx, *args, **kwargs):
        # Calculate the mean norm for each class, body, and head
        row_data = {"step": batch_idx}

        for cls in range(self.num_classes):
            if self.body_norms_per_class[cls]:
                body_mean_norm = sum(self.body_norms_per_class[cls]) / len(
                    self.body_norms_per_class[cls]
                )
            else:
                body_mean_norm = 0.0

            if self.head_norms_per_class[cls]:
                head_mean_norm = sum(self.head_norms_per_class[cls]) / len(
                    self.head_norms_per_class[cls]
                )
            else:
                head_mean_norm = 0.0

            # Record the mean norms for body and head for this class
            row_data[f"Class_{cls}_Body_Mean_Norm"] = body_mean_norm
            row_data[f"Class_{cls}_Head_Mean_Norm"] = head_mean_norm

        # Save the data for this batch
        self.grad_history.append(row_data)

    def on_train_end(self, trainer, *args, **kwargs):
        # Save the gradient norms to a CSV file
        if self._is_global_zero(trainer):
            file_path = os.path.join(
                self.log_dir, "gradient_norms_body_head_per_class.csv"
            )
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
        fieldnames = ["step"]
        fieldnames += [f"Class_{cls}_Body_Mean_Norm" for cls in range(self.num_classes)]
        fieldnames += [f"Class_{cls}_Head_Mean_Norm" for cls in range(self.num_classes)]

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.grad_history)

        log.info(f"Gradient norms (body & head per class) saved to {file_path}")
