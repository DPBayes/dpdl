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
