import json
import os
import torch
import logging
from ..utils import tensor_to_python_type
from .base_callback import Callback

log = logging.getLogger(__name__)


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
