import logging
from typing import Dict, Optional

import torch
import torchmetrics
from torchmetrics.text import Perplexity

from .custom_metrics_factory import CustomMetricsFactory

log = logging.getLogger(__name__)


def _build_custom_metrics(metric_config, key, default_kwargs, sync_on_compute):
    if not metric_config:
        return None
    return CustomMetricsFactory.build_metric_collection(metric_config[key], default_kwargs, sync_on_compute)


def _get_classification_metrics(
    output_dim: int,
    sync: bool,
    with_confusion_matrix: bool,
    custom_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
) -> torchmetrics.MetricCollection:
    # NB: If `sync_on_compute` is enabled, this breaks
    # distributed training. If this needs to be enabled,
    # then we also need to actually run the validation on
    # all the GPUs.
    metrics = {
        'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
            num_classes=output_dim,
            average='macro',
            sync_on_compute=sync,
        ),
        'MulticlassAccuracyWithMicro': torchmetrics.classification.MulticlassAccuracy(
            num_classes=output_dim,
            average='micro',
            sync_on_compute=sync,
        ),
        'MulticlassAccuracyPerClass': torchmetrics.classification.MulticlassAccuracy(
            num_classes=output_dim,
            average='none',
            sync_on_compute=sync,
        ),
    }

    if with_confusion_matrix:
        metrics['ConfusionMatrix'] = torchmetrics.ConfusionMatrix(
            task='multiclass' if output_dim > 2 else 'binary',
            num_classes=output_dim,
            sync_on_compute=sync,
        )

    if custom_metrics:
        metrics.update(custom_metrics)

    return torchmetrics.MetricCollection(metrics)


class LanguageModelMetrics(torchmetrics.MetricCollection):
    def __init__(
        self,
        vocab_size: int,
        ignore_index: int,
        sync: bool,
        custom_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    ) -> None:
        metrics = {
            'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
                num_classes=vocab_size,
                average='micro',
                ignore_index=ignore_index,
                sync_on_compute=sync,
            ),
            'Perplexity': Perplexity(
                ignore_index=ignore_index,
                sync_on_compute=sync,
            ),
        }

        if custom_metrics:
            metrics.update(custom_metrics)

        super().__init__(metrics)

    def update(self, preds, target) -> None:
        # Accuracy metrics use standard flattened inputs
        if not hasattr(preds, 'ndim'):
            return super().update(preds, target)

        # We need to shape the data for perplexity that expects 3D logits and 2D labels
        if preds.ndim == 3:
            shift_logits = preds[:, :-1, :].contiguous()                      # (batch, seq_len-1, vocab)
            shift_labels = target[:, 1:].contiguous()                         # (batch, seq_len-1)
            shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # (batch*(seq_len-1), vocab)
            shift_labels_flat = shift_labels.view(-1)                         # (batch*(seq_len-1))

            for metric in self.values():
                if isinstance(metric, Perplexity):
                    metric.update(shift_logits, shift_labels)
                else:
                    metric.update(shift_logits_flat, shift_labels_flat)

            return None

        return super().update(preds, target)


def _get_language_model_metrics(
    vocab_size: int,
    ignore_index: int,
    sync: bool,
    custom_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
) -> torchmetrics.MetricCollection:
    return LanguageModelMetrics(
        vocab_size=vocab_size,
        ignore_index=ignore_index,
        sync=sync,
        custom_metrics=custom_metrics,
    )


class MetricsFactory:

    @staticmethod
    def get_metrics(
        configuration,
        output_dim: Optional[int] = None,
    ) -> Dict[str, torchmetrics.MetricCollection]:
        task = configuration.task
        metric_conf = getattr(configuration, 'metric_conf', None)
        metric_config = CustomMetricsFactory.read_metric_config(metric_conf) if metric_conf else None

        # we only validate on rank 0, so there's no need to
        # synchronize when calculating the metrics.
        train_sync, eval_sync = True, False

        if task in ('ImageClassification', 'SequenceClassification'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing classification metrics.')

            if not output_dim or output_dim < 1:
                raise ValueError('output_dim required for classification tasks')

            classification_defaults = {
                'num_classes': output_dim,
                'task': 'multiclass' if output_dim > 2 else 'binary',
            }

            train = _get_classification_metrics(
                output_dim=output_dim,
                sync=train_sync,
                with_confusion_matrix=False,
                custom_metrics=_build_custom_metrics(metric_config, 'train_metrics', classification_defaults, train_sync),
            )
            valid = _get_classification_metrics(
                output_dim=output_dim,
                sync=eval_sync,
                with_confusion_matrix=False,
                custom_metrics=_build_custom_metrics(metric_config, 'valid_metrics', classification_defaults, eval_sync),
            )
            test = _get_classification_metrics(
                output_dim=output_dim,
                sync=eval_sync,
                with_confusion_matrix=True,
                custom_metrics=_build_custom_metrics(metric_config, 'test_metrics', classification_defaults, eval_sync),
            )

        elif task in ('CausalLM', 'InstructLM'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing language model metrics.')

            vocab_size = int(output_dim)
            ignore_index = -100

            language_defaults = {
                'num_classes': vocab_size,
                'ignore_index': ignore_index,
                'task': 'multiclass',
            }

            train = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=train_sync,
                custom_metrics=_build_custom_metrics(metric_config, 'train_metrics', language_defaults, train_sync),
            )
            valid = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
                custom_metrics=_build_custom_metrics(metric_config, 'valid_metrics', language_defaults, eval_sync),
            )
            test = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
                custom_metrics=_build_custom_metrics(metric_config, 'test_metrics', language_defaults, eval_sync),
            )

        else:
            raise ValueError(f'No metrics defined for task: {task}')

        metrics = {'train_metrics': train, 'valid_metrics': valid, 'test_metrics': test}
        return metrics
