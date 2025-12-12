from dataclasses import dataclass
from typing import Optional, Dict
from torchmetrics.text import Perplexity

import logging
import torch
import torchmetrics

log = logging.getLogger(__name__)

def _get_classification_metrics(
    num_classes: int,
    sync: bool,
    with_confusion_matrix: bool,
) -> torchmetrics.MetricCollection:
    # NB: If `sync_on_compute` is enabled, this breaks
    # distributed training. If this needs to be enabled,
    # then we also need to actually run the validation on
    # all the GPUs.
    metrics = {
        'MulticlassAccuracy': torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            average='macro',
            sync_on_compute=sync,
        ),
        'MulticlassAccuracyWithMicro': torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            average='micro',
            sync_on_compute=sync,
        ),
        'MulticlassAccuracyPerClass': torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes,
            average='none',
            sync_on_compute=sync,
        ),
    }

    if with_confusion_matrix:
        metrics['ConfusionMatrix'] = torchmetrics.ConfusionMatrix(
            task='multiclass' if num_classes > 2 else 'binary',
            num_classes=num_classes,
            sync_on_compute=sync,
        )

    return torchmetrics.MetricCollection(metrics).cuda()


def _get_language_model_metrics(
    vocab_size: int,
    ignore_index: int,
    sync: bool,
) -> torchmetrics.MetricCollection:
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

    return torchmetrics.MetricCollection(metrics).cuda()

class CustomAccuracyLog(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="mean")
    
    def update(self, value: float):
        """Update with your pre-calculated accuracy"""
        self.value = torch.tensor(value, device=self.device)
    
    def compute(self):
        """Return the stored value"""
        return self.value

def _metrics_diseases(
    vocab_size: int,
    ignore_index: int,
    sync: bool,
) -> torchmetrics.MetricCollection:
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
        'MulticlassAccuracyDisease': CustomAccuracyLog(
        ),
    }
    
    return torchmetrics.MetricCollection(metrics).cuda()


class MetricsFactory:

    @staticmethod
    def get_metrics(
        configuration,
        num_classes: Optional[int] = None,
    ) -> Dict[str, torchmetrics.MetricCollection]:
        task = configuration.task

        # we only validate on rank 0, so there's no need to
        # synchronize when calculating the metrics.
        train_sync, eval_sync = True, False

        if task in ('ImageClassification', 'SequenceClassification'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing classification metrics.')

            if not num_classes or num_classes < 1:
                raise ValueError('num_classes required for classification tasks')

            train = _get_classification_metrics(
                num_classes=num_classes,
                sync=train_sync,
                with_confusion_matrix=False,
            )
            valid = _get_classification_metrics(
                num_classes=num_classes,
                sync=eval_sync,
                with_confusion_matrix=False,
            )
            test = _get_classification_metrics(
                num_classes=num_classes,
                sync=eval_sync,
                with_confusion_matrix=True,
            )

        elif task in ('CausalLM', 'InstructLM'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing language model metrics.')

            vocab_size = int(num_classes)
            ignore_index = -100

            train = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=train_sync,
            )
            valid = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
            )
            test = _get_language_model_metrics(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
            )

        elif task == 'DiseaseTask':
            vocab_size = int(num_classes)
            ignore_index = -100
            train = _metrics_diseases(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=train_sync,
            )
            valid = _metrics_diseases(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
            )
            test = _metrics_diseases(
                vocab_size=vocab_size,
                ignore_index=ignore_index,
                sync=eval_sync,
            )

        else:
            raise ValueError(f'No metrics defined for task: {task}')

        metrics = {'train_metrics': train, 'valid_metrics': valid, 'test_metrics': test}
        return metrics
