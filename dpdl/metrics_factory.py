from dataclasses import dataclass
from typing import Optional, Dict
from torchmetrics.text import Perplexity

import logging
import torch
import torchmetrics

log = logging.getLogger(__name__)

def _get_classification_metrics(
    output_dim: int,
    sync: bool,
    with_confusion_matrix: bool,
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

    return torchmetrics.MetricCollection(metrics)


class LanguageModelMetrics(torchmetrics.MetricCollection):
    def __init__(self, vocab_size: int, ignore_index: int, sync: bool) -> None:
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

            self['Perplexity'].update(shift_logits, shift_labels)

            for name, metric in self.items():
                if name == 'Perplexity':
                    continue

                metric.update(shift_logits_flat, shift_labels_flat)

            return

        return super().update(preds, target)


def _get_language_model_metrics(
    vocab_size: int,
    ignore_index: int,
    sync: bool,
) -> torchmetrics.MetricCollection:

    return LanguageModelMetrics(
        vocab_size=vocab_size,
        ignore_index=ignore_index,
        sync=sync,
    )


class CustomAccuracyLog(torchmetrics.Metric):
    """A log-only metric that just stores a pre-computed scalar.

    The DiseaseTask generation eval (evaluate_diseases_accuracy_exact_matching)
    already aggregates its counts across ranks via all_reduce and then only rank
    0 calls update(). We therefore disable torchmetrics' own sync
    (sync_on_compute=False) so it doesn't all-reduce again and divide by the
    world size, yielding a value that is too low by a factor of world_size.
    """
    def __init__(self):
        super().__init__(sync_on_compute=False)
        self.add_state('value', default=torch.tensor(0.0), dist_reduce_fx='mean')

    def update(self, value: float):
        self.value = torch.tensor(value, device=self.device)

    def compute(self):
        return self.value


class DiseaseMetrics(LanguageModelMetrics):
    """Token-level LM metrics + generation-based disease/PII/confidence logs.

    The token-level metrics (MulticlassAccuracy, Perplexity) are updated during
    the normal forward pass via update(preds, target). The CustomAccuracyLog
    entries are updated by key from DiseaseTaskAdapter.eval_acc after generation,
    so we must NOT feed them the token logits here (they expect a scalar).
    """
    _LM_KEYS = ('MulticlassAccuracy', 'Perplexity')

    def __init__(self, vocab_size: int, ignore_index: int, sync: bool) -> None:
        super().__init__(vocab_size=vocab_size, ignore_index=ignore_index, sync=sync)
        self.add_metrics({
            # Fraction of generated answers that contain the correct disease name (utility).
            'MulticlassAccuracyDisease': CustomAccuracyLog(),
            # Per-field substring match rates.
            # disease / symptoms / treatment measure utility;
            # name / country / occupation / hobby measure PII leakage (should be low with DP).
            'AccuracyName': CustomAccuracyLog(),
            'AccuracyCountry': CustomAccuracyLog(),
            'AccuracyOccupation': CustomAccuracyLog(),
            'AccuracyHobby': CustomAccuracyLog(),
            'AccuracySymptoms': CustomAccuracyLog(),
            'AccuracyTreatment': CustomAccuracyLog(),
            # Confidence metrics derived from generation log-probabilities.
            # ConfidenceWeightedAccuracyDisease weights each prediction by the
            # model's probability of generating the disease tokens; the
            # MeanLogProb* pair exposes calibration (correct should score higher).
            'ConfidenceWeightedAccuracyDisease': CustomAccuracyLog(),
            'MeanLogProbCorrect': CustomAccuracyLog(),
            'MeanLogProbIncorrect': CustomAccuracyLog(),
        })

    def update(self, preds, target) -> None:
        # Only the token-level LM metrics are driven by the forward pass; the
        # CustomAccuracyLog entries are updated by key elsewhere.
        if not hasattr(preds, 'ndim'):
            return

        if preds.ndim == 3:
            shift_logits = preds[:, :-1, :].contiguous()
            shift_labels = target[:, 1:].contiguous()
            shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels_flat = shift_labels.view(-1)

            self['Perplexity'].update(shift_logits, shift_labels)
            self['MulticlassAccuracy'].update(shift_logits_flat, shift_labels_flat)
            return

        self['Perplexity'].update(preds, target)
        self['MulticlassAccuracy'].update(preds, target)


def _metrics_diseases(
    vocab_size: int,
    ignore_index: int,
    sync: bool,
) -> torchmetrics.MetricCollection:

    return DiseaseMetrics(
        vocab_size=vocab_size,
        ignore_index=ignore_index,
        sync=sync,
    )


class MetricsFactory:

    @staticmethod
    def get_metrics(
        configuration,
        output_dim: Optional[int] = None,
    ) -> Dict[str, torchmetrics.MetricCollection]:
        task = configuration.task

        # we only validate on rank 0, so there's no need to
        # synchronize when calculating the metrics.
        train_sync, eval_sync = True, False

        if task in ('ImageClassification', 'SequenceClassification'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing classification metrics.')

            if not output_dim or output_dim < 1:
                raise ValueError('output_dim required for classification tasks')

            train = _get_classification_metrics(
                output_dim=output_dim,
                sync=train_sync,
                with_confusion_matrix=False,
            )
            valid = _get_classification_metrics(
                output_dim=output_dim,
                sync=eval_sync,
                with_confusion_matrix=False,
            )
            test = _get_classification_metrics(
                output_dim=output_dim,
                sync=eval_sync,
                with_confusion_matrix=True,
            )

        elif task in ('CausalLM', 'InstructLM'):
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing language model metrics.')

            vocab_size = int(output_dim)
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
            if torch.distributed.get_rank() == 0:
                log.info(f'Task is "{configuration.task}", initializing disease metrics.')

            vocab_size = int(output_dim)
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
