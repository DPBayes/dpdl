import torchmetrics

from .configurationmanager import Configuration, Hyperparameters

class MetricsFactory:

    @staticmethod
    def get_metrics(configuration: Configuration, num_classes: int | None = None):

        metrics = {'train_metrics':None,'valid_metrics':None,'test_metrics':None}


        if configuration.task == 'ImageClassification':
            metrics['train_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="macro",
                    ).cuda(),
                    "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="micro",
                    ).cuda(),
                    "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="none",
                    ).cuda(),
                }
            )

            # we only validate on rank 0, so there's no need to
            # synchronize when calculating the metrics.
            # NB: If `sync_on_compute` is enabled, this breaks
            # distributed training. If this needs to be enabled,
            # then we also need to actually run the validation on
            # all the GPUs.
            metrics['valid_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="macro",
                        sync_on_compute=False,
                    ).cuda(),
                    "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="micro",
                        sync_on_compute=False,
                    ).cuda(),
                    "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="none",
                        sync_on_compute=False,
                    ).cuda(),
                }
            )

            metrics['test_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="macro",
                        sync_on_compute=False,
                    ).cuda(),
                    "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="micro",
                        sync_on_compute=False,
                    ).cuda(),
                    "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="none",
                        sync_on_compute=False,
                    ).cuda(),
                    "ConfusionMatrix": torchmetrics.ConfusionMatrix(
                        task="multiclass" if num_classes > 2 else "binary",
                        num_classes=num_classes,
                        sync_on_compute=False,
                    ).cuda(),
                }
            )
        elif configuration.task == 'SequenceClassification':
            metrics['train_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="macro",
                    ).cuda(),
                    "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="micro",
                    ).cuda(),
                    "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="none",
                    ).cuda(),
                }
            )

            # we only validate on rank 0, so there's no need to
            # synchronize when calculating the metrics.
            # NB: If `sync_on_compute` is enabled, this breaks
            # distributed training. If this needs to be enabled,
            # then we also need to actually run the validation on
            # all the GPUs.
            metrics['valid_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="macro",
                        sync_on_compute=False,
                    ).cuda(),
                    "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="micro",
                        sync_on_compute=False,
                    ).cuda(),
                    "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="none",
                        sync_on_compute=False,
                    ).cuda(),
                }
            )

            metrics['test_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="macro",
                        sync_on_compute=False,
                    ).cuda(),
                    "MulticlassAccuracyWithMicro": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="micro",
                        sync_on_compute=False,
                    ).cuda(),
                    "MulticlassAccuracyPerClass": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=num_classes,
                        average="none",
                        sync_on_compute=False,
                    ).cuda(),
                    "ConfusionMatrix": torchmetrics.ConfusionMatrix(
                        task="multiclass" if num_classes > 2 else "binary",
                        num_classes=num_classes,
                        sync_on_compute=False,
                    ).cuda(),
                }
            )
        elif configuration.task == 'CausalLM':
            # Determine vocab size; fallback to num_classes if provided
            vocab_size = getattr(configuration, 'vocab_size', None) or (num_classes if num_classes is not None else 0)
            metrics['train_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=vocab_size,
                        average="macro",
                    ).cuda(),
                    "Perplexity": torchmetrics.text.Perplexity().cuda()
                }
            )

            # we only validate on rank 0, so there's no need to
            # synchronize when calculating the metrics.
            # NB: If `sync_on_compute` is enabled, this breaks
            # distributed training. If this needs to be enabled,
            # then we also need to actually run the validation on
            # all the GPUs.
            metrics['valid_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=vocab_size,
                        average="macro",
                        sync_on_compute=False,
                    ).cuda(),
                    "Perplexity": torchmetrics.text.Perplexity().cuda(),
                }
            )

            metrics['test_metrics'] = torchmetrics.MetricCollection(
                {
                    "MulticlassAccuracy": torchmetrics.classification.MulticlassAccuracy(
                        num_classes=vocab_size,
                        average="macro",
                        sync_on_compute=False,
                    ).cuda(),
                    "Perplexity": torchmetrics.text.Perplexity().cuda()
                }
            )
        
        return metrics
