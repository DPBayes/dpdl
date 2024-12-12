import logging
import pathlib
from typing import List

from ..configurationmanager import Configuration, Hyperparameters
from ..utils import tensor_to_python_type
from .base_callback import Callback
from .body_head_gradient import RecordBodyAndHeadGradientNormsPerClassCallback
from .cosine_similarity import (
    RecordCosineSimilarityCallback,
    RecordPerClassCosineSimilarityCallback,
)
from .debug import DebugProbeCallback
from .epoch_stats import RecordEpochStatsCallback
from .gradient_proportion import RecordClippedProportionsPerClassCallback
from .gradient_stats import RecordGradientStatisticsCallback
from .per_class_accuracy import RecordPerClassAccuracyCallback
from .record_losses import RecordLossesByEpochCallback, RecordTrainLossByStepCallback

log = logging.getLogger(__name__)


class CallbackHandler:
    def __init__(self, callbacks: list = []):
        self.callbacks = callbacks

    def call(self, event, *args, **kwargs):
        for callback in self.callbacks:
            event_handler = getattr(callback, event)
            event_handler(*args, **kwargs)


class CallbackFactory:
    @staticmethod
    def get_callbacks(
        configuration: Configuration, hyperparams: Hyperparameters) -> List[Callback]:

        log_dir = configuration.log_dir
        experiment_name = configuration.experiment_name
        full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

        callbacks = [
            RecordEpochStatsCallback(use_steps=configuration.use_steps),
        ]

        if configuration.record_gradient_norms:
            max_grad_norm = hyperparams.max_grad_norm

            callbacks.append(
                RecordBodyAndHeadGradientNormsPerClassCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordCosineSimilarityCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordPerClassCosineSimilarityCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordPerClassAccuracyCallback(
                    log_dir=full_log_dir,
                )
            )

            callbacks.append(
                RecordClippedProportionsPerClassCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordGradientStatisticsCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

        if configuration.verbose_callback:
            callbacks.append(DebugProbeCallback())

        if configuration.record_loss_by_step:
            callbacks.append(RecordTrainLossByStepCallback(log_dir=full_log_dir))

        if configuration.record_loss_by_epoch:
            callbacks.append(RecordLossesByEpochCallback(log_dir=full_log_dir))

        return callbacks
