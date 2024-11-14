import logging
import pathlib
import torch

from typing import List

from .configurationmanager import Configuration, Hyperparameters
from .utils import tensor_to_python_type

from .callbacks import *

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
        callbacks = [
            RecordEpochStatsCallback(use_steps=configuration.use_steps),
        ]

        if configuration.record_snr:
            log_dir = configuration.log_dir
            experiment_name = configuration.experiment_name
            full_log_dir = pathlib.Path(f"{log_dir}/{experiment_name}")

            callbacks.append(RecordSNR(log_dir=full_log_dir))

        if configuration.record_gradient_norms:
            log_dir = configuration.log_dir
            experiment_name = configuration.experiment_name
            full_log_dir = pathlib.Path(f"{log_dir}/{experiment_name}")
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

        return callbacks
