import logging
import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .callbacks.callback_factory import CallbackFactory, CallbackHandler
from .configurationmanager import ConfigurationManager
from .datamodules import DataModule, DataModuleFactory
from .experimentmanager import (
    save_gradient_diagnostics,
    save_predict_metrics,
    save_predictions,
)
from .models.model_base import ModelBase
from .models.model_factory import ModelFactory
from .trainer import Trainer, TrainerFactory
from .utils import tensor_to_python_type

log = logging.getLogger(__name__)


def _all_gather_object_list(local_list):
    world = torch.distributed.get_world_size()
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, list(local_list))

    merged = []
    for part in gathered:
        merged.extend(part)

    return merged


class Predictor:

    def __init__(
        self,
        trainer: Trainer,
        dataset_split: str,
        config_manager: ConfigurationManager,
        save_gradient_data: Optional[bool] = False,
    ):
        """
        init Predictor

        args:
            trainer (Trainer): initialized Trainer object
            dataset_split (str): for a specific split of dataset, e.g., 'train', 'valid', 'test'
            save_gradient_data (bool): boolean indicating if we should save also gradient data
        """
        self.trainer = trainer
        self.dataset_split = dataset_split
        self.config_manager = config_manager
        self.save_gradient_data = save_gradient_data

    def predict(self, configuration):
        """
        Perform prediction using the model on the specified dataset split.
        All ranks compute prediction in parallel using DDP, and rank 0 gathers
        the results and saves them as a CSV file.

        Args:
            configuration: A Configuration object that provides dataset_split and log path.
        """

        # disable possible dropout, etc
        model = self.trainer.model
        model.eval()

        dataloader = self.trainer.get_dataloader(self.dataset_split)

        local_preds = []
        local_probs = []
        local_labels = []

        if self.save_gradient_data:
            grad_records = []  # collect here (label, pred, norm)

        self.trainer._unwrap_model().train_metrics.reset()

        # NB: We are using torch.func.grad which builds it's own graph
        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):

                if torch.distributed.get_rank() == 0:
                    log.info(f' - Predicting on batch {i}')

                X = X.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                logits = model(X)
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                self.trainer._unwrap_model().train_metrics.update(preds, y)

                local_preds.append(preds.cpu())
                local_probs.append(probs.cpu())
                local_labels.append(y.cpu())

                if self.save_gradient_data:
                    norms = self._per_sample_grad_norms(X, y)

                    for lbl, pred, norm in zip(y.tolist(), preds.tolist(), norms.tolist()):
                        record = {'label': lbl, 'pred': pred, 'norm': norm}
                        grad_records.append(record)

        metrics = self.trainer._unwrap_model().train_metrics.compute()

        torch.distributed.barrier()

        gathered_preds = _all_gather_object_list(local_preds)
        gathered_probs = _all_gather_object_list(local_probs)
        gathered_labels = _all_gather_object_list(local_labels)

        if torch.distributed.get_rank() == 0:
            all_preds = torch.cat([t.cpu() for t in gathered_preds])
            all_probs = torch.cat([t.cpu() for t in gathered_probs])
            all_labels = torch.cat([t.cpu() for t in gathered_labels])

            save_predictions(
                self.config_manager,
                labels=all_labels,
                preds=all_preds,
                probs=all_probs,
                split=self.dataset_split,
            )

            save_predict_metrics(self.config_manager, metrics)

        torch.distributed.barrier()

        if self.save_gradient_data:
            gathered_grad_recs = _all_gather_object_list(grad_records)
            if torch.distributed.get_rank() == 0:
                save_gradient_diagnostics(
                    self.config_manager,
                    gathered_grad_recs,
                    split=self.dataset_split,
                )

    def load_model(
        self,
        fpath: Optional[str],
    ) -> None:
        """
        Load weights into self.trainer.model.

        Args:
            fpath: Path to checkpoint file on disk.
            strict: Passed to load_state_dict.
            map_location: torch.load map_location (e.g. cpu/cuda).
        """
        model = self.trainer._unwrap_model()
        model.load_model(fpath)

        if torch.distributed.get_rank() == 0:
            log.info(f'Loaded weights from: {fpath}')

    def _get_model_params_and_buffers(self):
        model = self.trainer._unwrap_model()

        params = OrderedDict(
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        )
        buffers = OrderedDict(model.named_buffers())
        param_names = list(params.keys())

        return model, params, buffers, param_names

    def _per_sample_grad_norms(self, X: torch.Tensor, y: torch.Tensor):
        model, params, buffers, param_names = self._get_model_params_and_buffers()

        if not param_names:
            raise RuntimeError('Model has no trainable parameters to compute gradients for.')

        def loss_one(p, x, yi):
            logits = functional_call(model, (p, buffers), (x.unsqueeze(0),))
            return F.cross_entropy(logits, yi.unsqueeze(0), reduction='sum')

        gfun = grad(loss_one, argnums=0)

        # use torch.func.vmap to get per-example gradients
        per_sample_grads = vmap(
            lambda x, yi: gfun(params, x, yi),
            in_dims=(0, 0),
        )(X, y)

        batch_size = X.size(0)

        # accumulate norms here
        norms_sq = torch.zeros(batch_size, device=X.device, dtype=X.dtype)

        # compute the norms: norms_sq[i] = Σ_all_layers Σ_over_all_params( (grad[i, param])^2 )
        for name in param_names:
            grad_tensor = per_sample_grads[name]
            flat_grad = grad_tensor.reshape(batch_size, -1)

            # Σ_over_all_params_in_layer( (grad[i, param])^2 )
            norms_sq += flat_grad.square().sum(dim=1)
            per_sample_grads[name] = None  # drop reference to save memory

        return norms_sq.sqrt().clamp_min(1e-12)


class PredictorFactory:
    @staticmethod
    def get_predictor(config_manager) -> Predictor:
        """
        get predictor from configuration

        args:
            config_manager: configuration manager for managing configurations

        returns:
            Predictor: the predictor for prediction
        """
        configuration = config_manager.configuration
        hyperparams = config_manager.hyperparams

        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams)
        num_classes = datamodule.get_num_classes()

        trainer = TrainerFactory.get_trainer(config_manager)

        predictor = Predictor(
            trainer=trainer,
            dataset_split=configuration.dataset_split,
            config_manager=config_manager,
            save_gradient_data=configuration.prediction_save_gradient_data,
        )

        if fpath := getattr(configuration, 'model_weights_path', False):
            predictor.load_model(configuration.model_weights_path)

        return predictor
