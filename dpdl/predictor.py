import logging
import os
from collections import OrderedDict
from collections.abc import Mapping
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

                # Move inputs to CUDA
                if isinstance(X, Mapping):  # Special case for HF language models
                    X = {k: v.cuda(non_blocking=True) for k, v in X.items()}
                else:
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


    def _per_sample_grad_norms(self, X, y):
        model, params, buffers, param_names = self._get_model_params_and_buffers()

        if not param_names:
            raise RuntimeError('Model has no trainable parameters to compute gradients for.')

        is_mapping = isinstance(X, Mapping)

        # Figure out batch size
        if is_mapping:
            first_tensor = next(iter(X.values()))
            batch_size = first_tensor.size(0)
        else:
            batch_size = X.size(0)

        def loss_one(p, x_single, yi_single):
            # x_single is either a single example tensor or a dict of single-example tensors
            if isinstance(x_single, Mapping):
                x_batch = {k: v.unsqueeze(0) for k, v in x_single.items()}
            else:
                x_batch = x_single.unsqueeze(0)

            logits = functional_call(model, (p, buffers), (x_batch,))
            return F.cross_entropy(logits, yi_single.unsqueeze(0), reduction='sum')

        gfun = grad(loss_one, argnums=0)

        # Gradients are floating point; keep accumulation in parameter dtype to avoid Long casts
        first_param = next(iter(params.values()))
        param_dtype = first_param.dtype
        param_device = first_param.device
        norms_sq = torch.zeros(batch_size, device=param_device, dtype=param_dtype)

        if is_mapping:
            # HF / dict input: vmap is broken by data-dependent control flow, so loop per-sample
            for i in range(batch_size):
                x_i = {k: v[i] for k, v in X.items()}
                y_i = y[i]
                grads_i = gfun(params, x_i, y_i)

                total = torch.zeros((), device=param_device, dtype=param_dtype)
                for name in param_names:
                    g = grads_i[name].reshape(-1)
                    total += g.square().sum()
                norms_sq[i] = total
        else:
            # Plain tensor input: keep fast vmap path
            per_sample_grads = vmap(
                lambda x, yi: gfun(params, x, yi),
                in_dims=(0, 0),
            )(X, y)

            for name in param_names:
                grad_tensor = per_sample_grads[name]  # shape: [B, ...]
                flat_grad = grad_tensor.reshape(batch_size, -1)
                norms_sq += flat_grad.square().sum(dim=1)
                per_sample_grads[name] = None  # drop reference

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
