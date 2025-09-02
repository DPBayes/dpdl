import logging
import os
from collections import OrderedDict
from typing import Optional

import torch
from torch.func import grad, vmap

from .callbacks.callback_factory import CallbackFactory, CallbackHandler
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
from .configurationmanager import ConfigurationManager

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
            proto_unit, g_dim, has_bias = self._compute_class_prototypes()
            proto_unit_cpu = proto_unit.cpu()
            del proto_unit
            torch.cuda.empty_cache()

            # Hook for per-example grads
            head = self.trainer._unwrap_model().get_classifier()
            last = {'H': None}

            def _hook(mod, inputs, output):
                last['H'] = inputs[0].detach()

            hnd = head.register_forward_hook(_hook)

            grad_records = []  # collect here (label, pred, norm, angle)

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
                    H = last['H']  # [B, F] from hook
                    g, _, _ = self._per_sample_head_grads(head, H, y)  # [B, g_dim]

                    norms = g.norm(dim=1).clamp_min(1e-12)
                    pu = proto_unit_cpu.index_select(0, y.cpu()).to(y.device, non_blocking=True)  # [B, g_dim]
                    cos = (g * pu).sum(dim=1) / norms
                    angles = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))

                    for lbl, pred, norm, ang in zip(y.tolist(), preds.tolist(), norms.tolist(), angles.tolist()):
                        record = {'label': lbl, 'pred': pred, 'norm': norm, 'angle': ang}
                        grad_records.append(record)

        if self.save_gradient_data:
            # remove the hook, just in case (e.g. we don't really do anything afterwards)
            hnd.remove()

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

    def _per_sample_head_grads(self, head: torch.nn.Linear, H: torch.Tensor, y: torch.Tensor):
        """
        Compute per-sample gradients for the last Linear head using torch.func.
        Args:
            head: nn.Linear (W [C,F], b [C] or None)
            H:   [B,F] head input features (detached)
            y:   [B]   int64 labels
        Returns:
            g:   [B, gdim] where gdim = C*F (+ C if bias)
        """
        W = head.weight  # [C,F]
        b = head.bias  # [C] or None
        C, F = W.shape

        w_dim = C * F
        b_dim = C if b is not None else 0
        g_dim = w_dim + b_dim

        if b is not None:

            def loss_one(w, b_, h, yi):
                logits = h @ w.T + b_  # [C]
                return torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), yi.unsqueeze(0), reduction='sum'
                )

            gfun = grad(loss_one, argnums=(0, 1))
            gw, gb = vmap(gfun, in_dims=(None, None, 0, 0))(W, b, H, y)  # gw [B,C,F], gb [B,C]
            gw = gw.reshape(gw.size(0), w_dim)  # [B,w_dim]
            g = torch.cat([gw, gb], dim=1)  # [B,g_dim]
        else:

            def loss_one(w, h, yi):
                logits = h @ w.T
                return torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), yi.unsqueeze(0), reduction='sum'
                )

            gfun = grad(loss_one, argnums=0)
            gw = vmap(gfun, in_dims=(None, 0, 0))(W, H, y)  # [B,C,F]
            g = gw.reshape(gw.size(0), w_dim)  # [B,g_dim]

        return g, g_dim, (b is not None)

    def _compute_class_prototypes(self):
        """
        Forward-only + torch.func grads to build per-class prototype directions (unit vectors).
        """

        if torch.distributed.get_rank() == 0:
            log.info('Computing class prorotypes for angle calculation.')

        model = self.trainer.model
        model.eval()

        head = self.trainer._unwrap_model().get_classifier()
        device = head.weight.device
        C, F = head.weight.shape

        b_dim = head.out_features if head.bias is not None else 0
        g_dim = C * F + b_dim

        K = self.trainer.datamodule.get_num_classes()
        assert head.out_features == K, f'Head out_features={head.out_features} != num_classes={K}'

        proto_sum = torch.zeros(K, g_dim, device=device)
        counts = torch.zeros(K, dtype=torch.long, device=device)

        # Hook to capture head input H
        last = {'H': None}

        def _hook(mod, inputs, output):
            last['H'] = inputs[0].detach()

        hnd = head.register_forward_hook(_hook)

        dataloader = self.trainer.get_dataloader(self.dataset_split)

        for i, (X, Y) in enumerate(dataloader):
            if torch.distributed.get_rank() == 0:
                log.info(f' - Processing batch {i}')

            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)

            # Forward only to populate last['H']
            with torch.no_grad():
                _ = model(X)

            H = last['H']  # [B,F], captured by hook
            if H is None:
                raise RuntimeError('Classifier hook did not capture activations (H is None).')

            # Compute per-sample head gradients
            g, _, _ = self._per_sample_head_grads(head, H, Y)  # [B,g_dim]

            # Accumulate class prototypes
            for k in range(K):
                m = Y == k
                if m.any():
                    proto_sum[k] += g[m].sum(0)
                    counts[k] += int(m.sum())

            del H, g
            torch.cuda.empty_cache()

        hnd.remove()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(proto_sum)
            torch.distributed.all_reduce(counts)

        proto_mean = torch.where(
            counts.view(-1, 1) > 0,
            proto_sum / counts.clamp_min(1).view(-1, 1),
            torch.zeros_like(proto_sum),
        )
        proto_unit = proto_mean / proto_mean.norm(dim=1, keepdim=True).clamp_min(1e-12)
        has_bias = head.bias is not None

        return proto_unit, g_dim, has_bias


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
