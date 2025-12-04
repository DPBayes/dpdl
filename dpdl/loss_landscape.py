import logging
from collections.abc import Mapping
from pathlib import Path

import torch

try:
    from src_lla import viz_lla
    from src_lla.loss_landscapes.metrics.metric import Metric
except ImportError as exc:  # pragma: no cover - dependency is external
    raise ImportError(
        'loss-landscape-analysis is required for visualization. '
        'Install it from https://github.com/GabdullinN/loss-landscape-analysis.'
    ) from exc

from .configurationmanager import ConfigurationManager
from .trainer import Trainer, TrainerFactory

log = logging.getLogger(__name__)


class _AdapterAwareLoss(Metric):
    """
    Minimal metric wrapper that works with both tensor and dict inputs.
    """

    def __init__(self, loss_fn, batch, device: torch.device):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs, self.targets = batch
        self.device = device

    def __call__(self, model_wrapper) -> float:
        x = self._move_inputs(self.inputs)
        y = self.targets.to(device=self.device)

        preds = model_wrapper.forward(x)
        loss = self.loss_fn(preds, y)
        return float(loss.item())

    def _move_inputs(self, inputs):
        if isinstance(inputs, Mapping):
            return {k: v.to(device=self.device, non_blocking=True) for k, v in inputs.items()}
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device=self.device, non_blocking=True)
        return inputs


class LossLandscapeVisualizer:
    def __init__(
        self,
        trainer: Trainer,
        dataset_split: str,
        config_manager: ConfigurationManager,
    ):
        self.trainer = trainer
        self.dataset_split = dataset_split
        self.config_manager = config_manager

    def visualize(self):
        if _is_distributed() and torch.distributed.get_rank() != 0:
            log.info('Skipping loss landscape visualization on non-zero rank.')
            return

        model = self.trainer._unwrap_model()
        model.eval()

        batch = self._get_single_batch()
        device = next(model.parameters()).device
        device_str = device.type

        metric = _AdapterAwareLoss(
            loss_fn=model.criterion,
            batch=batch,
            device=device,
        )

        cfg = self.config_manager.configuration
        cur_name = cfg.experiment_name
        if cfg.visualization_name_suffix:
            cur_name = f'{cur_name}_{cfg.visualization_name_suffix}'

        normalization = cfg.visualization_normalization
        if normalization and normalization.lower() == 'none':
            normalization = None

        viz_dir = self._resolve_output_dir(cfg.visualization_dir)
        res_dir = self._resolve_output_dir(cfg.visualization_res_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)

        viz_lla(
            model=model,
            metric=metric,
            device=device_str,
            dist=cfg.visualization_distance,
            steps=cfg.visualization_steps,
            num_plots=cfg.visualization_num_plots,
            num_per_plot=cfg.visualization_num_per_plot,
            axes=cfg.visualization_axes,
            normalization=normalization,
            order=cfg.visualization_order,
            cur_name=cur_name,
            mode=cfg.visualization_mode,
            b_sqrt=cfg.visualization_b_sqrt,
            viz_dev=cfg.visualization_viz_dev,
            cap_loss=cfg.visualization_cap_loss,
            raa=cfg.visualization_freeze_layer,
            viz_dir=str(viz_dir),
            eval_hessian=cfg.visualization_eval_hessian,
            optimizer=self.trainer.optimizer if cfg.visualization_axes == 'adam' else None,
            res_dir=str(res_dir),
            calc_crit=cfg.visualization_calc_crit,
            n_kh=cfg.visualization_khn_power,
        )

    def _get_single_batch(self):
        dataloader = self.trainer.get_dataloader(self.dataset_split)
        if dataloader is None:
            raise RuntimeError(f'No dataloader found for split "{self.dataset_split}".')

        for batch in dataloader:
            return self.trainer.adapter.move_to_device(*batch)

        raise RuntimeError(f'Dataloader for split "{self.dataset_split}" is empty.')

    def _resolve_output_dir(self, configured: str) -> Path:
        cfg = self.config_manager.configuration
        out = Path(configured)
        if not out.is_absolute():
            out = Path(cfg.log_dir) / cfg.experiment_name / out
        return out


class LossLandscapeVisualizerFactory:
    @staticmethod
    def get_visualizer(config_manager: ConfigurationManager) -> LossLandscapeVisualizer:
        configuration = config_manager.configuration
        trainer = TrainerFactory.get_trainer(config_manager)

        visualizer = LossLandscapeVisualizer(
            trainer=trainer,
            dataset_split=configuration.visualization_dataset_split or configuration.dataset_split,
            config_manager=config_manager,
        )

        if fpath := getattr(configuration, 'model_weights_path', False):
            trainer._unwrap_model().load_model(fpath)

        return visualizer


def _is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()
