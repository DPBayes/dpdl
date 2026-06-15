import inspect
import logging
from typing import Any, Dict, Optional

import torchmetrics
import yaml

log = logging.getLogger(__name__)

_METRIC_MODULES = (
    torchmetrics,
    getattr(torchmetrics, 'classification', None),
    getattr(torchmetrics, 'text', None),
    getattr(torchmetrics, 'regression', None),
    getattr(torchmetrics, 'image', None),
    getattr(torchmetrics, 'audio', None),
)


class CustomMetricsFactory:
    @staticmethod
    def _resolve_metric_class(metric_name: str):
        for module in _METRIC_MODULES:
            if module and hasattr(module, metric_name):
                return getattr(module, metric_name)
        raise ValueError(f'Metric class "{metric_name}" not found in torchmetrics.')

    @staticmethod
    def _build_metric(metric_spec: Dict[str, Any], default_kwargs: Dict[str, Any], sync_on_compute: bool):
        metric_name = metric_spec['name']
        metric_cls = CustomMetricsFactory._resolve_metric_class(metric_name)

        sig_params = inspect.signature(metric_cls.__init__).parameters
        accepted = {n for n in sig_params if n != 'self'}
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())

        user_params = dict(metric_spec.get('params') or {})
        if not has_var_kwargs:
            unknown = sorted(set(user_params) - accepted)
            if unknown:
                raise ValueError(f'Metric "{metric_name}" does not accept parameter(s): {", ".join(unknown)}.')

        params = {k: v for k, v in default_kwargs.items() if has_var_kwargs or k in accepted}
        params.update(user_params)
        if has_var_kwargs or 'sync_on_compute' in accepted:
            params.setdefault('sync_on_compute', sync_on_compute)

        return metric_cls(**params)

    @staticmethod
    def _normalize_metric_entries(metric_entries, section_name: str, metric_conf: str) -> list[Dict[str, Any]]:
        normalized: list[Dict[str, Any]] = []

        for idx, entry in enumerate(metric_entries):
            if isinstance(entry, str):
                entry = {'name': entry, 'alias': entry, 'params': {}}
            elif not isinstance(entry, dict):
                raise ValueError(
                    f'Metric config file "{metric_conf}" section "{section_name}" entry #{idx + 1} must be a mapping or string.'
                )

            metric_name = entry.get('name')
            if not metric_name:
                raise ValueError(
                    f'Metric config file "{metric_conf}" section "{section_name}" entry #{idx + 1} is missing a metric name.'
                )

            alias = entry.get('alias') or metric_name
            params = entry.get('params') or {}
            if not isinstance(params, dict):
                raise ValueError(
                    f'Metric config file "{metric_conf}" section "{section_name}" entry #{idx + 1} has invalid params.'
                )

            normalized.append({
                'name': metric_name,
                'alias': alias,
                'params': params,
            })

        return normalized

    @staticmethod
    def read_metric_config(metric_conf: Optional[str]) -> Dict[str, list[Dict[str, Any]]]:
        if not metric_conf:
            return {'train_metrics': [], 'valid_metrics': [], 'test_metrics': []}

        with open(metric_conf, 'rb') as fh:
            raw_config = yaml.safe_load(fh) or {}

        if not isinstance(raw_config, dict):
            raise ValueError(f'Metric config file "{metric_conf}" must contain a mapping at the top level.')

        normalized_config: Dict[str, list[Dict[str, Any]]] = {
            'train_metrics': [],
            'valid_metrics': [],
            'test_metrics': [],
        }

        for key in ('train_metrics', 'valid_metrics', 'test_metrics'):
            entries = raw_config.get(key, None)
            if entries is None:
                continue
            if not isinstance(entries, list):
                raise ValueError(
                    f'Metric config file "{metric_conf}" section "{key}" must contain a list of metrics.'
                )

            normalized_config[key].extend(CustomMetricsFactory._normalize_metric_entries(entries, key, metric_conf))

        return normalized_config

    @staticmethod
    def build_metric_collection(metric_specs, default_kwargs: Dict[str, Any], sync_on_compute: bool) -> Dict[str, torchmetrics.Metric]:
        metrics: Dict[str, torchmetrics.Metric] = {}

        for metric_spec in metric_specs:
            alias = metric_spec['alias']
            metrics[alias] = CustomMetricsFactory._build_metric(metric_spec, default_kwargs, sync_on_compute)

        return metrics
