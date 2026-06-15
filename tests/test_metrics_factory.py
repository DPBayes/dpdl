from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from dpdl.custom_metrics_factory import CustomMetricsFactory
from dpdl.metrics_factory import MetricsFactory


def _example_metric_conf() -> Path:
    return Path('conf/metrics/example.conf')


def test_read_metric_config_uses_example_conf() -> None:
    config = CustomMetricsFactory.read_metric_config(str(_example_metric_conf()))

    assert [metric['name'] for metric in config['train_metrics']] == ['AUROC']
    assert [metric['alias'] for metric in config['train_metrics']] == ['train_auroc']
    assert [metric['alias'] for metric in config['valid_metrics']] == ['val_auroc']
    assert [metric['alias'] for metric in config['test_metrics']] == ['test_auroc']


def test_get_metrics_merges_custom_classification_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    configuration = SimpleNamespace(task='ImageClassification', metric_conf=str(_example_metric_conf()))
    metrics = MetricsFactory.get_metrics(configuration, output_dim=3)

    assert 'train_auroc' in metrics['train_metrics'].keys()
    assert 'val_auroc' in metrics['valid_metrics'].keys()
    assert 'test_auroc' in metrics['test_metrics'].keys()
    assert 'MulticlassAccuracy' in metrics['train_metrics'].keys()
    assert 'ConfusionMatrix' in metrics['test_metrics'].keys()


def test_get_metrics_merges_custom_language_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    configuration = SimpleNamespace(task='CausalLM', metric_conf=str(_example_metric_conf()))
    metrics = MetricsFactory.get_metrics(configuration, output_dim=8)

    assert 'train_auroc' in metrics['train_metrics'].keys()
    assert 'val_auroc' in metrics['valid_metrics'].keys()
    assert 'test_auroc' in metrics['test_metrics'].keys()
    assert 'MulticlassAccuracy' in metrics['train_metrics'].keys()
    assert 'Perplexity' in metrics['test_metrics'].keys()
