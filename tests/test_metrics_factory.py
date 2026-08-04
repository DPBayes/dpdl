from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from dpdl.metrics_factory import ClassificationMetrics, MetricsFactory, CustomMetricsFactory


def _example_metric_config() -> Path:
    return Path('conf/metrics/example.conf')


def test_read_metric_config_uses_example_conf() -> None:
    config = CustomMetricsFactory.read_metric_config(str(_example_metric_config()))

    assert [metric['name'] for metric in config['train_metrics']] == ['AUROC']
    assert [metric['name'] for metric in config['valid_metrics']] == ['AUROC']
    assert [metric['name'] for metric in config['test_metrics']] == ['AUROC']


def test_get_metrics_merges_custom_classification_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    configuration = SimpleNamespace(task='ImageClassification', metric_config=str(_example_metric_config()))
    metrics = MetricsFactory.get_metrics(configuration, output_dim=3)

    assert 'AUROC' in metrics['train_metrics'].keys()
    assert 'AUROC' in metrics['valid_metrics'].keys()
    assert 'AUROC' in metrics['test_metrics'].keys()
    assert 'MulticlassAccuracy' in metrics['train_metrics'].keys()
    assert 'ConfusionMatrix' in metrics['test_metrics'].keys()


def test_classification_metrics_update_argmaxes_logits_for_confusion_matrix() -> None:
    metrics = ClassificationMetrics(output_dim=3, sync=False, with_confusion_matrix=True)

    logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.2, 3.0]])
    target = torch.tensor([0, 2])

    metrics.update(logits, target)

    confusion_matrix = metrics['ConfusionMatrix'].compute()
    assert confusion_matrix.sum().item() == 2
    assert confusion_matrix[0, 0].item() == 1
    assert confusion_matrix[2, 2].item() == 1


def test_get_metrics_merges_custom_language_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)

    configuration = SimpleNamespace(task='CausalLM', metric_config=str(_example_metric_config()))
    metrics = MetricsFactory.get_metrics(configuration, output_dim=8)

    assert 'AUROC' in metrics['train_metrics'].keys()
    assert 'AUROC' in metrics['valid_metrics'].keys()
    assert 'AUROC' in metrics['test_metrics'].keys()
    assert 'MulticlassAccuracy' in metrics['train_metrics'].keys()
    assert 'Perplexity' in metrics['test_metrics'].keys()
