import csv
import json
from types import SimpleNamespace

import pytest
import torch

from dpdl.callbacks.optimizer_stats import RecordOptimizerStatsCallback


class _Dataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class _DataModule:
    def __init__(self, size):
        self.loader = SimpleNamespace(dataset=_Dataset(size))

    def get_dataloader(self, split):
        assert split == 'train'
        return self.loader


class _Trainer:
    def __init__(self, model, optimizer, dataset_size=45_000):
        self.model = model
        self.optimizer = optimizer
        self.datamodule = _DataModule(dataset_size)
        self.device = torch.device('cpu')
        self.target_epsilon = 2.0
        self.target_delta = 1e-5

    def _unwrap_model(self):
        return self.model

    def get_epsilon(self):
        return 1.999


def _make_adam(parameters):
    optimizer = torch.optim.Adam(
        parameters,
        lr=0.006,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    optimizer.noise_multiplier = 1.6
    optimizer.expected_batch_size = 512
    optimizer.max_grad_norm = 16
    optimizer.normalize_clipping = True
    optimizer.loss_reduction = 'mean'
    optimizer.secure_mode = False
    return optimizer


def test_records_adam_identity_and_optimizer_metadata(tmp_path, monkeypatch):
    model = torch.nn.Linear(3, 2, dtype=torch.float64)
    optimizer = _make_adam(model.parameters())
    trainer = _Trainer(model, optimizer)
    callback = RecordOptimizerStatsCallback(tmp_path, max_grad_norm=16)
    monkeypatch.setattr(callback, '_distributed_world_size', lambda: 8)
    monkeypatch.setattr(callback, '_all_gather_norms', lambda norms: norms)

    callback.on_train_start(trainer)
    callback.on_train_batch_start(trainer, 0, None)
    optimizer.zero_grad()
    model(torch.ones(4, 3, dtype=torch.float64)).sum().backward()
    for parameter in model.parameters():
        parameter.grad_sample = torch.ones((4, *parameter.shape), dtype=parameter.dtype)
    callback.on_train_physical_batch_end(trainer)
    optimizer.step()
    callback.on_train_batch_end(trainer, 0, None, 0.0)
    callback.on_train_end(trainer)

    with (tmp_path / 'optimizer_stats.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert float(row['noise_multiplier']) == pytest.approx(1.6)
    assert float(row['expected_batch_size_per_worker']) == pytest.approx(512)
    assert float(row['effective_batch_size']) == pytest.approx(4096)
    assert float(row['sample_rate']) == pytest.approx(4096 / 45_000)
    assert float(row['normalized_noise']) == pytest.approx(1.6 / 4096)
    assert float(row['adam_identity_relative_residual']) < 1e-12
    assert float(row['adam_actual_update_rms']) == pytest.approx(float(row['adam_expected_update_rms']), rel=1e-12)

    metadata = json.loads((tmp_path / 'optimizer_metadata.json').read_text())
    assert metadata['optimizer_class'] == 'torch.optim.adam.Adam'
    assert metadata['world_size'] == 8
    assert metadata['effective_batch_size'] == pytest.approx(4096)
    assert metadata['sample_rate'] == pytest.approx(4096 / 45_000)
    assert metadata['normalized_noise'] == pytest.approx(1.6 / 4096)
    assert metadata['completed_optimizer_steps'] == 1
    assert metadata['achieved_epsilon'] == pytest.approx(1.999)
    assert metadata['all_trainable_parameters_covered_once'] is True
    assert metadata['trainable_parameter_count'] == 8
    assert metadata['optimizer_parameter_count'] == 8
    assert {item['name'] for item in metadata['parameter_inventory']} == {'weight', 'bias'}
    assert metadata['adam_identity_tolerance'] == pytest.approx(1e-5)
    assert metadata['adam_identity_max_relative_residual'] < 1e-12
    assert metadata['adam_identity_passed'] is True


def test_metadata_rejects_missing_trainable_parameter(tmp_path):
    model = torch.nn.Linear(3, 2)
    optimizer = _make_adam([model.weight])
    trainer = _Trainer(model, optimizer, dataset_size=100)
    callback = RecordOptimizerStatsCallback(tmp_path, max_grad_norm=1)

    callback.on_train_start(trainer)

    metadata = callback._optimizer_metadata
    assert metadata['all_trainable_parameters_covered_once'] is False
    assert metadata['missing_trainable_parameters'] == ['bias']
    assert metadata['duplicated_trainable_parameters'] == []
    assert metadata['non_trainable_optimizer_parameters'] == []
