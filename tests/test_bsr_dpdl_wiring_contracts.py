from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dpdl.trainer import DifferentiallyPrivateTrainer


def _build_loader(
    *,
    n_samples: int = 32,
    in_dim: int = 4,
    n_classes: int = 3,
    batch_size: int = 8,
) -> DataLoader:
    gen = torch.Generator().manual_seed(20260219)
    x = torch.randn(n_samples, in_dim, generator=gen)
    y = torch.randint(0, n_classes, size=(n_samples,), generator=gen)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )


class _DummyDataModule:
    def __init__(self, loader: DataLoader):
        self._loader = loader
        self.batch_size = int(loader.batch_size)

    def get_dataloader(self, split: str) -> DataLoader:
        assert split == "train"
        return self._loader

    def set_dataloader(self, split: str, loader: DataLoader) -> None:
        assert split == "train"
        self._loader = loader


@dataclass
class _CallRecord:
    method: str
    kwargs: dict


class _FakePrivacyEngine:
    def __init__(self):
        self.calls: list[_CallRecord] = []
        self.noise_mechanism_config = None
        self.sampling_semantics = None
        self.get_epsilon_calls: list[tuple[float, dict]] = []

    def make_private(self, **kwargs):
        self.calls.append(_CallRecord(method="make_private", kwargs=kwargs))
        self.noise_mechanism_config = kwargs["noise_mechanism_config"]
        self.sampling_semantics = kwargs["sampling_semantics"]
        return kwargs["module"], kwargs["optimizer"], kwargs["data_loader"]

    def make_private_with_epsilon(self, **kwargs):
        self.calls.append(_CallRecord(method="make_private_with_epsilon", kwargs=kwargs))
        self.noise_mechanism_config = kwargs["noise_mechanism_config"]
        self.sampling_semantics = kwargs["sampling_semantics"]
        return kwargs["module"], kwargs["optimizer"], kwargs["data_loader"]

    def get_epsilon(self, delta, **kwargs):
        self.get_epsilon_calls.append((float(delta), dict(kwargs)))
        return 1.2345


def _build_trainer_stub(
    *,
    privacy_engine: _FakePrivacyEngine,
    target_epsilon: float | None,
    noise_multiplier: float | None,
    sampling_mode: str,
    bsr_coeffs: list[float],
    bsr_bands: int,
    total_steps: int,
    bsr_z_std: float | None = None,
    bsr_mf_sensitivity: float | None = None,
    bsr_max_participations: int | None = None,
    bsr_min_separation: int | None = None,
    bsr_iterations_number: int | None = None,
) -> DifferentiallyPrivateTrainer:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = _build_loader()

    trainer = object.__new__(DifferentiallyPrivateTrainer)
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.datamodule = _DummyDataModule(loader)
    trainer.device = torch.device("cpu")

    trainer.noise_multiplier = noise_multiplier
    trainer.max_grad_norm = 1.0
    trainer.clipping_mode = "flat"
    trainer.target_epsilon = target_epsilon
    trainer.target_delta = 1e-5
    trainer.noise_batch_ratio = None
    trainer.seed = 123
    trainer.poisson_sampling = False
    trainer.normalize_clipping = False
    trainer.accountant = "bsr"
    trainer.noise_mechanism = "bsr"
    trainer.sampling_mode = sampling_mode
    trainer.bsr_coeffs = list(bsr_coeffs)
    trainer.bsr_z_std = bsr_z_std
    trainer.bsr_bands = int(bsr_bands)
    trainer.bsr_max_participations = bsr_max_participations
    trainer.bsr_min_separation = bsr_min_separation
    trainer.bsr_mf_sensitivity = bsr_mf_sensitivity
    trainer.bsr_iterations_number = bsr_iterations_number
    trainer.bnb_b = None
    trainer.bnb_p = None
    trainer.bnb_bands = None
    trainer.bnb_num_samples = None
    trainer.bnb_seed = None
    trainer.total_steps = int(total_steps)
    trainer.epochs = None
    trainer.physical_batch_size = int(loader.batch_size)
    trainer.privacy_engine = privacy_engine
    return trainer


def _dump_bsr_contract(*, trace_call: dict, pe_call: dict) -> dict:
    trace_state = trace_call["noise_mechanism_config"].mechanism_state
    pe_state = pe_call["noise_mechanism_config"].mechanism_state
    trace_meta = (
        trace_call["sampling_semantics"].privacy_metadata
        if trace_call["sampling_semantics"] is not None
        else {}
    )
    pe_meta = (
        pe_call["sampling_semantics"].privacy_metadata
        if pe_call["sampling_semantics"] is not None
        else {}
    )
    return {
        "trace_state": dict(trace_state),
        "trace_meta": dict(trace_meta),
        "pe_state": dict(pe_state),
        "pe_meta": dict(pe_meta),
    }


def test_dpdl_bsr_contract_make_private_with_epsilon_cyclic(monkeypatch) -> None:
    fake_pe = _FakePrivacyEngine()
    trainer = _build_trainer_stub(
        privacy_engine=fake_pe,
        target_epsilon=8.0,
        noise_multiplier=None,
        sampling_mode="cyclic_poisson",
        bsr_coeffs=[1.0, 0.4, 0.1],
        bsr_bands=3,
        total_steps=12,
        bsr_iterations_number=12,
    )

    trace_calls: list[dict] = []
    monkeypatch.setattr(
        "dpdl.trainer.opacus.distributed.DifferentiallyPrivateDistributedDataParallel",
        lambda m: m,
    )
    monkeypatch.setattr(
        DifferentiallyPrivateTrainer,
        "_log_bsr_trace",
        staticmethod(lambda **kwargs: trace_calls.append(kwargs)),
    )

    trainer.setup()

    assert len(fake_pe.calls) == 1
    assert fake_pe.calls[0].method == "make_private_with_epsilon"
    assert len(trace_calls) == 1

    pe_kwargs = fake_pe.calls[0].kwargs
    trace_kwargs = trace_calls[0]
    contract = _dump_bsr_contract(trace_call=trace_kwargs, pe_call=pe_kwargs)
    assert contract["trace_state"] == contract["pe_state"]
    assert contract["trace_meta"] == contract["pe_meta"]
    assert "sensitivity_scale" in contract["pe_state"]
    assert "sensitivity_scale" in contract["pe_meta"]
    assert pe_kwargs["noise_mechanism_config"].accounting_mode == "bsr_accountant"
    assert pe_kwargs["sampling_semantics"].sampling_mode == "cyclic_poisson"
    assert "bsr_sensitivity_scale" in pe_kwargs

    # Final epsilon should be delegated with only delta; PrivacyEngine resolves
    # mechanism/sampling state from its stored config.
    eps = trainer.get_epsilon()
    assert eps == pytest.approx(1.2345, rel=0.0, abs=1e-12)
    assert fake_pe.get_epsilon_calls == [(1e-5, {})]


def test_dpdl_bsr_contract_make_private_fixed_batch(monkeypatch) -> None:
    fake_pe = _FakePrivacyEngine()
    trainer = _build_trainer_stub(
        privacy_engine=fake_pe,
        target_epsilon=None,
        noise_multiplier=0.7,
        sampling_mode="torch_sampler",
        bsr_coeffs=[1.0, 0.2],
        bsr_bands=2,
        total_steps=8,
        bsr_z_std=0.02,
        bsr_mf_sensitivity=1.3,
        bsr_max_participations=2,
        bsr_min_separation=2,
    )

    trace_calls: list[dict] = []
    monkeypatch.setattr(
        "dpdl.trainer.opacus.distributed.DifferentiallyPrivateDistributedDataParallel",
        lambda m: m,
    )
    monkeypatch.setattr(
        DifferentiallyPrivateTrainer,
        "_log_bsr_trace",
        staticmethod(lambda **kwargs: trace_calls.append(kwargs)),
    )

    trainer.setup()

    assert len(fake_pe.calls) == 1
    assert fake_pe.calls[0].method == "make_private"
    assert len(trace_calls) == 1

    pe_kwargs = fake_pe.calls[0].kwargs
    trace_kwargs = trace_calls[0]
    contract = _dump_bsr_contract(trace_call=trace_kwargs, pe_call=pe_kwargs)
    assert contract["trace_state"] == contract["pe_state"]
    assert contract["trace_meta"] == contract["pe_meta"]
    assert contract["pe_state"]["mf_sensitivity"] == pytest.approx(1.3, rel=0.0, abs=1e-12)
    assert contract["pe_state"]["z_std"] == pytest.approx(0.02, rel=0.0, abs=1e-12)
    assert pe_kwargs["sampling_semantics"].sampling_mode == "torch_sampler"
