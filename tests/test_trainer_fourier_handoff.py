from pathlib import Path
import sys

import pytest
import torch
import opacus

pytest.importorskip('torch')
pytest.importorskip('opacus')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from opacus.mechanism_contracts import FourierClippingConfig

from dpdl.configurationmanager import ConfigurationManager
from dpdl.trainer import TrainerFactory


def _fourier_cli_params(image_dataset_path: Path, **overrides) -> dict:
    params = {
        'command': 'train',
        'device': 'cpu',
        'dataset_name': 'local-image',
        'dataset_path': str(image_dataset_path),
        'model_name': 'bsr-test-net',
        'privacy': True,
        'use_steps': True,
        'total_steps': 1,
        'batch_size': 4,
        'physical_batch_size': 4,
        'num_workers': 0,
        'seed': 42,
        'split_seed': 42,
        'max_grad_norm': 1.0,
        'poisson_sampling': True,
        'target_epsilon': None,
        'noise_multiplier': 1.0,
        'noise_batch_ratio': None,
        'clipping_mode': 'fourier',
        'noise_mechanism': 'gaussian',
        'accountant': 'prv',
        'sampling_mode': None,
        'fourier_layout': 'layer_matrix_columns',
        'fourier_block_size': 16,
        'fourier_retain_count': 4,
        'pretrained': False,
    }
    params.update(overrides)
    return params


def _patch_distributed(monkeypatch: pytest.MonkeyPatch, *, world_size: int) -> None:
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: world_size)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda: None)
    monkeypatch.setattr(
        opacus.distributed,
        'DifferentiallyPrivateDistributedDataParallel',
        lambda model: model,
    )


def test_trainer_forwards_fourier_clipping_config_separately(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_distributed(monkeypatch, world_size=1)
    captured: dict[str, dict] = {}

    def fake_make_private(self, *args, **kwargs):
        captured['kwargs'] = dict(kwargs)
        self.noise_mechanism_config = kwargs['noise_mechanism_config']
        self.fourier_clipping_config = kwargs['fourier_clipping_config']
        self.sampling_semantics = kwargs.get('sampling_semantics')
        return kwargs['module'], kwargs['optimizer'], kwargs['data_loader']

    monkeypatch.setattr(opacus.PrivacyEngine, 'make_private', fake_make_private)

    cfg_mgr = ConfigurationManager(_fourier_cli_params(image_dataset_path))
    TrainerFactory.get_trainer(cfg_mgr)

    kwargs = captured['kwargs']
    assert kwargs['clipping'] == 'fourier'
    assert kwargs['noise_mechanism_config'] is None
    assert isinstance(kwargs['fourier_clipping_config'], FourierClippingConfig)

    state = kwargs['fourier_clipping_config'].state_dict()
    assert state['layout'] == 'layer_matrix_columns'
    assert state['transform'] == 'dct'
    assert state['mode'] == 'fixed_lowpass'
    assert state['block_size'] == 16
    assert state['retain_count'] == 4
    assert state['privacy_claim_valid'] is False
    assert state['reported_epsilon_is_nominal'] is True
    assert state['reported_epsilon_excludes_selection'] is False


def test_trainer_preserves_base_mechanism_with_fourier_clipping(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_distributed(monkeypatch, world_size=1)
    captured: dict[str, dict] = {}

    def fake_make_private(self, *args, **kwargs):
        captured['kwargs'] = dict(kwargs)
        self.noise_mechanism_config = kwargs['noise_mechanism_config']
        self.fourier_clipping_config = kwargs['fourier_clipping_config']
        self.sampling_semantics = kwargs.get('sampling_semantics')
        return kwargs['module'], kwargs['optimizer'], kwargs['data_loader']

    monkeypatch.setattr(opacus.PrivacyEngine, 'make_private', fake_make_private)

    cfg_mgr = ConfigurationManager(
        _fourier_cli_params(
            image_dataset_path,
            noise_mechanism='bsr',
            accountant='bsr',
            poisson_sampling=False,
            sampling_mode='torch_sampler',
            bsr_bands=4,
            bsr_coeffs=[1.0, 0.2],
        )
    )
    TrainerFactory.get_trainer(cfg_mgr)

    mechanism_config = captured['kwargs']['noise_mechanism_config']
    assert mechanism_config.mechanism == 'bsr'
    assert mechanism_config.accounting_mode == 'bsr_accountant'
    assert captured['kwargs']['fourier_clipping_config'].layout == 'layer_matrix_columns'


def test_trainer_allows_distributed_fixed_fourier_handoff(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_distributed(monkeypatch, world_size=2)
    captured: dict[str, dict] = {}

    def fake_make_private(self, *args, **kwargs):
        captured['kwargs'] = dict(kwargs)
        self.noise_mechanism_config = kwargs['noise_mechanism_config']
        self.fourier_clipping_config = kwargs['fourier_clipping_config']
        self.sampling_semantics = kwargs.get('sampling_semantics')
        return kwargs['module'], kwargs['optimizer'], kwargs['data_loader']

    monkeypatch.setattr(opacus.PrivacyEngine, 'make_private', fake_make_private)

    cfg_mgr = ConfigurationManager(_fourier_cli_params(image_dataset_path))
    TrainerFactory.get_trainer(cfg_mgr)

    assert captured['kwargs']['fourier_clipping_config'].mode == 'fixed_lowpass'


def test_trainer_rejects_distributed_adaptive_fourier_before_opacus_handoff(
    image_dataset_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_distributed(monkeypatch, world_size=2)

    def fail_make_private(self, *args, **kwargs):
        raise AssertionError('DPDL should reject distributed adaptive Fourier before PrivacyEngine handoff')

    monkeypatch.setattr(opacus.PrivacyEngine, 'make_private', fail_make_private)

    cfg_mgr = ConfigurationManager(
        _fourier_cli_params(
            image_dataset_path,
            fourier_mode='adaptive_topk_leaky',
            allow_unaccounted_fourier_selection=True,
        )
    )
    with pytest.raises(ValueError, match='adaptive_topk_leaky'):
        TrainerFactory.get_trainer(cfg_mgr)
