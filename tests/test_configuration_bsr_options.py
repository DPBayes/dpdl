from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import Configuration


def test_bandmf_cyclic_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bandmf',
        accountant='bandmf',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=8,
    )
    assert cfg.noise_mechanism == 'bandmf'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_bandmf_cyclic_rejects_fixed_batch_knobs() -> None:
    with pytest.raises(ValidationError, match='fixed-batch BSR only'):
        Configuration(
            command='train',
            noise_mechanism='bandmf',
            accountant='bandmf',
            poisson_sampling=False,
            sampling_mode='cyclic_poisson',
            bsr_bands=8,
            bsr_mf_sensitivity=1.0,
        )


def test_cyclic_requires_bandmf_mechanism() -> None:
    with pytest.raises(ValidationError, match='requires --noise-mechanism bandmf'):
        Configuration(
            command='train',
            noise_mechanism='bsr',
            accountant='bsr',
            poisson_sampling=False,
            sampling_mode='cyclic_poisson',
            bsr_bands=8,
        )


def test_fixed_batch_bsr_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.sampling_mode == 'torch_sampler'


def test_fixed_batch_bsr_rejects_cyclic() -> None:
    with pytest.raises(ValidationError, match='requires --noise-mechanism bandmf'):
        Configuration(
            command='train',
            noise_mechanism='bsr',
            accountant='bsr',
            poisson_sampling=False,
            sampling_mode='cyclic_poisson',
            bsr_bands=4,
        )


def test_bnb_valid_balls_in_bins_with_alias() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bnb',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_n_bins',
        bnb_b=4,
        bnb_bands=2,
    )
    assert cfg.sampling_mode == 'balls_in_bins'
    assert cfg.bnb_bands == 2


def test_bnb_rejects_b_min_sep() -> None:
    with pytest.raises(ValidationError, match='temporarily disabled'):
        Configuration(
            command='train',
            noise_mechanism='bnb',
            accountant='bnb',
            poisson_sampling=False,
            sampling_mode='b_min_sep',
            bnb_b=4,
            bnb_bands=2,
        )


def test_bnb_rejects_torch_sampler() -> None:
    with pytest.raises(ValidationError, match='requires --sampling-mode balls_in_bins'):
        Configuration(
            command='train',
            noise_mechanism='bnb',
            accountant='bnb',
            poisson_sampling=False,
            sampling_mode='torch_sampler',
            bnb_b=4,
            bnb_bands=2,
        )


def test_gaussian_rejects_mechanism_specific_accountants() -> None:
    with pytest.raises(ValidationError, match='Gaussian mechanism does not support mechanism-specific accountants'):
        Configuration(
            command='train',
            noise_mechanism='gaussian',
            accountant='bnb',
        )
