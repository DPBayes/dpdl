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


def test_bsr_cyclic_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=8,
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_bisr_cyclic_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bisr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=8,
    )
    assert cfg.noise_mechanism == 'bisr'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_fixed_batch_bsr_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        bsr_bands=4,
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.sampling_mode == 'torch_sampler'


def test_fixed_batch_bisr_valid_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bisr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        bsr_bands=4,
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bisr'
    assert cfg.sampling_mode == 'torch_sampler'


def test_fixed_batch_bsr_cyclic_is_explicitly_allowed() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bsr',
        poisson_sampling=False,
        sampling_mode='cyclic_poisson',
        bsr_bands=4,
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.sampling_mode == 'cyclic_poisson'


def test_bisr_rejects_wrong_accountant() -> None:
    with pytest.raises(ValidationError, match='BISR mechanism requires --accountant in \\{bnb, bsr\\}'):
        Configuration(
            command='train',
            noise_mechanism='bisr',
            accountant='prv',
            poisson_sampling=False,
            sampling_mode='cyclic_poisson',
            bsr_bands=4,
        )


def test_bisr_rejects_unsupported_sampling_mode() -> None:
    with pytest.raises(ValidationError, match='requires --accountant bnb'):
        Configuration(
            command='train',
            noise_mechanism='bisr',
            accountant='bsr',
            poisson_sampling=False,
            sampling_mode='balls_in_bins',
            bsr_bands=4,
        )


def test_bandmf_valid_balls_in_bins_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bandmf',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
        bsr_coeffs=[1.0, 0.2],
    )
    assert cfg.noise_mechanism == 'bandmf'
    assert cfg.accountant == 'bnb'
    assert cfg.sampling_mode == 'balls_in_bins'


def test_bsr_valid_balls_in_bins_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bsr',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
    )
    assert cfg.noise_mechanism == 'bsr'
    assert cfg.accountant == 'bnb'
    assert cfg.sampling_mode == 'balls_in_bins'


def test_bisr_valid_balls_in_bins_minimal() -> None:
    cfg = Configuration(
        command='train',
        noise_mechanism='bisr',
        accountant='bnb',
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        bnb_b=4,
        bsr_bands=2,
    )
    assert cfg.noise_mechanism == 'bisr'
    assert cfg.accountant == 'bnb'
    assert cfg.sampling_mode == 'balls_in_bins'


def test_balls_in_bins_mf_rejects_missing_bnb_b() -> None:
    with pytest.raises(ValidationError, match='balls_in_bins sampling requires --bnb-b'):
        Configuration(
            command='train',
            noise_mechanism='bsr',
            accountant='bnb',
            poisson_sampling=False,
            sampling_mode='balls_in_bins',
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
