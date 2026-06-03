from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import Configuration


def _fourier_kwargs(**overrides):
    kwargs = {
        'command': 'train',
        'clipping_mode': 'fourier',
        'noise_mechanism': 'gaussian',
        'fourier_block_size': 16,
        'fourier_retain_frac': 0.25,
    }
    kwargs.update(overrides)
    return kwargs


def test_fourier_fixed_lowpass_validates_and_resolves_retain_fraction() -> None:
    cfg = Configuration(**_fourier_kwargs())

    assert cfg.clipping_mode == 'fourier'
    assert cfg.noise_mechanism == 'gaussian'
    assert cfg.fourier_layout == 'layer_matrix_columns'
    assert cfg.fourier_transform == 'dct'
    assert cfg.fourier_mode == 'fixed_lowpass'
    assert cfg.fourier_resolved_retain_count == 4
    assert cfg.privacy_claim_valid is False
    assert cfg.reported_epsilon_is_nominal is True
    assert cfg.reported_epsilon_excludes_selection is False
    assert cfg.disable_epsilon_logging is False


def test_fourier_fixed_lowpass_validates_retain_count_and_layout() -> None:
    cfg = Configuration(
        **_fourier_kwargs(
            fourier_layout='parameter_blockwise',
            fourier_retain_frac=None,
            fourier_retain_count=3,
        )
    )

    assert cfg.fourier_layout == 'parameter_blockwise'
    assert cfg.fourier_resolved_retain_count == 3


@pytest.mark.parametrize(
    'kwargs, message',
    [
        ({'fourier_retain_frac': None}, 'requires exactly one'),
        ({'fourier_retain_frac': 0.5, 'fourier_retain_count': 2}, 'requires exactly one'),
        ({'fourier_retain_frac': 0.0}, '0 < frac <= 1'),
        ({'fourier_retain_frac': 1.2}, '0 < frac <= 1'),
        ({'fourier_block_size': 4, 'fourier_retain_frac': None, 'fourier_retain_count': 5}, '1 <= count <= --fourier-block-size'),
        ({'fourier_block_size': 0, 'fourier_retain_frac': None, 'fourier_retain_count': 1}, '--fourier-block-size must be >= 1'),
    ],
)
def test_fourier_rejects_invalid_projection_parameters(kwargs: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Configuration(**_fourier_kwargs(**kwargs))


def test_fourier_rejects_invalid_layout() -> None:
    with pytest.raises(ValidationError, match='fourier_layout'):
        Configuration(**_fourier_kwargs(fourier_layout='fft2'))


def test_fourier_adaptive_mode_requires_explicit_nonclaim_opt_in() -> None:
    with pytest.raises(ValidationError, match='allow-unaccounted-fourier-selection'):
        Configuration(
            **_fourier_kwargs(
                fourier_mode='adaptive_topk_leaky',
                fourier_retain_frac=None,
                fourier_retain_count=4,
            )
        )


def test_fourier_adaptive_mode_marks_selection_exclusion() -> None:
    cfg = Configuration(
        **_fourier_kwargs(
            fourier_mode='adaptive_topk_leaky',
            fourier_retain_frac=None,
            fourier_retain_count=4,
            allow_unaccounted_fourier_selection=True,
        )
    )

    assert cfg.privacy_claim_valid is False
    assert cfg.reported_epsilon_is_nominal is True
    assert cfg.reported_epsilon_excludes_selection is True
    assert cfg.disable_epsilon_logging is False


@pytest.mark.parametrize(
    'mechanism, extra',
    [
        ('gaussian', {}),
        ('bandmf', {'accountant': 'bandmf', 'poisson_sampling': False, 'sampling_mode': 'torch_sampler', 'bsr_bands': 4, 'bsr_coeffs': [1.0, 0.2]}),
        ('bsr', {'accountant': 'bsr', 'poisson_sampling': False, 'sampling_mode': 'torch_sampler', 'bsr_bands': 4, 'bsr_coeffs': [1.0, 0.2]}),
        ('bisr', {'accountant': 'bsr', 'poisson_sampling': False, 'sampling_mode': 'torch_sampler', 'bsr_bands': 4, 'bsr_coeffs': [1.0, 0.2]}),
        ('bandinvmf', {'accountant': 'bsr', 'poisson_sampling': False, 'sampling_mode': 'torch_sampler', 'bsr_bands': 4, 'bsr_coeffs': [1.0, 0.2]}),
        ('bifr', {'accountant': 'bsr', 'poisson_sampling': False, 'sampling_mode': 'torch_sampler', 'bsr_bands': 4, 'bifr_frac': 0.5}),
        ('blt', {'accountant': 'blt', 'poisson_sampling': False, 'sampling_mode': 'torch_sampler', 'blt_buffers': 2}),
    ],
)
def test_fixed_fourier_accepts_supported_base_mechanisms(mechanism: str, extra: dict) -> None:
    cfg = Configuration(**_fourier_kwargs(noise_mechanism=mechanism, **extra))

    assert cfg.clipping_mode == 'fourier'
    assert cfg.noise_mechanism == mechanism
    assert cfg.reported_epsilon_is_nominal is True


def test_old_noise_mechanism_fourier_is_rejected_with_migration_hint() -> None:
    with pytest.raises(ValidationError, match='clipping-mode fourier'):
        Configuration(
            command='train',
            noise_mechanism='fourier',
            fourier_retain_count=4,
        )


def test_gaussian_dpsgd_does_not_require_fourier_options() -> None:
    cfg = Configuration(command='train', noise_mechanism='gaussian', clipping_mode='flat')

    assert cfg.noise_mechanism == 'gaussian'
    assert cfg.fourier_resolved_retain_count is None
    assert cfg.privacy_claim_valid is None
    assert cfg.reported_epsilon_is_nominal is None
    assert cfg.reported_epsilon_excludes_selection is None
    assert cfg.reported_epsilon_excludes_sampling is None
