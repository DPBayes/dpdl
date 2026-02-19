import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from dpdl.bsr import generate_bsr_coeffs_from_sgd_workload
from dpdl.trainer import DifferentiallyPrivateTrainer


def test_resolve_or_generate_bsr_coeffs_prefers_explicit_for_bsr() -> None:
    got = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=[1.0, 0.25],
        bsr_bands=8,
        bnb_bands=None,
        bsr_alpha=1.0,
        bsr_beta=0.95,
    )
    assert got == [1.0, 0.25]


def test_resolve_or_generate_bsr_coeffs_prefers_explicit_for_bnb() -> None:
    got = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bnb',
        explicit_coeffs=[1.0, 0.2, 0.1],
        bsr_bands=None,
        bnb_bands=16,
        bsr_alpha=1.0,
        bsr_beta=0.95,
    )
    assert got == [1.0, 0.2, 0.1]


def test_resolve_or_generate_bsr_coeffs_autogenerates_from_bsr_bands() -> None:
    got = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=None,
        bsr_bands=4,
        bnb_bands=None,
        bsr_alpha=1.0,
        bsr_beta=0.95,
    )
    assert len(got) == 4
    assert got[0] == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_resolve_or_generate_bsr_coeffs_autogenerates_from_bnb_bands() -> None:
    got = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bnb',
        explicit_coeffs=None,
        bsr_bands=None,
        bnb_bands=3,
        bsr_alpha=1.0,
        bsr_beta=0.95,
    )
    assert len(got) == 3
    assert got[0] == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_resolve_or_generate_bsr_coeffs_requires_bands_without_explicit() -> None:
    with pytest.raises(ValueError, match='auto coefficient generation requires'):
        DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
            noise_mechanism='bsr',
            explicit_coeffs=None,
            bsr_bands=None,
            bnb_bands=None,
            bsr_alpha=1.0,
            bsr_beta=0.95,
        )


def test_resolve_or_generate_bsr_coeffs_uses_default_alpha_beta_when_missing() -> None:
    got = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=None,
        bsr_bands=4,
        bnb_bands=None,
        bsr_alpha=None,
        bsr_beta=None,
    )
    expected = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.0,
        weight_decay=1.0,
    )
    assert got == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_resolve_or_generate_bsr_coeffs_alpha_beta_change_generated_coeffs() -> None:
    got_default = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=None,
        bsr_bands=4,
        bnb_bands=None,
        bsr_alpha=None,
        bsr_beta=None,
    )
    got_custom = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=None,
        bsr_bands=4,
        bnb_bands=None,
        bsr_alpha=0.99,
        bsr_beta=0.9,
    )
    assert got_custom != got_default


def test_resolve_or_generate_bsr_coeffs_explicit_wins_over_alpha_beta_and_bands() -> None:
    got = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=[1.0, 0.01, 0.001],
        bsr_bands=100,
        bnb_bands=None,
        bsr_alpha=0.99,
        bsr_beta=0.9,
    )
    assert got == [1.0, 0.01, 0.001]


def test_resolve_or_generate_bsr_coeffs_bsr_fixed_and_cyclic_share_generation_path() -> None:
    # Fixed-batch and cyclic BSR flows both resolve coeffs through this shared helper.
    fixed_batch = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=None,
        bsr_bands=8,
        bnb_bands=999,
        bsr_alpha=0.997,
        bsr_beta=0.91,
    )
    cyclic = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=None,
        bsr_bands=8,
        bnb_bands=1,
        bsr_alpha=0.997,
        bsr_beta=0.91,
    )
    assert fixed_batch == pytest.approx(cyclic, rel=0.0, abs=1e-12)


def test_resolve_or_generate_bsr_coeffs_bsr_fixed_and_cyclic_share_explicit_override() -> None:
    # Explicit coeffs must dominate regardless of whether caller is fixed-batch or cyclic.
    fixed_batch = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=[1.0, 0.3, 0.2],
        bsr_bands=8,
        bnb_bands=None,
        bsr_alpha=0.997,
        bsr_beta=0.91,
    )
    cyclic = DifferentiallyPrivateTrainer._resolve_or_generate_bsr_coeffs(
        noise_mechanism='bsr',
        explicit_coeffs=[1.0, 0.3, 0.2],
        bsr_bands=64,
        bnb_bands=None,
        bsr_alpha=1.0,
        bsr_beta=0.0,
    )
    assert fixed_batch == [1.0, 0.3, 0.2]
    assert cyclic == [1.0, 0.3, 0.2]
