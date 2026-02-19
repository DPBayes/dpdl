import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

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

