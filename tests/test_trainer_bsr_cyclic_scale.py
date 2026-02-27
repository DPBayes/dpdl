import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from dpdl.trainer import DifferentiallyPrivateTrainer


def test_resolve_bsr_cyclic_sensitivity_scale_from_coeffs_and_steps() -> None:
    got = DifferentiallyPrivateTrainer._resolve_bsr_cyclic_sensitivity_scale(
        coeffs=[1.0, 2.0],
        steps=2,
        iterations_number=None,
    )
    assert got == pytest.approx((1.0**2 + 2.0**2) ** 0.5, rel=0.0, abs=1e-12)


def test_resolve_bsr_cyclic_sensitivity_scale_respects_iterations_override() -> None:
    got = DifferentiallyPrivateTrainer._resolve_bsr_cyclic_sensitivity_scale(
        coeffs=[1.0, 2.0],
        steps=4,
        iterations_number=2,
    )
    assert got == pytest.approx((1.0**2 + 2.0**2) ** 0.5, rel=0.0, abs=1e-12)


def test_resolve_bsr_cyclic_sensitivity_scale_handles_missing_inputs() -> None:
    assert DifferentiallyPrivateTrainer._resolve_bsr_cyclic_sensitivity_scale(
        coeffs=[],
        steps=2,
    ) is None
    assert DifferentiallyPrivateTrainer._resolve_bsr_cyclic_sensitivity_scale(
        coeffs=[1.0],
        steps=None,
    ) is None


def test_resolve_bsr_cyclic_sensitivity_scale_rejects_invalid_steps() -> None:
    with pytest.raises(ValueError, match='steps >= 1'):
        DifferentiallyPrivateTrainer._resolve_bsr_cyclic_sensitivity_scale(
            coeffs=[1.0],
            steps=2,
            iterations_number=0,
        )


def test_resolve_bsr_cyclic_sensitivity_scale_rejects_steps_below_bands() -> None:
    with pytest.raises(ValueError, match='steps >= bands'):
        DifferentiallyPrivateTrainer._resolve_bsr_cyclic_sensitivity_scale(
            coeffs=[1.0, 0.5, 0.25],
            steps=2,
            iterations_number=None,
            bands=3,
        )
