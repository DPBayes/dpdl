from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.trainer import DifferentiallyPrivateTrainer


def test_validate_correlated_mechanism_state_accepts_valid_inputs() -> None:
    DifferentiallyPrivateTrainer._validate_correlated_mechanism_state(
        coeffs=[1.0, 0.5, 0.2],
        z_std=0.03,
        sensitivity_scale=1.2,
    )


def test_validate_correlated_mechanism_state_rejects_empty_coeffs() -> None:
    with pytest.raises(ValueError, match='non-empty coeffs'):
        DifferentiallyPrivateTrainer._validate_correlated_mechanism_state(
            coeffs=[],
            z_std=0.03,
            sensitivity_scale=1.2,
        )


def test_validate_correlated_mechanism_state_rejects_non_finite_coeff() -> None:
    with pytest.raises(ValueError, match='coeffs must be finite'):
        DifferentiallyPrivateTrainer._validate_correlated_mechanism_state(
            coeffs=[1.0, float('nan')],
            z_std=0.03,
            sensitivity_scale=1.2,
        )


def test_validate_correlated_mechanism_state_rejects_tiny_c0() -> None:
    with pytest.raises(ValueError, match='coeffs\\[0\\] > 1e-12'):
        DifferentiallyPrivateTrainer._validate_correlated_mechanism_state(
            coeffs=[1e-15, 0.2],
            z_std=0.03,
            sensitivity_scale=1.2,
        )


def test_validate_correlated_mechanism_state_rejects_bad_z_std() -> None:
    with pytest.raises(ValueError, match='z_std must be finite and >= 0'):
        DifferentiallyPrivateTrainer._validate_correlated_mechanism_state(
            coeffs=[1.0],
            z_std=float('inf'),
            sensitivity_scale=1.2,
        )


def test_validate_correlated_mechanism_state_rejects_bad_sensitivity_scale() -> None:
    with pytest.raises(ValueError, match='sensitivity_scale must be finite and > 0'):
        DifferentiallyPrivateTrainer._validate_correlated_mechanism_state(
            coeffs=[1.0],
            z_std=0.03,
            sensitivity_scale=0.0,
        )
