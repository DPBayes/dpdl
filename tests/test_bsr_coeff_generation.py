from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dpdl.bsr import generate_bsr_coeffs_from_sgd_workload


@pytest.mark.parametrize(
    ("bands", "alpha", "beta", "expected"),
    [
        # Fixed-grid JAX-reference values (precomputed from the reference recurrence).
        (
            4,
            0.999,
            0.0,
            [1.0, 0.4995, 0.374250375, 0.3115634371875],
        ),
        (
            4,
            0.999,
            0.95,
            [1.0, 0.9744999999999999, 0.949950375, 0.9263115840625],
        ),
        (
            6,
            0.99,
            0.9,
            [
                1.0,
                0.9450000000000001,
                0.8940375,
                0.8467790625,
                0.802920315234375,
                0.7621833404636719,
            ],
        ),
        (
            5,
            0.9999,
            0.9,
            [1.0, 0.9499500000000001, 0.9036525037500001, 0.8607948235621876, 0.8210916311378281],
        ),
    ],
)
def test_bsr_coeffs_match_jax_reference_grid(
    *,
    bands: int,
    alpha: float,
    beta: float,
    expected: list[float],
) -> None:
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=bands,
        momentum=beta,
        weight_decay=alpha,
    )
    assert coeffs == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_bsr_coeffs_match_dpmfsgd_defaults_first_five() -> None:
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.0,
        weight_decay=0.999,
    )
    expected = [1.0, 0.4995, 0.374250375, 0.3115634371875, 0.27234538953152343]
    for c, e in zip(coeffs, expected):
        assert c == pytest.approx(e, rel=1e-12, abs=1e-12)


def test_bsr_coeffs_no_weight_decay_maps_to_alpha_one() -> None:
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.0,
        weight_decay=0.0,
    )
    # alpha=1, beta=0 -> coeffs are Catalan-style r_i sequence.
    expected = [1.0, 0.5, 0.375, 0.3125]
    for c, e in zip(coeffs, expected):
        assert c == pytest.approx(e, rel=1e-12, abs=1e-12)


def test_bsr_coeffs_alpha_equals_beta_limit() -> None:
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=4,
        momentum=0.9,
        weight_decay=0.9,
    )
    assert coeffs == pytest.approx([1.0, 0.9, 0.81, 0.729], rel=1e-12, abs=1e-12)


def test_bsr_coeffs_near_boundary_alpha_one_beta_close_to_one() -> None:
    coeffs = generate_bsr_coeffs_from_sgd_workload(
        bands=5,
        momentum=0.999,
        weight_decay=1.0,
    )
    expected = [1.0, 0.9995, 0.999000375, 0.9985011246875, 0.9980022487502734]
    assert coeffs == pytest.approx(expected, rel=1e-12, abs=1e-12)
