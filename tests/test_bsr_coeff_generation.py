from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from dpdl.bsr import generate_bsr_coeffs_from_sgd_workload


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
