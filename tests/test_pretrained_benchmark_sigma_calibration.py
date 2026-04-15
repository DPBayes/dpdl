from __future__ import annotations

import json
from pretrained_benchmark_imports import (
    filter_rows,
    iter_rows,
    CalibratedSigmaRow,
    build_report_payload,
    load_report,
    save_report,
)


def test_pretrained_benchmark_manifest_covers_expected_matrix_and_contract() -> None:
    rows = list(iter_rows())

    assert len(rows) == 3 * 2 * 7 * 5
    assert {row.dataset_name for row in rows} == {
        "uoft-cs/cifar100",
        "dpdl-benchmark/sun397",
        "dpdl-benchmark/cassava",
    }
    assert {row.regime for row in rows} == {"amplified", "nonamplified"}
    assert {row.method for row in rows} == {"idb1", "bsr", "bisr", "bandmf", "bandinvmf", "bifr", "blt"}
    assert {row.epsilon for row in rows} == {0.5, 1.0, 2.0, 4.0, 8.0}

    cifar_rows = filter_rows(dataset_name="uoft-cs/cifar100")
    sun_rows = filter_rows(dataset_name="dpdl-benchmark/sun397")
    cassava_rows = filter_rows(dataset_name="dpdl-benchmark/cassava")

    assert {row.label_field for row in cifar_rows} == {"fine_label"}
    assert {row.epochs for row in cifar_rows} == {8}
    assert {row.dataset_size for row in cifar_rows} == {50000}

    assert {row.label_field for row in sun_rows} == {"label"}
    assert {row.epochs for row in sun_rows} == {8}
    assert {row.dataset_size for row in sun_rows} == {76127}

    assert {row.label_field for row in cassava_rows} == {"label"}
    assert {row.epochs for row in cassava_rows} == {32}
    assert {row.dataset_size for row in cassava_rows} == {5656}

    assert {row.optimizer for row in rows} == {"paper-sgd"}
    assert {row.batch_size for row in rows} == {512}
    assert {row.physical_batch_size for row in rows} == {32}
    assert {row.max_grad_norm for row in rows} == {10.0}
    assert {row.bands for row in rows} == {1, 4}
    assert {row.delta for row in rows} == {1e-5}
    assert {row.pretrained for row in rows} == {True}
    assert {row.calibrated_for for row in rows} == {"final_evaluation_round"}

    bifr_rows = filter_rows(method="bifr")
    assert {row.bifr_frac for row in bifr_rows} == {0.25}
    assert {row.method_policy for row in bifr_rows} == {"explicit_single_slice_canonical_frac_0p25_v1"}

    blt_rows = filter_rows(method="blt")
    assert {row.blt_rank for row in blt_rows} == {2}
    assert {row.blt_selection_mode for row in blt_rows} == {"implicit_workload_default"}
    assert {row.method_policy for row in blt_rows} == {"workload_rank2_implicit_default_v1"}


def test_pretrained_benchmark_calibration_report_shape_preserves_row_metadata() -> None:
    row = CalibratedSigmaRow(
        row_id="uoft-cs/cifar100|amplified|bisr|eps1p0",
        dataset_name="uoft-cs/cifar100",
        label_field="fine_label",
        dataset_size=50000,
        epochs=8,
        steps_per_epoch=98,
        total_steps=784,
        regime="amplified",
        method="bisr",
        epsilon=1.0,
        delta=1e-5,
        model_name="vit_tiny_patch16_224.augreg_in21k",
        optimizer="paper-sgd",
        pretrained=True,
        batch_size=512,
        physical_batch_size=32,
        max_grad_norm=10.0,
        bands=5,
        noise_mechanism="bisr",
        accountant="bnb",
        sampling_mode="balls_in_bins",
        poisson_sampling=False,
        explicit_coeffs=None,
        calibrated_for="final_evaluation_round",
        method_policy="explicit_single_slice_canonical_frac_0p25_v1",
        bifr_frac=0.25,
        blt_rank=None,
        blt_selection_mode=None,
        noise_multiplier=4.2,
        bnb_num_samples=100000,
        bnb_calibration_mode="optimistic",
    )

    payload = build_report_payload([row])

    assert payload["metadata"]["delta"] == 1e-5
    assert payload["metadata"]["calibrated_for"] == "final_evaluation_round"
    assert payload["metadata"]["row_count"] == 1

    out_row = payload["rows"][0]
    assert out_row["dataset_name"] == "uoft-cs/cifar100"
    assert out_row["label_field"] == "fine_label"
    assert out_row["dataset_size"] == 50000
    assert out_row["epsilon"] == 1.0
    assert out_row["delta"] == 1e-5
    assert out_row["noise_multiplier"] == 4.2
    assert out_row["calibrated_for"] == "final_evaluation_round"
    assert out_row["method_policy"] == "explicit_single_slice_canonical_frac_0p25_v1"
    assert out_row["bifr_frac"] == 0.25


def test_pretrained_benchmark_save_report_round_trips_rows(tmp_path) -> None:
    row = CalibratedSigmaRow(
        row_id="uoft-cs/cifar100|amplified|bisr|eps1p0",
        dataset_name="uoft-cs/cifar100",
        label_field="fine_label",
        dataset_size=50000,
        epochs=8,
        steps_per_epoch=98,
        total_steps=784,
        regime="amplified",
        method="bisr",
        epsilon=1.0,
        delta=1e-5,
        model_name="vit_tiny_patch16_224.augreg_in21k",
        optimizer="paper-sgd",
        pretrained=True,
        batch_size=512,
        physical_batch_size=32,
        max_grad_norm=10.0,
        bands=5,
        noise_mechanism="bisr",
        accountant="bnb",
        sampling_mode="balls_in_bins",
        poisson_sampling=False,
        explicit_coeffs=None,
        calibrated_for="final_evaluation_round",
        method_policy="workload_rank2_implicit_default_v1",
        bifr_frac=None,
        blt_rank=2,
        blt_selection_mode="implicit_workload_default",
        noise_multiplier=4.2,
        bnb_num_samples=100000,
        bnb_calibration_mode="optimistic",
    )
    out = tmp_path / "report.json"
    payload = save_report(out, [row])
    reloaded = load_report(out)

    assert json.loads(out.read_text()) == payload
    assert reloaded["rows"][0]["row_id"] == row.row_id
    assert reloaded["rows"][0]["noise_multiplier"] == row.noise_multiplier
    assert reloaded["rows"][0]["blt_rank"] == 2
    assert reloaded["rows"][0]["blt_selection_mode"] == "implicit_workload_default"
