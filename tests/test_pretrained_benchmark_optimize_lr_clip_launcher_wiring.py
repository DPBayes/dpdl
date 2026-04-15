from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DPDL_DIR = REPO_ROOT / "dpdl"
LAUNCHER = REPO_ROOT / "scripts" / "OPTIMIZE-PRETRAINED-BENCHMARK-LR-CLIP.sh"
_MODULE_PATH = REPO_ROOT / "local-scripts" / "pretrained_benchmark_sigma_calibration.py"
_MANIFEST_PATH = REPO_ROOT / "local-scripts" / "pretrained_benchmark_manifest.py"
_SPEC = importlib.util.spec_from_file_location(
    "pretrained_benchmark_sigma_calibration", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "pretrained_benchmark_manifest", _MANIFEST_PATH
)
assert _MANIFEST_SPEC is not None and _MANIFEST_SPEC.loader is not None
_MANIFEST = importlib.util.module_from_spec(_MANIFEST_SPEC)
sys.modules[_MANIFEST_SPEC.name] = _MANIFEST
_MANIFEST_SPEC.loader.exec_module(_MANIFEST)

iter_rows = _MANIFEST.iter_rows
CalibratedSigmaRow = _MODULE.CalibratedSigmaRow
build_report_payload = _MODULE.build_report_payload


def _write_sigma_report(tmp_path: Path) -> Path:
    calibrated_rows = []
    for row in iter_rows():
        row_dict = {
            "row_id": row.row_id,
            "dataset_name": row.dataset_name,
            "label_field": row.label_field,
            "dataset_size": row.dataset_size,
            "epochs": row.epochs,
            "steps_per_epoch": row.steps_per_epoch,
            "total_steps": row.total_steps,
            "regime": row.regime,
            "method": row.method,
            "epsilon": row.epsilon,
            "delta": row.delta,
            "model_name": row.model_name,
            "optimizer": row.optimizer,
            "pretrained": row.pretrained,
            "batch_size": row.batch_size,
            "physical_batch_size": row.physical_batch_size,
            "max_grad_norm": row.max_grad_norm,
            "bands": row.bands,
            "noise_mechanism": row.noise_mechanism,
            "accountant": row.accountant,
            "sampling_mode": row.sampling_mode,
            "poisson_sampling": row.poisson_sampling,
            "explicit_coeffs": row.explicit_coeffs,
            "calibrated_for": row.calibrated_for,
            "method_policy": row.method_policy,
            "bifr_frac": row.bifr_frac,
            "blt_buffers": row.blt_buffers,
            "blt_selection_mode": row.blt_selection_mode,
            "noise_multiplier": 1.2345,
            "bnb_num_samples": 100000,
            "bnb_calibration_mode": "optimistic",
        }
        calibrated_rows.append(CalibratedSigmaRow(**row_dict))
    report = build_report_payload(calibrated_rows)
    path = tmp_path / "sigma_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _run_launcher(
    *,
    sigma_report: Path,
    datasets: str,
    methods: str,
    regimes: str,
    epsilons: str,
    extra_env: dict[str, str] | None = None,
) -> str:
    env = os.environ.copy()
    env.update(
        {
            "SUBMIT_MODE": "print",
            "DATASETS": datasets,
            "METHODS": methods,
            "REGIMES": regimes,
            "EPSILONS": epsilons,
            "SEED_START": "42",
            "SEED_END": "42",
            "N_TRIALS": "3",
            "LOG_DIR_BASE": str(sigma_report.parent / "logs"),
            "SIGMA_REPORT": str(sigma_report),
        }
    )
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=DPDL_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def test_pretrained_optimize_lr_clip_launcher_targets_learning_rate_and_clip_for_report_row(
    tmp_path: Path,
) -> None:
    sigma_report = _write_sigma_report(tmp_path)

    stdout = _run_launcher(
        sigma_report=sigma_report,
        datasets="uoft-cs/cifar100",
        methods="bisr",
        regimes="amplified",
        epsilons="1",
    )

    assert "Pretrained benchmark LR+clip BO launcher" in stdout
    assert "Submitting dataset=uoft-cs/cifar100 regime=amplified method=bisr epsilon=1" in stdout
    assert "--dataset-label-field fine_label" in stdout
    assert "--total-steps 784" in stdout
    assert "--target-hypers learning_rate" in stdout
    assert "--target-hypers max_grad_norm" in stdout
    assert "--optuna-config conf/optuna_hypers_pretrained_benchmark_lr_clip.conf" in stdout
    assert "--noise-multiplier" in stdout
    assert "--target-epsilon" not in stdout
    assert "--sampling-mode balls_in_bins" in stdout
    assert "--bnb-b 98" in stdout


def test_pretrained_optimize_lr_clip_launcher_emits_true_dpsgd_poisson_prv_row(tmp_path: Path) -> None:
    sigma_report = _write_sigma_report(tmp_path)

    stdout = _run_launcher(
        sigma_report=sigma_report,
        datasets="dpdl-benchmark/cassava",
        methods="dpsgd",
        regimes="nonamplified",
        epsilons="4",
    )

    assert "Submitting dataset=dpdl-benchmark/cassava regime=poissonprv method=dpsgd epsilon=4" in stdout
    assert "--noise-mechanism gaussian" in stdout
    assert "--accountant prv" in stdout
    assert "--poisson-sampling" in stdout
    assert "--target-epsilon 4" in stdout
    assert "--target-hypers learning_rate" in stdout
    assert "--target-hypers max_grad_norm" in stdout
    assert "--noise-multiplier" not in stdout
    assert "--bsr-coeffs" not in stdout
    assert "--bsr-bands" not in stdout


def test_pretrained_optimize_lr_clip_launcher_defaults_cover_full_method_set(tmp_path: Path) -> None:
    sigma_report = _write_sigma_report(tmp_path)

    stdout = _run_launcher(
        sigma_report=sigma_report,
        datasets="uoft-cs/cifar100",
        methods="dpsgd idb1 bsr bisr bandmf bandinvmf bifr blt",
        regimes="amplified nonamplified",
        epsilons="8",
    )

    assert "METHODS=dpsgd idb1 bsr bisr bandmf bandinvmf bifr blt" in stdout
    assert "REGIMES=amplified nonamplified" in stdout
    assert "N_TRIALS=3" in stdout
    assert "OPTUNA_CONFIG=conf/optuna_hypers_pretrained_benchmark_lr_clip.conf" in stdout
    assert "BIFR_POLICY=explicit_single_slice_only_no_auto_frac_search" in stdout
    assert "BLT_POLICY=workload_driven_rank_selection" in stdout


def test_pretrained_optimize_lr_clip_launcher_reconstructs_bifr_and_blt_rows(tmp_path: Path) -> None:
    sigma_report = _write_sigma_report(tmp_path)

    bifr_stdout = _run_launcher(
        sigma_report=sigma_report,
        datasets="uoft-cs/cifar100",
        methods="bifr",
        regimes="nonamplified",
        epsilons="1",
    )
    assert "policy=explicit_single_slice_canonical_frac_0p25_v1" in bifr_stdout
    assert "--noise-mechanism bifr" in bifr_stdout
    assert "--accountant bsr" in bifr_stdout
    assert "--bifr-frac 0.25" in bifr_stdout

    blt_stdout = _run_launcher(
        sigma_report=sigma_report,
        datasets="uoft-cs/cifar100",
        methods="blt",
        regimes="amplified",
        epsilons="1",
    )
    assert "policy=workload_buffers2_implicit_default_v1" in blt_stdout
    assert "blt_selection_mode=implicit_workload_default" in blt_stdout
    assert "--noise-mechanism blt" in blt_stdout
    assert "--accountant bnb" in blt_stdout
    assert "--blt-buffers 2" in blt_stdout
    assert "--bsr-bands" not in blt_stdout
