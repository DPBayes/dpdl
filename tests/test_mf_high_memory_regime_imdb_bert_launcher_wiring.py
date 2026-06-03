from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DPDL_DIR = REPO_ROOT / "dpdl"
LAUNCHER = (
    REPO_ROOT.parent
    / "dpdl-experiments"
    / "experiments"
    / "53-mf-high-memory-regime-imdb-bert"
    / "scripts"
    / "run_and_resume.sh"
)
LR_ONLY_CONFIG = DPDL_DIR / "conf" / "optuna_hypers-mf-high-memory-imdb-bert-lr-only.conf"
LR_BANDS_CONFIG = DPDL_DIR / "conf" / "optuna_hypers-mf-high-memory-imdb-bert-lr-bands.conf"


def _run_launcher(
    *,
    methods: str,
    epsilons: str = "0.25",
    seeds: str = "42",
    extra_env: dict[str, str] | None = None,
    override_log_paths: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "SUBMIT_MODE": "print",
            "METHODS": methods,
            "EPSILONS": epsilons,
            "SEEDS": seeds,
            "DPDL_DIR": str(DPDL_DIR),
        }
    )
    if override_log_paths:
        env.update(
            {
                "LOG_DIR_BASE": "outputs/test-mf-high-memory-regime-imdb-bert",
                "CMD_LOG_DIR_BASE": "experiments/53-mf-high-memory-regime-imdb-bert/data",
            }
        )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _experiment_names(stdout: str) -> list[str]:
    return re.findall(r"--experiment-name\s+([^\s]+)", stdout)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_high_memory_imdb_bert_optuna_configs_are_exact() -> None:
    lr_only = _load_yaml(LR_ONLY_CONFIG)
    lr_bands = _load_yaml(LR_BANDS_CONFIG)

    assert set(lr_only) == {"learning_rate"}
    assert lr_only["learning_rate"] == {
        "min": "1e-5",
        "max": 0.1,
        "type": "float",
        "log_space": True,
    }
    assert set(lr_bands) == {"learning_rate", "bsr_bands"}
    assert lr_bands["learning_rate"] == lr_only["learning_rate"]
    assert lr_bands["bsr_bands"] == {
        "type": "ordered",
        "options": [2, 4, 8, 16, 32, 64],
    }


def test_launcher_emits_standard_opacus_dpsgd_row() -> None:
    proc = _run_launcher(methods="dpsgd")
    stdout = proc.stdout
    stderr = proc.stderr

    assert "MF high-memory IMDb+BERT launcher" in stdout
    assert "TASK=SequenceClassification" in stdout
    assert "DATASET_NAME=stanfordnlp/imdb" in stdout
    assert "MODEL_NAME=bert-base-uncased" in stdout
    assert "Submitting dpsgd eps=0.25 seed=42" in stderr
    assert "run.py optimize" in stdout
    assert "--llm" in stdout
    assert "--task SequenceClassification" in stdout
    assert "--model-name bert-base-uncased" in stdout
    assert "--dataset-name stanfordnlp/imdb" in stdout
    assert "--dataset-text-fields text" in stdout
    assert "--dataset-label-field label" in stdout
    assert "--max-length" not in stdout
    assert "--no-pretrained" not in stdout
    assert "--subset-size 1.0" in stdout
    assert "--epochs 10" in stdout
    assert "--batch-size 512" in stdout
    assert "--physical-batch-size 64" in stdout
    assert "--max-grad-norm 10" in stdout
    assert "--optimizer SGD" in stdout
    assert "--optimizer-momentum 0" in stdout
    assert "--optimizer-weight-decay 0" in stdout
    assert "--target-hypers learning_rate" in stdout
    assert "--target-hypers bsr_bands" not in stdout
    assert "--n-trials 25" in stdout
    assert "--optuna-config conf/optuna_hypers-mf-high-memory-imdb-bert-lr-only.conf" in stdout
    assert "--target-epsilon 0.25" in stdout
    assert "--noise-mechanism gaussian" in stdout
    assert "--accountant prv" in stdout
    assert "--poisson-sampling" in stdout
    assert "--sampling-mode" not in stdout
    assert "--no-poisson-sampling" not in stdout
    assert "--bnb-num-samples" not in stdout
    assert "--bnb-chunk-size" not in stdout


def test_launcher_emits_fixed_p64_bandinvmf_row() -> None:
    stdout = _run_launcher(methods="bandinvmf").stdout

    assert "--llm" in stdout
    assert "--task SequenceClassification" in stdout
    assert "--noise-mechanism bandinvmf" in stdout
    assert "--accountant bnb" in stdout
    assert "--sampling-mode balls_in_bins" in stdout
    assert "--no-poisson-sampling" in stdout
    assert "--bnb-num-samples 500000" in stdout
    assert "--bnb-chunk-size 10000" in stdout
    assert "--bsr-bands 64" in stdout
    assert "--target-hypers learning_rate" in stdout
    assert "--target-hypers bsr_bands" not in stdout
    assert "--n-trials 25" in stdout
    assert "--optuna-config conf/optuna_hypers-mf-high-memory-imdb-bert-lr-only.conf" in stdout
    assert "bandinvmfp64" in stdout
    assert "--bnb-b " not in stdout
    assert "--no-pretrained" not in stdout
    assert "--max-length" not in stdout


def test_launcher_emits_optimized_bandwidth_bsr_row() -> None:
    stdout = _run_launcher(methods="bsr").stdout

    assert "--noise-mechanism bsr" in stdout
    assert "--accountant bnb" in stdout
    assert "--sampling-mode balls_in_bins" in stdout
    assert "--no-poisson-sampling" in stdout
    assert "--target-hypers learning_rate --target-hypers bsr_bands" in stdout
    assert "--bsr-bands" not in stdout
    assert "--n-trials 50" in stdout
    assert "--optuna-config conf/optuna_hypers-mf-high-memory-imdb-bert-lr-bands.conf" in stdout
    assert "bsrpopt" in stdout
    assert "trials50" in stdout
    assert "--no-pretrained" not in stdout
    assert "--max-length" not in stdout


def test_launcher_emits_optimized_bandwidth_bisr_row() -> None:
    stdout = _run_launcher(methods="bisr").stdout

    assert "--noise-mechanism bisr" in stdout
    assert "--target-hypers learning_rate --target-hypers bsr_bands" in stdout
    assert "--bsr-bands" not in stdout
    assert "--n-trials 50" in stdout
    assert "--optuna-config conf/optuna_hypers-mf-high-memory-imdb-bert-lr-bands.conf" in stdout
    assert "bisrpopt" in stdout


def test_launcher_supports_repeat_seed_surface() -> None:
    stdout = _run_launcher(methods="dpsgd", seeds="42 43 44").stdout
    names = _experiment_names(stdout)

    assert len(names) == 3
    assert any("seed42" in name for name in names)
    assert any("seed43" in name for name in names)
    assert any("seed44" in name for name in names)


def test_launcher_experiment_names_are_unique_and_hyper_stamped() -> None:
    stdout = _run_launcher(methods="dpsgd bandinvmf bsr", epsilons="0.25 0.5").stdout
    names = _experiment_names(stdout)

    assert len(names) == 6
    assert len(set(names)) == len(names)
    for name in names:
        assert "imdb-bert-base-uncased" in name
        assert "ep10" in name
        assert "bs512" in name
        assert "pbs64" in name
        assert "clip10" in name
        assert "sub1p0" in name
        assert "seed42" in name
        assert ("eps0p25" in name) or ("eps0p5" in name)
        assert any(token in name for token in ("dpsgd", "bandinvmfp64", "bsrpopt"))
        if "bsrpopt" in name:
            assert "trials50" in name
        else:
            assert "trials25" in name


def test_launcher_skips_completed_rows_from_runtime_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "outputs"
    experiment_name = (
        "53-mf-highmem-imdb-bert-base-uncased-dpsgd-eps0p25-ep10-"
        "bs512-pbs64-clip10-sub1p0-trials25-seed42"
    )
    experiment_dir = log_dir / experiment_name
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "runtime").write_text("0:00:01\n")

    proc = _run_launcher(
        methods="dpsgd",
        extra_env={"LOG_DIR_BASE": str(log_dir)},
    )

    assert f"Skipping completed {experiment_name}" in proc.stderr
    assert "run.py optimize" not in proc.stdout


def test_launcher_skips_rows_already_in_queue() -> None:
    queued = (
        "53-mf-highmem-imdb-bert-base-uncased-dpsgd-eps0p25-ep10-"
        "bs512-pbs64-clip10-sub1p0-trials25-seed42"
    )
    proc = _run_launcher(
        methods="dpsgd",
        extra_env={"SQUEUE_CMD": f"printf '%s\\n' '{queued}'"},
    )

    assert f"Skipping queued {queued}" in proc.stderr
    assert "run.py optimize" not in proc.stdout


def test_launcher_defaults_match_run8_scratch_path_contract(tmp_path: Path) -> None:
    dpdl_dir = tmp_path / "dpdl"
    (dpdl_dir / "conf").mkdir(parents=True)
    (dpdl_dir / "run.py").write_text("# test stub\n")
    (
        dpdl_dir
        / "conf"
        / "optuna_hypers-mf-high-memory-imdb-bert-lr-only.conf"
    ).write_text("learning_rate:\n  min: 1e-5\n  max: 0.1\n")
    (
        dpdl_dir
        / "conf"
        / "optuna_hypers-mf-high-memory-imdb-bert-lr-bands.conf"
    ).write_text("learning_rate:\n  min: 1e-5\n  max: 0.1\n")

    proc = _run_launcher(
        methods="dpsgd",
        extra_env={"DPDL_DIR": str(dpdl_dir)},
        override_log_paths=False,
    )
    stdout = proc.stdout

    host_default = (
        f"{dpdl_dir}/../dpdl-experiments/"
        "experiments/53-mf-high-memory-regime-imdb-bert/data"
    )
    command_default = (
        "/scratch/dpdl-experiments/"
        "experiments/53-mf-high-memory-regime-imdb-bert/data"
    )
    assert f"LOG_DIR_BASE={host_default}" in stdout
    assert f"CMD_LOG_DIR_BASE={command_default}" in stdout
    assert f"--log-dir {command_default}" in stdout
    assert f"--optuna-journal {command_default}/optuna-" in stdout
