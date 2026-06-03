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
    / "51-mf-low-memory-regime-imdb-bert"
    / "scripts"
    / "run_and_resume.sh"
)
OPTUNA_CONFIG = (
    DPDL_DIR / "conf" / "optuna_hypers-mf-low-memory-regime-imdb-bert-lr-only.conf"
)


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
                "LOG_DIR_BASE": "outputs/test-mf-low-memory-regime-imdb-bert",
                "CMD_LOG_DIR_BASE": "experiments/51-mf-low-memory-regime-imdb-bert/data",
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


def test_lr_only_config_matches_imdb_bert_low_memory_search_space() -> None:
    config = yaml.safe_load(OPTUNA_CONFIG.read_text())

    assert set(config) == {"learning_rate"}
    assert config["learning_rate"] == {
        "min": "1e-5",
        "max": 0.1,
        "type": "float",
        "log_space": True,
    }


def test_launcher_emits_standard_opacus_dpsgd_row() -> None:
    proc = _run_launcher(methods="dpsgd")
    stdout = proc.stdout
    stderr = proc.stderr

    assert "MF low-memory IMDb+BERT launcher" in stdout
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
    assert "--subset-size 1.0" in stdout
    assert "--epochs 10" in stdout
    assert "--batch-size 512" in stdout
    assert "--physical-batch-size 64" in stdout
    assert "--max-grad-norm 10" in stdout
    assert "--optimizer SGD" in stdout
    assert "--optimizer-momentum 0" in stdout
    assert "--optimizer-weight-decay 0" in stdout
    assert "--target-hypers learning_rate" in stdout
    assert "--n-trials 25" in stdout
    assert "--optuna-config conf/optuna_hypers-mf-low-memory-regime-imdb-bert-lr-only.conf" in stdout
    assert "--target-epsilon 0.25" in stdout
    assert "--noise-mechanism gaussian" in stdout
    assert "--accountant prv" in stdout
    assert "--poisson-sampling" in stdout
    assert "--sampling-mode" not in stdout
    assert "--no-poisson-sampling" not in stdout
    assert "--bnb-num-samples" not in stdout
    assert "--bnb-chunk-size" not in stdout
    assert "--no-pretrained" not in stdout


def test_launcher_emits_blt_bnb_balls_in_bins_row() -> None:
    proc = _run_launcher(methods="blt")
    stdout = proc.stdout
    stderr = proc.stderr

    assert "Submitting blt eps=0.25 seed=42" in stderr
    assert "--llm" in stdout
    assert "--task SequenceClassification" in stdout
    assert "--noise-mechanism blt" in stdout
    assert "--accountant bnb" in stdout
    assert "--sampling-mode balls_in_bins" in stdout
    assert "--no-poisson-sampling" in stdout
    assert "--bnb-num-samples 500000" in stdout
    assert "--bnb-chunk-size 10000" in stdout
    assert "--blt-buffers 4" in stdout
    assert "--poisson-sampling" not in stdout
    assert "--bnb-b " not in stdout
    assert "--no-pretrained" not in stdout


def test_launcher_supports_repeat_seed_surface() -> None:
    stdout = _run_launcher(methods="dpsgd", seeds="42 43 44").stdout
    names = _experiment_names(stdout)

    assert len(names) == 3
    assert any("seed42" in name for name in names)
    assert any("seed43" in name for name in names)
    assert any("seed44" in name for name in names)


def test_launcher_experiment_names_are_unique_and_hyper_stamped() -> None:
    stdout = _run_launcher(methods="dpsgd bsr", epsilons="0.25 0.5").stdout
    names = _experiment_names(stdout)

    assert len(names) == 4
    assert len(set(names)) == len(names)
    for name in names:
        assert "imdb-bert-base-uncased" in name
        assert "ep10" in name
        assert "bs512" in name
        assert "pbs64" in name
        assert "clip10" in name
        assert "sub1p0" in name
        assert "trials25" in name
        assert "seed42" in name
        assert ("eps0p25" in name) or ("eps0p5" in name)
        assert any(token in name for token in ("dpsgd", "bsrp16"))


def test_launcher_skips_completed_rows_from_runtime_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "outputs"
    experiment_name = (
        "51-mf-lowmem-imdb-bert-base-uncased-dpsgd-eps0p25-ep10-"
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
        "51-mf-lowmem-imdb-bert-base-uncased-dpsgd-eps0p25-ep10-"
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
        / "optuna_hypers-mf-low-memory-regime-imdb-bert-lr-only.conf"
    ).write_text("learning_rate:\n  min: 1e-5\n  max: 0.1\n")

    proc = _run_launcher(
        methods="dpsgd",
        extra_env={"DPDL_DIR": str(dpdl_dir)},
        override_log_paths=False,
    )
    stdout = proc.stdout

    host_default = (
        f"{dpdl_dir}/../dpdl-experiments/"
        "experiments/51-mf-low-memory-regime-imdb-bert/data"
    )
    command_default = (
        "/scratch/dpdl-experiments/"
        "experiments/51-mf-low-memory-regime-imdb-bert/data"
    )
    assert f"LOG_DIR_BASE={host_default}" in stdout
    assert f"CMD_LOG_DIR_BASE={command_default}" in stdout
    assert f"--log-dir {command_default}" in stdout
    assert f"--optuna-journal {command_default}/optuna-" in stdout
