import os
import subprocess
import sys
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')


def _run_smoke(cmd: list[str], env: dict, cwd: Path) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _assert_metrics(log_dir: Path, expected_metrics: set[str]) -> None:
    metrics_path = log_dir / 'test_metrics'
    assert metrics_path.exists(), f'Missing metrics file: {metrics_path}'

    with metrics_path.open('r') as fh:
        metrics = json.load(fh)

    assert 'loss' in metrics, 'Missing loss in metrics output.'
    for key in expected_metrics:
        assert key in metrics, f'Missing metric "{key}" in metrics output.'

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), f'Non-finite metric value for "{key}".'


def _assert_artifacts(log_dir: Path, *, expect_epsilon: bool, expected_metrics: set[str]) -> None:
    assert log_dir.exists(), f'Missing experiment directory: {log_dir}'
    assert (log_dir / 'runtime').exists(), f'Missing runtime file: {log_dir / "runtime"}'
    _assert_metrics(log_dir, expected_metrics)

    if expect_epsilon:
        assert (log_dir / 'final_epsilon').exists(), f'Missing final epsilon file: {log_dir / "final_epsilon"}'


@pytest.mark.llm
@pytest.mark.integration
def test_smoke_llm_causal(tmp_path: Path, text_dataset_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['_TYPER_STANDARD_TRACEBACK'] = '1'
    cmd = [
        sys.executable,
        'run.py',
        'train',
        '--llm',
        '--task',
        'CausalLM',
        '--model-name',
        'sshleifer/tiny-gpt2',
        '--dataset-name',
        'local-llm-causal',
        '--dataset-path',
        str(text_dataset_path),
        '--dataset-text-fields',
        'text',
        '--use-steps',
        '--total-steps',
        '2',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--max-length',
        '64',
        '--device',
        'cpu',
        '--no-privacy',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-llm-causal',
    ]

    _run_smoke(cmd, env, repo_root)
    _assert_artifacts(
        tmp_path / 'smoke-llm-causal',
        expect_epsilon=False,
        expected_metrics={'Perplexity', 'MulticlassAccuracy'},
    )


@pytest.mark.llm
@pytest.mark.integration
def test_smoke_llm_causal_private(tmp_path: Path, text_dataset_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['_TYPER_STANDARD_TRACEBACK'] = '1'
    cmd = [
        sys.executable,
        'run.py',
        'train',
        '--llm',
        '--task',
        'CausalLM',
        '--model-name',
        'sshleifer/tiny-gpt2',
        '--dataset-name',
        'local-llm-causal-private',
        '--dataset-path',
        str(text_dataset_path),
        '--dataset-text-fields',
        'text',
        '--use-steps',
        '--total-steps',
        '1',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--max-length',
        '64',
        '--device',
        'cpu',
        '--privacy',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-llm-causal-private',
    ]

    _run_smoke(cmd, env, repo_root)
    _assert_artifacts(
        tmp_path / 'smoke-llm-causal-private',
        expect_epsilon=True,
        expected_metrics={'Perplexity', 'MulticlassAccuracy'},
    )


@pytest.mark.llm
@pytest.mark.integration
@pytest.mark.gpu
def test_smoke_llm_sequence_classification(tmp_path: Path, text_dataset_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available.')

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['_TYPER_STANDARD_TRACEBACK'] = '1'
    cmd = [
        sys.executable,
        'run.py',
        'train',
        '--llm',
        '--task',
        'SequenceClassification',
        '--model-name',
        'sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english',
        '--dataset-name',
        'local-llm-seq',
        '--dataset-path',
        str(text_dataset_path),
        '--dataset-text-fields',
        'text',
        '--dataset-label-field',
        'label',
        '--use-steps',
        '--total-steps',
        '2',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--max-length',
        '64',
        '--device',
        'cuda',
        '--no-privacy',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-llm-seq',
    ]

    _run_smoke(cmd, env, repo_root)
    _assert_artifacts(
        tmp_path / 'smoke-llm-seq',
        expect_epsilon=False,
        expected_metrics={'MulticlassAccuracy'},
    )
