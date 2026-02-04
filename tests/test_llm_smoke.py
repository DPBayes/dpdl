import os
import subprocess
import sys
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
