import os
import sys
from pathlib import Path

import pytest

from integration_utils import (
    assert_final_epsilon,
    assert_runtime,
    assert_test_metrics,
    run_command,
)

torch = pytest.importorskip('torch')


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

    run_command(cmd, env, repo_root)
    log_dir = tmp_path / 'smoke-llm-causal'
    assert_runtime(log_dir)
    assert_test_metrics(log_dir, expected_keys={'Perplexity', 'MulticlassAccuracy'})


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

    run_command(cmd, env, repo_root)
    log_dir = tmp_path / 'smoke-llm-causal-private'
    assert_runtime(log_dir)
    assert_test_metrics(log_dir, expected_keys={'Perplexity', 'MulticlassAccuracy'})
    assert_final_epsilon(log_dir)


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

    run_command(cmd, env, repo_root)
    log_dir = tmp_path / 'smoke-llm-seq'
    assert_runtime(log_dir)
    assert_test_metrics(log_dir, expected_keys={'MulticlassAccuracy'})
