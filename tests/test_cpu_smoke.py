from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import assert_runtime, assert_test_metrics, base_env, run_distributed


def test_cpu_smoke_train(tmp_path: Path, image_dataset_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'train',
        '--device',
        'cpu',
        '--dataset-name',
        'local-image',
        '--dataset-path',
        str(image_dataset_path),
        '--model-name',
        'resnet18',
        '--no-pretrained',
        '--no-privacy',
        '--use-steps',
        '--total-steps',
        '2',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-cpu',
    ]

    run_distributed(cmd_args, env, repo_root)

    assert_test_metrics(tmp_path / 'smoke-cpu')
    assert_runtime(tmp_path / 'smoke-cpu')
