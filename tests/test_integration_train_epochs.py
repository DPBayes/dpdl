from pathlib import Path
import math

import pytest

pytest.importorskip('torch')

from integration_utils import (
    assert_config_and_hyperparams,
    base_env,
    load_json,
    run_distributed,
)


@pytest.mark.integration
def test_integration_train_epochs(tmp_path: Path, image_dataset_path: Path) -> None:
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
        '--epochs',
        '1',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--seed',
        '42',
        '--split-seed',
        '42',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'train-epochs',
    ]

    run_distributed(cmd_args, env, repo_root)

    assert_config_and_hyperparams(
        tmp_path / 'train-epochs',
        expected_config={
            'command': 'train',
            'device': 'cpu',
            'dataset_name': 'local-image',
            'dataset_path': str(image_dataset_path),
            'model_name': 'resnet18',
            'privacy': False,
            'use_steps': False,
            'log_dir': str(tmp_path),
            'experiment_name': 'train-epochs',
            'seed': 42,
            'split_seed': 42,
        },
        expected_hyperparams={
            'epochs': 1,
            'total_steps': None,
            'batch_size': 4,
        },
    )

    metrics_path = tmp_path / 'train-epochs' / 'test_metrics'
    assert metrics_path.exists(), 'Expected test_metrics to be written.'

    metrics = load_json(metrics_path)
    assert 'loss' in metrics, 'Expected loss in test_metrics.'
    assert 'MulticlassAccuracy' in metrics, 'Expected MulticlassAccuracy in test_metrics.'
    assert math.isfinite(metrics['loss']), 'Expected loss to be finite.'
    assert math.isfinite(metrics['MulticlassAccuracy']), 'Expected accuracy to be finite.'

    runtime_path = tmp_path / 'train-epochs' / 'runtime'
    assert runtime_path.exists(), 'Expected runtime to be written.'
