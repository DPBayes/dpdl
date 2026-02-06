from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import (
    assert_config_and_hyperparams,
    assert_runtime,
    assert_test_metrics,
    base_env,
    run_distributed,
)


@pytest.mark.integration
def test_integration_feature_cache(tmp_path: Path, image_dataset_path: Path) -> None:
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
        '--peft',
        'head-only',
        '--cache-features',
        '--use-steps',
        '--total-steps',
        '2',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '1',
        '--seed',
        '42',
        '--split-seed',
        '42',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'feature-cache',
    ]

    result = run_distributed(cmd_args, env, repo_root)

    assert_config_and_hyperparams(
        tmp_path / 'feature-cache',
        expected_config={
            'command': 'train',
            'device': 'cpu',
            'dataset_name': 'local-image',
            'dataset_path': str(image_dataset_path),
            'model_name': 'resnet18',
            'privacy': False,
            'peft': 'head-only',
            'cache_features': True,
            'use_steps': True,
            'num_workers': 1,
            'log_dir': str(tmp_path),
            'experiment_name': 'feature-cache',
            'seed': 42,
            'split_seed': 42,
        },
        expected_hyperparams={
            'epochs': None,
            'total_steps': 2,
            'batch_size': 4,
        },
    )

    assert_test_metrics(tmp_path / 'feature-cache', expected_keys={'MulticlassAccuracy'})
    assert_runtime(tmp_path / 'feature-cache')

    output = f'{result.stdout}\n{result.stderr}'
    assert 'Feature caching enabled, caching features.' in output
    assert 'Feature caching finished.' in output
