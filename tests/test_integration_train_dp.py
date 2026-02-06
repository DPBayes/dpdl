from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from integration_utils import (
    assert_config_and_hyperparams,
    assert_runtime,
    assert_test_metrics,
    base_env,
    get_expected_loss,
    run_distributed,
)


@pytest.mark.integration
def test_integration_train_dp(tmp_path: Path, image_dataset_path: Path) -> None:
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
        'vit_tiny_patch16_224.augreg_in21k',
        '--no-pretrained',
        '--privacy',
        '--use-steps',
        '--total-steps',
        '2',
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
        '--target-epsilon',
        '8',
        '--max-grad-norm',
        '1.0',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'train-dp',
    ]

    run_distributed(cmd_args, env, repo_root)

    assert_config_and_hyperparams(
        tmp_path / 'train-dp',
        expected_config={
            'command': 'train',
            'device': 'cpu',
            'dataset_name': 'local-image',
            'dataset_path': str(image_dataset_path),
            'model_name': 'vit_tiny_patch16_224.augreg_in21k',
            'privacy': True,
            'use_steps': True,
            'log_dir': str(tmp_path),
            'experiment_name': 'train-dp',
            'seed': 42,
            'split_seed': 42,
        },
        expected_hyperparams={
            'epochs': None,
            'total_steps': 2,
            'batch_size': 4,
            'target_epsilon': 8.0,
            'max_grad_norm': 1.0,
        },
    )

    metrics = assert_test_metrics(tmp_path / 'train-dp')

    expected_loss = get_expected_loss('train_dp')
    assert metrics['loss'] == pytest.approx(expected_loss, rel=0, abs=1e-6)

    assert_runtime(tmp_path / 'train-dp')
