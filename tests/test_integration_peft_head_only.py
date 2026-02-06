from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import (
    assert_config_and_hyperparams,
    assert_runtime,
    assert_test_metrics,
    base_env,
    get_expected_loss,
    run_distributed,
)


@pytest.mark.integration
def test_integration_peft_head_only(tmp_path: Path, image_dataset_path: Path) -> None:
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
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'peft-head-only',
    ]

    run_distributed(cmd_args, env, repo_root)

    assert_config_and_hyperparams(
        tmp_path / 'peft-head-only',
        expected_config={
            'command': 'train',
            'device': 'cpu',
            'dataset_name': 'local-image',
            'dataset_path': str(image_dataset_path),
            'model_name': 'resnet18',
            'privacy': False,
            'peft': 'head-only',
            'use_steps': True,
            'log_dir': str(tmp_path),
            'experiment_name': 'peft-head-only',
            'seed': 42,
            'split_seed': 42,
        },
        expected_hyperparams={
            'epochs': None,
            'total_steps': 2,
            'batch_size': 4,
        },
    )

    metrics = assert_test_metrics(tmp_path / 'peft-head-only')

    expected_loss = get_expected_loss('peft_head_only')
    assert metrics['loss'] == pytest.approx(expected_loss, rel=0, abs=1e-6)

    assert_runtime(tmp_path / 'peft-head-only')
