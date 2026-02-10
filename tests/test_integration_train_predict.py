from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import (
    assert_config_and_hyperparams,
    assert_predictions,
    assert_predict_metrics,
    assert_runtime,
    base_env,
    run_distributed,
)


@pytest.mark.integration
def test_integration_train_predict(tmp_path: Path, image_dataset_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'train-predict',
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
        '--predict-dataset-split',
        'test',
        '--seed',
        '42',
        '--split-seed',
        '42',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'train-predict',
    ]

    run_distributed(cmd_args, env, repo_root)

    assert_config_and_hyperparams(
        tmp_path / 'train-predict',
        expected_config={
            'command': 'train-predict',
            'device': 'cpu',
            'dataset_name': 'local-image',
            'dataset_path': str(image_dataset_path),
            'model_name': 'resnet18',
            'privacy': False,
            'use_steps': True,
            'predict_dataset_split': 'test',
            'log_dir': str(tmp_path),
            'experiment_name': 'train-predict',
            'seed': 42,
            'split_seed': 42,
        },
        expected_hyperparams={
            'epochs': None,
            'total_steps': 2,
            'batch_size': 4,
        },
    )

    assert_predictions(tmp_path / 'train-predict', split='test', expected_len=8)
    assert_predict_metrics(tmp_path / 'train-predict')

    assert_runtime(tmp_path / 'train-predict')
