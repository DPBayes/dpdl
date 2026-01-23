from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import base_env, load_json, run_distributed


@pytest.mark.integration
def test_integration_train_predict(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'train-predict',
        '--device',
        'cpu',
        '--dataset-name',
        'fake',
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
        '--dataset-split',
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

    preds_path = tmp_path / 'train-predict' / 'predictions_test.json'
    assert preds_path.exists(), 'Expected predictions_test.json to be written.'

    preds = load_json(preds_path)
    assert isinstance(preds, list), 'Expected predictions JSON to be a list.'
    assert len(preds) == 8, 'Expected fake test split size of 8.'

    metrics_path = tmp_path / 'train-predict' / 'predict_metrics.json'
    assert metrics_path.exists(), 'Expected predict_metrics.json to be written.'
