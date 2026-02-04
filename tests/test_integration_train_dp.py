from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from integration_utils import base_env, get_expected_loss, load_json, run_distributed


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

    metrics_path = tmp_path / 'train-dp' / 'test_metrics'
    assert metrics_path.exists(), 'Expected test_metrics to be written.'

    metrics = load_json(metrics_path)
    assert 'loss' in metrics, 'Expected loss in test_metrics.'

    expected_loss = get_expected_loss('train_dp')
    assert metrics['loss'] == pytest.approx(expected_loss, rel=0, abs=1e-6)

    runtime_path = tmp_path / 'train-dp' / 'runtime'
    assert runtime_path.exists(), 'Expected runtime to be written.'
