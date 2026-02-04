from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import base_env, get_expected_loss, load_json, run_distributed


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

    metrics_path = tmp_path / 'peft-head-only' / 'test_metrics'
    assert metrics_path.exists(), 'Expected test_metrics to be written.'

    metrics = load_json(metrics_path)
    assert 'loss' in metrics, 'Expected loss in test_metrics.'

    expected_loss = get_expected_loss('peft_head_only')
    assert metrics['loss'] == pytest.approx(expected_loss, rel=0, abs=1e-6)

    runtime_path = tmp_path / 'peft-head-only' / 'runtime'
    assert runtime_path.exists(), 'Expected runtime to be written.'
