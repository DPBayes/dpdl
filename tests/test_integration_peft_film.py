from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import base_env, get_expected_loss, load_json, run_distributed


@pytest.mark.integration
def test_integration_peft_film(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'train',
        '--device',
        'cpu',
        '--dataset-name',
        'fake',
        '--model-name',
        'vit_tiny_patch16_224.augreg_in21k',
        '--no-pretrained',
        '--no-privacy',
        '--peft',
        'film',
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
        'peft-film',
    ]

    run_distributed(cmd_args, env, repo_root)

    metrics_path = tmp_path / 'peft-film' / 'test_metrics'
    assert metrics_path.exists(), 'Expected test_metrics to be written.'

    metrics = load_json(metrics_path)
    assert 'loss' in metrics, 'Expected loss in test_metrics.'

    expected_loss = get_expected_loss('peft_film')
    assert metrics['loss'] == pytest.approx(expected_loss, rel=0, abs=1e-6)

    runtime_path = tmp_path / 'peft-film' / 'runtime'
    assert runtime_path.exists(), 'Expected runtime to be written.'
