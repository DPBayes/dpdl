from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from integration_utils import base_env, load_json, run_distributed


def _run_dp(tmp_path: Path, experiment: str, epsilon: float) -> float:
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
        str(epsilon),
        '--max-grad-norm',
        '1.0',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        experiment,
    ]

    run_distributed(cmd_args, env, repo_root)

    metrics_path = tmp_path / experiment / 'test_metrics'
    assert metrics_path.exists(), 'Expected test_metrics to be written.'

    metrics = load_json(metrics_path)
    assert 'loss' in metrics, 'Expected loss in test_metrics.'
    return metrics['loss']


@pytest.mark.integration
def test_dp_lower_epsilon_higher_loss(tmp_path: Path) -> None:
    # Higher epsilon (weaker privacy) should not yield worse loss than lower epsilon.
    loss_high_eps = _run_dp(tmp_path, 'dp-eps-8', epsilon=8)
    loss_low_eps = _run_dp(tmp_path, 'dp-eps-2', epsilon=2)

    assert loss_low_eps >= loss_high_eps - 1e-6
