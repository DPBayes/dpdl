from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from integration_utils import (
    base_env,
    get_expected_hpo_params,
    get_expected_loss,
    load_json,
    run_distributed,
)


@pytest.mark.integration
def test_integration_hpo_dp(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'optimize',
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
        '8',
        '--max-grad-norm',
        '1.0',
        '--target-hypers',
        'learning_rate',
        '--n-trials',
        '1',
        '--optuna-config',
        'tests/fixtures/optuna_hypers_small.conf',
        '--optuna-sampler',
        'RandomSampler',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'hpo-dp',
        '--optuna-journal',
        str(tmp_path / 'optuna-dp.journal'),
    ]

    run_distributed(cmd_args, env, repo_root)

    metrics_path = tmp_path / 'hpo-dp' / 'hpo_metrics.json'
    assert metrics_path.exists(), 'Expected hpo_metrics.json to be written.'

    metrics = load_json(metrics_path)
    assert isinstance(metrics, list) and metrics, 'Expected non-empty hpo_metrics list.'
    assert 'loss' in metrics[0], 'Expected loss in hpo_metrics entry.'

    expected_loss = get_expected_loss('hpo_dp_trial0')
    assert metrics[0]['loss'] == pytest.approx(expected_loss, rel=0, abs=1e-6)

    best_params_path = tmp_path / 'hpo-dp' / 'best-params.json'
    assert best_params_path.exists(), 'Expected best-params.json to be written.'
    best_params = load_json(best_params_path)

    expected_params = get_expected_hpo_params('hpo_dp_trial0')
    for key, expected_value in expected_params.items():
        assert key in best_params, f'Missing expected hyperparameter: {key}'
        if isinstance(expected_value, float):
            assert best_params[key] == pytest.approx(expected_value, rel=0, abs=1e-8)
        else:
            assert best_params[key] == expected_value

    for artifact in ('best-value', 'best-params.json', 'final-metrics'):
        path = tmp_path / 'hpo-dp' / artifact
        assert path.exists(), f'Expected {artifact} to be written.'
