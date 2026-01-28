from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import base_env, load_json, run_distributed


@pytest.mark.integration
def test_integration_hpo_full_batch_size(tmp_path: Path) -> None:
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
        '--seed',
        '42',
        '--split-seed',
        '42',
        '--target-hypers',
        'batch_size',
        '--n-trials',
        '1',
        '--optuna-config',
        'tests/fixtures/optuna_hypers_full_batch.conf',
        '--optuna-sampler',
        'RandomSampler',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'hpo-full-batch',
        '--optuna-journal',
        str(tmp_path / 'optuna-full-batch.journal'),
    ]

    run_distributed(cmd_args, env, repo_root)

    raw_params_path = tmp_path / 'hpo-full-batch' / 'best-params-raw-idx.json'
    assert raw_params_path.exists(), 'Expected best-params-raw-idx.json to be written.'
    raw_params = load_json(raw_params_path)
    assert raw_params.get('batch_size_idx') == 0, 'Expected ordered batch_size to select the -1 sentinel.'

    best_params_path = tmp_path / 'hpo-full-batch' / 'best-params.json'
    assert best_params_path.exists(), 'Expected best-params.json to be written.'

    best_params = load_json(best_params_path)
    assert 'batch_size' in best_params, 'Expected batch_size in best-params.json.'
    assert best_params['batch_size'] == 20, 'Expected full batch size to match dataset size (20).'
