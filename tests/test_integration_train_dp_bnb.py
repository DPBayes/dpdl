from pathlib import Path
import os

import pytest

pytest.importorskip('torch')
pytest.importorskip('opacus')

from integration_utils import (
    assert_config_and_hyperparams,
    assert_runtime,
    assert_test_metrics,
    base_env,
    run_distributed,
)


def _run_dp_bnb(
    tmp_path: Path,
    image_dataset_path: Path,
    *,
    experiment: str,
    use_target_epsilon: bool,
    sampling_mode: str = 'balls_in_bins',
) -> dict:
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
        '1',
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
        '--max-grad-norm',
        '1.0',
        '--no-poisson-sampling',
        '--noise-mechanism',
        'bnb',
        '--accountant',
        'bnb',
        '--sampling-mode',
        sampling_mode,
        '--bsr-coeffs',
        '1.0',
        '--bnb-b',
        '2',
        '--bnb-bands',
        '1',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        experiment,
    ]

    if use_target_epsilon:
        cmd_args.extend(['--target-epsilon', '8'])
    else:
        cmd_args.extend(['--noise-multiplier', '10.0'])

    run_distributed(cmd_args, env, repo_root)

    expected_hypers = {
        'epochs': None,
        'total_steps': 1,
        'batch_size': 4,
        'max_grad_norm': 1.0,
    }
    if use_target_epsilon:
        expected_hypers['target_epsilon'] = 8.0
    else:
        expected_hypers['target_epsilon'] = None
        expected_hypers['noise_multiplier'] = 10.0

    expected_config = {
        'command': 'train',
        'device': 'cpu',
        'dataset_name': 'local-image',
        'dataset_path': str(image_dataset_path),
        'privacy': True,
        'use_steps': True,
        'noise_mechanism': 'bnb',
        'accountant': 'bnb',
        'sampling_mode': sampling_mode,
        'poisson_sampling': False,
        'bnb_b': 2,
        'bnb_bands': 1,
    }

    assert_config_and_hyperparams(
        tmp_path / experiment,
        expected_config=expected_config,
        expected_hyperparams=expected_hypers,
    )

    metrics = assert_test_metrics(tmp_path / experiment)
    assert_runtime(tmp_path / experiment)
    return metrics


@pytest.mark.integration
def test_integration_train_dp_bnb_fixed_noise_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bnb-fixed-noise',
        use_target_epsilon=False,
    )
    assert 'loss' in metrics


@pytest.mark.integration
def test_integration_train_dp_bnb_fixed_noise_balls_in_bins_path(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    metrics = _run_dp_bnb(
        tmp_path,
        image_dataset_path,
        experiment='train-dp-bnb-fixed-noise-balls-in-bins',
        use_target_epsilon=False,
        sampling_mode='balls_in_bins',
    )
    assert 'loss' in metrics


# XXX: This is EXTREMELY slow. First of all the MC estimation of sigma
#      is slow and then it completely blows up with target epsilon, since
#      each iteration of binary search reques the slow MC estimation.
#@pytest.mark.integration
#def test_integration_train_dp_bnb_target_epsilon_path(
#    tmp_path: Path, image_dataset_path: Path
#) -> None:
#    metrics = _run_dp_bnb(
#        tmp_path,
#        image_dataset_path,
#        experiment='train-dp-bnb-target-epsilon',
#        use_target_epsilon=True,
#    )
#    assert 'loss' in metrics
