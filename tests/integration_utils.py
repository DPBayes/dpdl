import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_distributed(cmd_args: list[str], env: dict, cwd: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc_per_node=1',
        *cmd_args,
    ]
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_experiment_config(log_dir: Path) -> tuple[dict, dict]:
    config_path = log_dir / 'configuration.json'
    hyperparams_path = log_dir / 'hyperparameters.json'
    assert config_path.exists(), f'Expected configuration.json at {config_path}'
    assert hyperparams_path.exists(), f'Expected hyperparameters.json at {hyperparams_path}'
    return load_json(config_path), load_json(hyperparams_path)


def _assert_expected(actual: dict, expected: dict, label: str) -> None:
    for key, expected_value in expected.items():
        assert key in actual, f'Missing {label} key: {key}'
        actual_value = actual[key]
        if isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value, rel=0, abs=1e-8), (
                f'Unexpected {label} value for "{key}": {actual_value}'
            )
        else:
            assert actual_value == expected_value, (
                f'Unexpected {label} value for "{key}": {actual_value}'
            )


def assert_config_and_hyperparams(
    log_dir: Path,
    expected_config: dict | None = None,
    expected_hyperparams: dict | None = None,
) -> None:
    config, hyperparams = load_experiment_config(log_dir)

    if expected_config:
        _assert_expected(config, expected_config, 'configuration')

    if expected_hyperparams:
        _assert_expected(hyperparams, expected_hyperparams, 'hyperparameters')


def load_expected_losses() -> dict:
    fixture_path = Path(__file__).parent / 'fixtures' / 'expected_losses.json'
    if not fixture_path.exists():
        pytest.skip('Missing expected losses fixture file.')
    data = load_json(fixture_path)
    if not isinstance(data, dict):
        pytest.skip('Expected losses fixture file is not a JSON object.')
    return data


def get_expected_loss(key: str) -> float:
    data = load_expected_losses()
    if key not in data:
        pytest.skip(f'Missing expected loss for key: {key}')
    return data[key]


def load_expected_hpo_params() -> dict:
    fixture_path = Path(__file__).parent / 'fixtures' / 'expected_hpo_params.json'
    if not fixture_path.exists():
        pytest.skip('Missing expected HPO params fixture file.')
    data = load_json(fixture_path)
    if not isinstance(data, dict):
        pytest.skip('Expected HPO params fixture file is not a JSON object.')
    return data


def get_expected_hpo_params(key: str) -> dict:
    data = load_expected_hpo_params()
    if key not in data:
        pytest.skip(f'Missing expected HPO params for key: {key}')
    params = data[key]
    if not isinstance(params, dict):
        pytest.skip(f'Expected HPO params for key {key} is not a JSON object.')
    return params


def base_env() -> dict:
    env = os.environ.copy()
    env['_TYPER_STANDARD_TRACEBACK'] = '1'
    return env
