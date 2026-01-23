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
    env['DPDL_FAKE_DATASET'] = '1'
    env['_TYPER_STANDARD_TRACEBACK'] = '1'
    return env
