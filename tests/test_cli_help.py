import importlib.util
import os
import subprocess
import sys

import pytest


for module in ('torch', 'typer', 'multiprocess'):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f'Missing dependency: {module}', allow_module_level=True)


def test_cli_help() -> None:
    env = dict(**os.environ)
    env['_TYPER_STANDARD_TRACEBACK'] = '1'
    result = subprocess.run(
        [sys.executable, 'run.py', '--help'],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert 'Command to run' in result.stdout
