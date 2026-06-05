import importlib.util
import os
import subprocess
import sys

import pytest

def test_custom_model_folder() -> None:
    env = dict(**os.environ)
    env['_TYPER_STANDARD_TRACEBACK'] = '1'

    cmd_args = [
        sys.executable,
        'run.py',
        'show-layers',
        '--model-name',
        'dummy_net',
        '--epochs',
        '1'
    ]
    
    result = subprocess.run(
        cmd_args,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert 'DummyNet' in result.stdout
    assert 'Conv2d' in result.stdout

def test_custom_model_path() -> None:
    env = dict(**os.environ)
    env['_TYPER_STANDARD_TRACEBACK'] = '1'
    
    cmd_args = [
        sys.executable,
        'run.py',
        'show-layers',
        '--model-name',
        './dpdl/models/dummy_net.py',
        '--epochs',
        '1'
    ]
    
    result = subprocess.run(
        cmd_args,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert 'DummyNet' in result.stdout
    assert 'Conv2d' in result.stdout