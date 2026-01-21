import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')


def _should_skip_gpu_tests() -> bool:
    if os.environ.get('DPDL_RUN_GPU_TESTS') != '1':
        return True
    return not torch.cuda.is_available()


def _run_smoke(cmd: list[str], env: dict, cwd: Path) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.gpu
def test_smoke_train_non_dp(tmp_path: Path) -> None:
    if _should_skip_gpu_tests():
        pytest.skip('GPU smoke tests disabled or CUDA not available.')

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['DPDL_FAKE_DATASET'] = '1'
    env['_TYPER_STANDARD_TRACEBACK'] = '1'

    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc_per_node=1',
        'run.py',
        'train',
        '--device',
        'cuda',
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
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-non-dp',
    ]

    _run_smoke(cmd, env, repo_root)


@pytest.mark.gpu
def test_smoke_train_dp(tmp_path: Path) -> None:
    if _should_skip_gpu_tests():
        pytest.skip('GPU smoke tests disabled or CUDA not available.')

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['DPDL_FAKE_DATASET'] = '1'
    env['_TYPER_STANDARD_TRACEBACK'] = '1'

    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc_per_node=1',
        'run.py',
        'train',
        '--device',
        'cuda',
        '--dataset-name',
        'fake',
        '--model-name',
        'resnet18',
        '--no-pretrained',
        '--use-steps',
        '--total-steps',
        '2',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-dp',
    ]

    _run_smoke(cmd, env, repo_root)
