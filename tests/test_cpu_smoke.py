import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('torch')


def test_cpu_smoke_train(tmp_path: Path) -> None:
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
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'smoke-cpu',
    ]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
