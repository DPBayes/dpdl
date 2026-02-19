from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import Configuration


def test_configuration_accepts_bnb_linear_decay_lr_scheduler() -> None:
    cfg = Configuration(
        command='train',
        lr_scheduler='bnb_linear_decay',
    )
    assert cfg.lr_scheduler == 'bnb_linear_decay'


def test_configuration_rejects_unknown_lr_scheduler() -> None:
    with pytest.raises(ValidationError):
        Configuration(
            command='train',
            lr_scheduler='unknown_schedule',
        )
