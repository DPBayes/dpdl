from pathlib import Path
import sys

import pytest

pytest.importorskip('torch')
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import Configuration
from dpdl.schedulers import SchedulerFactory, bnb_linear_decay_factor


def test_bnb_linear_decay_factor_matches_paper_piecewise_schedule() -> None:
    total_steps = 2000

    assert bnb_linear_decay_factor(step=0, total_steps=total_steps) == pytest.approx(1.0)
    assert bnb_linear_decay_factor(step=500, total_steps=total_steps) == pytest.approx(1.0)
    assert bnb_linear_decay_factor(step=1000, total_steps=total_steps) == pytest.approx(0.525)
    assert bnb_linear_decay_factor(step=1500, total_steps=total_steps) == pytest.approx(0.05)
    assert bnb_linear_decay_factor(step=1999, total_steps=total_steps) == pytest.approx(0.05)


def test_scheduler_factory_returns_none_for_none_scheduler() -> None:
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([p], lr=0.1)
    cfg = Configuration(command='train', lr_scheduler='none')
    assert SchedulerFactory.get_scheduler(
        configuration=cfg,
        optimizer=optimizer,
        total_steps=100,
    ) is None


def test_scheduler_factory_requires_total_steps_for_bnb_linear_decay() -> None:
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([p], lr=0.1)
    cfg = Configuration(command='train', lr_scheduler='bnb_linear_decay')
    with pytest.raises(ValueError, match='requires a resolved total_steps'):
        SchedulerFactory.get_scheduler(
            configuration=cfg,
            optimizer=optimizer,
            total_steps=None,
        )


def test_bnb_linear_decay_scheduler_updates_learning_rate() -> None:
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([p], lr=1.0)
    cfg = Configuration(command='train', lr_scheduler='bnb_linear_decay')
    scheduler = SchedulerFactory.get_scheduler(
        configuration=cfg,
        optimizer=optimizer,
        total_steps=20,
    )
    assert scheduler is not None

    seen_lrs = []
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        seen_lrs.append(optimizer.param_groups[0]['lr'])

    assert max(seen_lrs) == pytest.approx(1.0)
    assert seen_lrs[-1] == pytest.approx(0.05)
