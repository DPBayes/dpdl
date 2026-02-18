from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import ConfigurationManager


def test_configuration_manager_requires_explicit_privacy_target_path() -> None:
    with pytest.raises(ValueError, match='requires one explicit target path'):
        ConfigurationManager(
            {
                'command': 'train',
                'privacy': True,
                'target_epsilon': None,
                'noise_multiplier': None,
                'noise_batch_ratio': None,
                'epochs': 1,
            }
        )


@pytest.mark.parametrize(
    ("target_epsilon", "noise_multiplier", "noise_batch_ratio"),
    [
        (1.0, None, None),
        (-1.0, None, None),
        (None, 0.8, None),
        (None, None, 0.2),
    ],
)
def test_configuration_manager_accepts_any_explicit_privacy_target_path(
    target_epsilon: float | None,
    noise_multiplier: float | None,
    noise_batch_ratio: float | None,
) -> None:
    manager = ConfigurationManager(
        {
            'command': 'train',
            'privacy': True,
            'target_epsilon': target_epsilon,
            'noise_multiplier': noise_multiplier,
            'noise_batch_ratio': noise_batch_ratio,
            'epochs': 1,
            'batch_size': 4,
            'max_grad_norm': 1.0,
        }
    )
    assert manager.configuration.privacy is True
