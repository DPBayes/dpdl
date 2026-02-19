from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdl.configurationmanager import Configuration


def test_b_min_sep_sampling_mode_is_disabled() -> None:
    with pytest.raises(
        ValidationError,
        match="b_min_sep sampling is temporarily disabled",
    ):
        Configuration(
            command="train",
            noise_mechanism="bnb",
            accountant="bnb",
            poisson_sampling=False,
            sampling_mode="b_min_sep",
            bnb_b=2,
            bnb_p=0.2,
            bnb_bands=2,
        )
