from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def image_dataset_path() -> Path:
    return Path(__file__).parent / 'fixtures' / 'datasets' / 'image'


@pytest.fixture(scope='session')
def text_dataset_path() -> Path:
    return Path(__file__).parent / 'fixtures' / 'datasets' / 'text'
