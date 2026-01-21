from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from dpdl.utils import safe_open, tensor_to_python_type


def test_tensor_to_python_type() -> None:
    data = {
        'a': torch.tensor(3),
        'b': [torch.tensor([1, 2]), torch.tensor(4)],
    }
    converted = tensor_to_python_type(data)

    assert converted == {'a': 3, 'b': [[1, 2], 4]}


def test_safe_open_writes_file(tmp_path: Path) -> None:
    target = tmp_path / 'out.txt'
    with safe_open(target, 'w') as fh:
        fh.write('ok\n')

    assert target.read_text() == 'ok\n'
