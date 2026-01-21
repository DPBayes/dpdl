import os
import tempfile

import pytest

torch = pytest.importorskip('torch')


@pytest.fixture(scope='session', autouse=True)
def distributed_process_group():
    if not torch.distributed.is_available():
        pytest.skip('torch.distributed is not available.')

    if torch.distributed.is_initialized():
        yield
        return

    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29501')

    init_file = tempfile.NamedTemporaryFile(delete=False)
    init_file.close()
    init_method = f'file://{init_file.name}'

    try:
        torch.distributed.init_process_group(
            backend='gloo',
            rank=0,
            world_size=1,
            init_method=init_method,
        )
    except Exception as exc:
        os.unlink(init_file.name)
        pytest.skip(f'Could not initialize process group: {exc}')

    yield

    torch.distributed.destroy_process_group()
    if os.path.exists(init_file.name):
        os.unlink(init_file.name)
