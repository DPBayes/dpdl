import datetime
import os
import sys

import multiprocess
import torch
import typer

from dpdl.cli import cli
from dpdl.logger_config import configure_logger


def setup_torch():
    # Enable TensorFloat-32 for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Reproducible results
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    # Fix Huggingface datasets map to work with multiple proceses.
    torch.set_num_threads(1)

    # Set to spawn so HF datasets map work with distributed
    multiprocess.set_start_method('spawn', force=True)


def main():
    log = configure_logger()
    setup_torch()

    world_size = os.getenv('WORLD_SIZE')
    rank = os.getenv('RANK')
    local_rank = os.getenv('LOCAL_RANK')

    if world_size is None or local_rank is None or rank is None:
        log.error(
            "Script not correctly started: Environment variables 'WORLD_SIZE', 'RANK', and 'LOCAL_RANK' missing."
            "Defaulting to single GPU configuration."
        )
        world_size = '1'
        rank = '0'
        local_rank = '0'

    world_size = int(world_size)
    local_rank = int(local_rank)
    rank = int(rank)

    log.info(
        f'Rank {rank} initializing - our world size is {world_size} and local rank is {local_rank}.'
    )

    # We only have one visible device exposed by `run_wrapper.sh` as recommended by AMD
    torch.cuda.set_device(0)

    # Initialize the process group
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        device_id=torch.device('cuda', 0),  # Only one visible device
    )

    log.info(f'Rank {rank} initialized.')

    if torch.distributed.get_rank() == 0:
        log.info('All ranks initialized.')

    exit_code = 0
    try:
        typer.run(cli)
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 0
    finally:

        try:
            # Make sure all processes are in sync before destroying process group
            torch.distributed.barrier()
        except Exception:
            pass  # If ranks are uneven, don't block shutdown forever

        torch.distributed.destroy_process_group()

        log.info(f'Rank {rank} done!')

    return exit_code


if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) == 1:
        typer.run(cli)   # print help and exit
        sys.exit(0)

    main()
