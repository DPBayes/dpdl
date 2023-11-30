import os
import sys
import torch
import typer

from dpdl.cli import cli
from dpdl.logger_config import configure_logger

# reproducible results
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

def main():
    log = configure_logger()

    world_size = os.getenv('WORLD_SIZE')
    rank = os.getenv('RANK')
    local_rank = os.getenv('LOCAL_RANK')

    if world_size is None or local_rank is None or rank is None:
        log.error("Script not correctly started: Environment variables 'WORLD_SIZE', 'RANK', and 'LOCAL_RANK' missing.")
        sys.exit(1)

    world_size = int(world_size)
    local_rank = int(local_rank)
    rank = int(rank)

    log.info(f'Rank {rank} initializing - our world size is {world_size} and local rank is {local_rank}.')

    # Initialize the process group
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    log.info(f'Rank {rank} initialized.')

    if torch.distributed.get_rank() == 0:
        log.info('All ranks initialized.')

    typer.run(cli)

    torch.distributed.destroy_process_group()

    log.info('And we are done!')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('--help')

    main()
