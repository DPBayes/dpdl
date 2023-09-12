import os
import sys
import torch
import typer

from dpdl.cli import cli
from dpdl.logger_config import configure_logger

def main():
    log = configure_logger()

    world_size = os.getenv('WORLD_SIZE')
    local_rank = os.getenv('LOCAL_RANK')

    if world_size is None or local_rank is None:
        log.error("Missing environment variables 'WORLD_SIZE' and 'LOCAL_RANK'. This script should be "
                  "started with the 'torch.distributed.run' module, like this: \n"
                  "python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=<# of GPUS> --rdzv_endpoint=localhost:<PORT> run.py")
        sys.exit(1)

    log.info(f'Initializing worker for training.')

    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(local_rank)
    log.info(f'Rank {local_rank} initialized.')

    torch.cuda.set_device(local_rank)

    if torch.distributed.get_rank() == 0:
        log.info(f'All ranks initialized.')

    typer.run(cli)

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('--help')

    main()
