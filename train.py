import logging
import os
import sys
import torch
import typer

from dpdl.cli import cli

def configure_logger() -> logging.Logger:
    log = logging.getLogger('dpdl')
    log.setLevel(logging.INFO)

    # create a stream handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the new handler
    log.addHandler(handler)

    return log

def main():
    log = configure_logger()

    world_size = os.getenv('WORLD_SIZE')
    local_rank = os.getenv('LOCAL_RANK')

    if world_size is None or local_rank is None:
        log.error("Missing environment variables 'WORLD_SIZE' and 'LOCAL_RANK'. This script should be "
                  "started with the 'torch.distributed.run' module, like this: \n"
                  "python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=<# of GPUS> --rdzv_endpoint=localhost:<PORT> run.py")
        sys.exit(1)

    log.info(f'Initializing {world_size} workers for training.')

    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(local_rank)
    log.info(f'Rank {local_rank} initialized.')

    torch.cuda.set_device(local_rank)

    if torch.distributed.get_rank() == 0:
        log.info(f'All ranks initialized.')

    typer.run(cli)

if __name__ == '__main__':
    main()
