#!/bin/bash

# Distributed settings
export MASTER_PORT=$(expr 30000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export HIP_VISIBLE_DEVICES=$SLURM_LOCALID

python3 "$@"
