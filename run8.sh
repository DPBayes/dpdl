#!/bin/bash
#SBATCH --account=project_462000213
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus=8
#SBATCH --time=15:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --error=slurm-%x.%j.out
#SBATCH --output=slurm-%x.%j.stdout

if [ $SLURM_LOCALID -eq 0 ]; then
  rm -rf /dev/shm/*
  rocm-smi || true
else
  sleep 2
fi

module use /appl/local/csc/modulefiles/
module load pytorch
module list

# ResNet illegal memery access fix
export MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW=0

export PROJECT="project_462000213"
export DATA_DIR="/scratch/$PROJECT/data"
export HF_DATASETS_CACHE="$DATA_DIR/huggingface"
export _TYPER_STANDARD_TRACEBACK=1

source /scratch/$PROJECT/venvs/dpdl/bin/activate

set -xv
srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:0 $*
