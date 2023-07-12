#!/bin/bash
#SBATCH --account=project_2003275
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:a100:4
#SBATCH --time=06:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --error=slurm-%x.%j.out
#SBATCH --output=slurm-%x.%j.stdout

module purge
module load pytorch
module list

export PROJECT="project_2003275"
export DATA_DIR="/scratch/$PROJECT/aki_temp/data"
export HF_DATASETS_CACHE="$DATA_DIR/huggingface"
#export SING_IMAGE="/scratch/project_2003275/aki_temp/singularity-images/pytorch_2.0.1_csc.sif"
#export CUDA_LAUNCH_BLOCKING=1
export _TYPER_STANDARD_TRACEBACK=1
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

source /scratch/$PROJECT/aki_temp/venvs/dpdl/bin/activate

echo "USING MASTER PORT: $MASTER_PORT" >&2
set -xv
srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --rdzv_endpoint=localhost:$MASTER_PORT $*
