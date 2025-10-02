#!/bin/bash
#SBATCH --account=project_2003275
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --mem-per-gpu=40G
#SBATCH --error=slurm-%x.%j.out
#SBATCH --output=slurm-%x.%j.stdout

module purge
module load pytorch
module list

export PROJECT="project_2003275"
export DATA_DIR="/scratch/$PROJECT/yuan_temp/data"
export HF_HOME="/scratch/$PROJECT/yuan_temp/data"
export HF_DATASETS_CACHE="$DATA_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$DATA_DIR/huggingface_hub"
export TORCH_HOME="$DATA_DIR/torch"
export _TYPER_STANDARD_TRACEBACK=1


source /scratch/$PROJECT/yuan_temp/venvs/dpdl/bin/activate

echo $HF_HOME

set -xv
srun ./run_wrapper.sh $@