#!/bin/bash
#SBATCH --account=project_462000213
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --time=1:00:00
#SBATCH --mem=32G

module purge
module load pytorch
module list

export PROJECT="project_462000213"
export DATA_DIR="/scratch/$PROJECT/data"
export HF_DATASETS_CACHE="$DATA_DIR/huggingface"
export PYTHONUSERBASE="$DATA_DIR/python/lightning-opacus"

set -xv
python3 $*
