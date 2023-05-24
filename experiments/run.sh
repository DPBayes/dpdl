#!/bin/bash
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=project_462000213

module purge
module load pytorch
module list

export PROJECT="project_462000213"
export DATA_DIR="/scratch/$PROJECT/data"
export HUGGINGFACE_DATA_DIR="/scratch/$PROJECT/data/huggingface"

set -xv
python3 $*
