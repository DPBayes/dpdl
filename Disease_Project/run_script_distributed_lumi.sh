#!/bin/bash
#SBATCH --account=project_462001244
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH --mem-per-gpu=60G
#SBATCH --threads-per-core=1
#SBATCH --error=slurm-%x.%j.out
#SBATCH --output=slurm-%x.%j.stdout
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# Load CSC PyTorch
module use /appl/local/csc/modulefiles/
module load LUMI partition/G
module load pytorch
module load libjpeg-turbo
module list


# Fix for illegal memory access with convolutional networks
#export MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW=0

# Project specific settings
export PROJECT="project_462001244"
export DATA_DIR="/scratch/$PROJECT/data"
export HF_DATASETS_CACHE="$DATA_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="$DATA_DIR/huggingface_hub"
export TORCH_HOME="$DATA_DIR/torch"
export _TYPER_STANDARD_TRACEBACK=1


export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

##export NCCL_DEBUG=INFO
##export NCCL_DEBUG_SUBSYS=INIT
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_NET_GDR_LEVEL=3

# Activate virtual environment
source /scratch/$PROJECT/venvs/dpdl-llms/bin/activate

# Run the wrapper script with srun
set -xv
##srun --cpu-bind=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000 ./run_wrapper.sh $@
srun ./run_wrapper.sh $@
