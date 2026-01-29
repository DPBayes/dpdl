# Slurm script with multiple GPU

This is an example SLURM batch script for a single-node, multi-GPU PyTorch job.

## Annotated script

```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT_NAME>              # Billing account / allocation (cluster-specific)
#SBATCH --partition=<PARTITION_NAME>          # Partition/queue to run on (e.g., gpu, debug, etc.)
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --ntasks-per-node=8                   # Number of tasks (MPI ranks) per node; often = number of GPUs
#SBATCH --cpus-per-task=7                     # CPU cores assigned to each task
#SBATCH --gpus=8                              # GPUs requested on the node
#SBATCH --time=00:30:00                       # Wall-clock time limit (HH:MM:SS)
#SBATCH --mem-per-gpu=60G                     # Memory per GPU (cluster-dependent interpretation)
#SBATCH --threads-per-core=1                  # Disable SMT/HyperThreading if supported/desired

#SBATCH --error=slurm-%x.%j.err               # STDERR file: %x=job name, %j=job id
#SBATCH --output=slurm-%x.%j.out              # STDOUT file
#SBATCH --exclusive                           # Reserve the whole node (avoid sharing with other jobs)

#-------------------------------#
# Module environment setup      #
#-------------------------------#

# Many HPC sites provide software through Environment Modules / Lmod.
# "module use" adds an extra module search path (site-specific).
module use <PATH_TO_SITE_MODULEFILES>/

# Load cluster/site-specific environment modules.
# Replace these with what your cluster provides (names vary widely).
module load <CLUSTER_STACK_MODULE>            # e.g., "clustername partition/G" or a compiler/MPI stack
module load <PYTORCH_MODULE>                  # e.g., "pytorch" built for the target GPUs
module load <AUXILIARY_MODULE>                # e.g., libjpeg-turbo if torchvision/image IO needs it

module list                                   # Print loaded modules to logs for reproducibility

#-------------------------------#
# Optional ROCm/MIOpen workaround
#-------------------------------#

# Some AMD ROCm + MIOpen configurations can hit rare "illegal memory access"
# issues for specific convolution algorithms. This env var disables one kernel
# family. Only enable if you see such crashes.
#export MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW=0

#-------------------------------#
# Project/cache configuration    #
#-------------------------------#

# Use a generic project identifier for scratch paths (do not hardcode real IDs).
# You may set these to any directory you have write access to.
export PROJECT_ID="<PROJECT_OR_ALLOCATION_ID>"     # Placeholder; used only to build paths
export DATA_DIR="/scratch/${PROJECT_ID}/data"      # Scratch dataset + cache root (cluster-specific)

# Hugging Face datasets/model caches (offline/online both use these locations).
export HF_DATASETS_CACHE="${DATA_DIR}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${DATA_DIR}/hf_hub"

# PyTorch cache for pretrained weights, etc.
export TORCH_HOME="${DATA_DIR}/torch"

# Typer (CLI framework) setting: show full tracebacks on error (helpful for debugging).
export _TYPER_STANDARD_TRACEBACK=1

# If you want strict offline mode (no network calls), set this.
# Use HF_HUB_OFFLINE=1 as well if you want to force the Hub offline.
export HF_DATASETS_OFFLINE=1
#export HF_HUB_OFFLINE=1

# Python environment activation
source <PATH_TO_YOUR_PROJECT>/.venv/bin/activate


set -xv
srun  --cpu-bind=mask_cpu:<CPU_MASK> ./slurm_run_wrapper.sh "$@"

```


## Minimal runnable script

```bash
#!/bin/bash
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus=8
#SBATCH --time=00:30:00
#SBATCH --mem-per-gpu=60G
#SBATCH --threads-per-core=1
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err
#SBATCH --exclusive

module use <MODULEFILES_PATH>
module load pytorch
module load libjpeg-turbo
module list

export DATA_DIR=<SCRATCH_DATA_DIR>
export HF_DATASETS_CACHE="$DATA_DIR/hf_datasets"
export HUGGINGFACE_HUB_CACHE="$DATA_DIR/hf_hub"
export TORCH_HOME="$DATA_DIR/torch"
export HF_DATASETS_OFFLINE=1

source <PROJECT_PATH>/.venv/bin/activate

srun ./slurm_run_wrapper.sh "$@"
```
