#!/bin/bash

show_help() {
    echo "Usage: $0 script_name [options...]"
    echo ""
    echo "script_name               Name of the script to be created."
    echo ""
    echo "Options:"
    echo "  --help                  Show this help message."
    echo "  project                 Slurm project (default: project_462000213)."
    echo "  partition               Slurm partition (default: standard-g)."
    echo "  gpus                    Number of GPUs (default: 8)."
    echo "  time                    Time allocation (default: 1:00:00, 00:15:00 for dev-g)."
    echo "  mem_per_gpu             Memory per GPU (default: 60G)."
    echo "  cpus_per_task           Number of CPUs per task (default: 7)."
    echo ""
    echo "Example:"
    echo "  $0 run.sh project_462000213 small-g 1"
}

# Check for --help option
if [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# First argument is the script name, defaults to "run8.sh" if not provided
script_name=$1
if [[ "$script_name" == "" ]]; then
    show_help
    exit 0
fi

# Wrapper script sets the environment variables after "srun" has been called
wrapper_script="run_wrapper.sh"

project=${2:-"project_462000213"}
partition=${3:-"standard-g"}
gpus=${4:-8}
ntasks_per_node=$gpus
time=${5:-"1:00:00"}
mem_per_gpu=${6:-"60G"}
cpus_per_task=${7:-7}
exclusive="--exclusive"
cpu_bind_mask="0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
nodes=1

srun_args="--account=$project --partition=$partition --nodes=$nodes --ntasks-per-node=$ntasks_per_node --cpus-per-task=$cpus_per_task --gpus=$gpus --time=$time --mem-per-gpu=$mem_per_gpu --threads-per-core=1"

# if we are using all the GPUs, then set GPU binding and reserve the whole node
if [ "$gpus" == "8" ]; then
    srun_args="$srun_args --cpu-bind=mask_cpu:$cpu_bind_mask"
    srun_args="$srun_args --exclusive"
fi

if [ "$partition" == "dev-g" ]; then
    time="00:15:00"
fi

# Create the wrapper script dynamically
cat <<EOF > $wrapper_script
#!/bin/bash

# Distributed settings
export MASTER_PORT=\$(expr 30000 + \$(echo -n \$SLURM_JOBID | tail -c 4))
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=\$SLURM_NPROCS
export LOCAL_RANK=\$SLURM_LOCALID
export RANK=\$SLURM_PROCID
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID

# Finally, run the program
python3 "\$@"
EOF

chmod +x $wrapper_script

# Create the specified main script dynamically
cat <<EOF > $script_name
#!/bin/bash
#SBATCH --account=$project
#SBATCH --partition=$partition
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --gpus=$gpus
#SBATCH --time=$time
#SBATCH --mem-per-gpu=$mem_per_gpu
#SBATCH --threads-per-core=1
#SBATCH --error=slurm-%x.%j.out
#SBATCH --output=slurm-%x.%j.stdout

# Load CSC PyTorch
module use /appl/local/csc/modulefiles/
module load pytorch
module list

# Fix for illegal memory access with convolutional networks
export MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW=0

# Project specific settings
export PROJECT="$project"
export DATA_DIR="/scratch/\$PROJECT/data"
export HF_DATASETS_CACHE="\$DATA_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="\$DATA_DIR/huggingface_hub"
export TORCH_HOME="\$DATA_DIR/torch"
export _TYPER_STANDARD_TRACEBACK=1

# Activate virtual environment
source /scratch/\$PROJECT/venvs/dpdl/bin/activate

# Run the wrapper script with srun
set -xv
srun $srun_args ./$wrapper_script \$@
EOF

# Make the main script executable
chmod +x $script_name

echo "Created scripts: $script_name and $wrapper_script."

