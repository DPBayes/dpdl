#!/bin/bash
############################################################
### Experiment: Noise-batch ratio initial testing
############################################################

set -euo pipefail

# Base configurations
EXPERIMENT_BASE="29-noise-batch-ratio"
LOG_DIR="/scratch/$PROJECT/experiments/$EXPERIMENT_BASE/data_cifar100"
mkdir -p $LOG_DIR

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k")
DATASETS=(
    "cifar100"
)

# Hypers from experiment 25 HPO
# {"learning_rate": 0.006796578090758156, "batch_size": 45000, "max_grad_norm": 0.00010000000000000009}
#
# Scale learning rate down using Adam's square root law
# In [4]: math.sqrt(45000/192) * 0.006796578090758156
# Out[4]: 0.10405092699585614
LEARNING_RATE="0.006796578090758156"
BATCH_SIZE="192"
MAX_GRAD_NORM="1e-5"
SEEDS="42"

# (50_000/192)*40 = 10416.66...
STEPS=10417

# ' '.join(map(str, [2**-x for x in range(6, 13)]))
NOISE_BATCH_RATIOS="0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625"

declare -A DATASET_LABEL_FIELDS
DATASET_LABEL_FIELDS=(
    ["cifar100"]="fine_label"
)

declare -A SUBSET_SIZES
SUBSET_SIZES=(
    ["cifar100"]="1.0"
)

# Other settings
OVERWRITE_EXPERIMENT="--overwrite-experiment" # These are fast to run, overwrite by default
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
PEFT="--peft film"
RECORD_LOSS="--record-loss-by-step --record-loss-by-epoch"
EVALUATION_MODE="--evaluation-mode"
CHECKPOINT="--checkpoint-step-interval 100"
DISABLE_EPSILON_LOGGING="--disable-epsilon-logging"

# Function to check if a job is already in queue
function is_job_in_queue() {
    local experiment_name=$1
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0  # Job is in queue
    else
        return 1  # Job is not in queue
    fi
}

for seed in $SEEDS
do
    # Loop over configurations
    for model in "${MODELS[@]}"
    do
        for dataset in "${DATASETS[@]}"
        do
            # Get the label field name for this dataset
            label_field=${DATASET_LABEL_FIELDS[$dataset]}

            # Get the configured subset size for this dataset
            subset_size=${SUBSET_SIZES[$dataset]}

            # Remove possible prefix from the dataset name
            clean_dataset_name=${dataset#dpdl-benchmark/}

            for noise_batch_ratio in $NOISE_BATCH_RATIOS
            do
                EXPERIMENT_NAME="${model}_${clean_dataset_name}_Subset${subset_size}_NoiseBatchRatio${noise_batch_ratio}_Seed${seed}"

                EXPERIMENT_DIR="$LOG_DIR/$EXPERIMENT_NAME"
                mkdir -p "$EXPERIMENT_DIR"

                # If `runtime` file exists, then the job has been completed
                if [ -f "$EXPERIMENT_DIR/runtime" ]; then
                    echo "Experiment $EXPERIMENT_NAME has completed. Skipping submission."
                    continue
                fi

                if is_job_in_queue $EXPERIMENT_NAME; then
                    echo "Experiment $EXPERIMENT_NAME is already in the queue."
                    continue  # Skip to the next iteration
                fi

                # Submit the job
                sbatch -J $EXPERIMENT_NAME run8.sh run.py train \
                    --num-workers 7 \
                    --model-name $model \
                    --dataset-name $dataset \
                    --subset-size $subset_size \
                    --dataset-label-field $label_field \
                    --batch-size $BATCH_SIZE \
                    --learning-rate $LEARNING_RATE \
                    --max-grad-norm $MAX_GRAD_NORM \
                    --total-steps $STEPS \
                    --noise-batch-ratio $noise_batch_ratio \
                    --seed $seed \
                    --physical-batch-size 40 \
                    --experiment-name $EXPERIMENT_NAME \
                    --log-dir $LOG_DIR \
                    $ZERO_HEAD \
                    $PEFT \
                    $PRIVACY \
                    $USE_STEPS \
                    $NORMALIZE_CLIPPING \
                    $OVERWRITE_EXPERIMENT \
                    $RECORD_LOSS \
                    $EVALUATION_MODE \
                    $CHECKPOINT \
                    $DISABLE_EPSILON_LOGGING

                SBATCH_EXIT_CODE=$?

                if [ $SBATCH_EXIT_CODE -eq 0 ]; then
                    echo "Job $EXPERIMENT_NAME submitted successfully."
                else
                    echo "Submission of $EXPERIMENT_NAME failed with exit code $SBATCH_EXIT_CODE."
                    exit 1
                fi
            done
        done
    done
done
