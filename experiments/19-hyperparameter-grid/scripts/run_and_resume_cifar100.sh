#!/bin/bash
############################################################
### Experiment: Hyperparameter grid (10% of cifar100)
############################################################

set -euo pipefail

# Base configurations
EXPERIMENT_BASE="19-hyperparameter-grid"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data_cifar100_epsilon8"
mkdir -p $LOG_DIR

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k")
DATASETS=(
#    "dpdl-benchmark/sun397"
#    "dpdl-benchmark/svhn_cropped_balanced"
    "cifar100"
)

EPOCHS=40
LEARNING_RATES="0.0005 0.0009596915518332421 0.001842015749320193 0.003535533905932737 0.006786044041487265 0.013025018273967286 0.025"
BATCH_SIZES="192 512 1024 2048 4096 -1"
MAX_GRAD_NORMS="0.001 0.004135185542000139 0.01709975946676697 0.07071067811865475 0.2924017738212867 1.2091355875609793 5.0"
#EPSILONS="1 8"
EPSILONS="8"
SEEDS="42"

declare -A DATASET_LABEL_FIELDS
DATASET_LABEL_FIELDS=(
    ["dpdl-benchmark/sun397"]="label"
    ["dpdl-benchmark/svhn_cropped_balanced"]="label"
    ["cifar100"]="fine_label"
)

declare -A SUBSET_SIZES
SUBSET_SIZES=(
    ["dpdl-benchmark/sun397"]="0.1"
    ["dpdl-benchmark/svhn_cropped_balanced"]="0.1"
    ["cifar100"]="0.1"
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

            for batch_size in $BATCH_SIZES
            do
                for learning_rate in $LEARNING_RATES
                do
                    rounded_learning_rate=$(printf "%.4f" $learning_rate)

                    for max_grad_norm in $MAX_GRAD_NORMS
                    do
                        rounded_max_grad_norm=$(printf "%.3f" $max_grad_norm)

                        for epsilon in $EPSILONS
                        do
                            rounded_epsilon=$(printf "%.2f" $epsilon)

                            EXPERIMENT_NAME="${model}_${clean_dataset_name}_Subset${subset_size}_Epsilon${rounded_epsilon}_BatchSize${batch_size}_LearningRate${rounded_learning_rate}_MaxGradNorm${rounded_max_grad_norm}_Seed${seed}"

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
                            sbatch -J $EXPERIMENT_NAME run8-rocm.sh run.py train \
                                --num-workers 7 \
                                --model-name $model \
                                --dataset-name $dataset \
                                --subset-size $subset_size \
                                --dataset-label-field $label_field \
                                --batch-size $batch_size \
                                --learning-rate $learning_rate \
                                --max-grad-norm $max_grad_norm \
                                --epochs $EPOCHS \
                                --target-epsilon $epsilon \
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
                                $EVALUATION_MODE

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
        done
    done
done
