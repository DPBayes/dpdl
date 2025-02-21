#!/bin/bash
###############################################################
### Experiment: ε to ε -transfer: First run vs Optimized hypers
###     Step 2: Training round.
###############################################################

set -euo pipefail

# Base configurations
EXPERIMENT_BASE="26-epsilon-transfer-first-run-vs-optimized-hypers"
LOG_BASE="/scratch/$PROJECT/experiments/$EXPERIMENT_BASE/data_train_with_hpo_hypers"
mkdir -p "$LOG_BASE"

EPOCHS="40"
TARGET_EPSILONS="0.5 1 2 4 8 16"

# Directory with HPO step experiment data
SOURCE_DATA_DIR="experiments/$EXPERIMENT_BASE/data"

# Training flags
OVERWRITE_EXPERIMENT="--overwrite-experiment"
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
PEFT="--peft film"
EVALUATION_MODE="--evaluation-mode"
VALIDATION_FREQUENCY="--validation-frequency 0"

# Function to check if a job is already in queue
function is_job_in_queue() {
    local experiment_name=$1
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0
    else
        return 1
    fi
}

# Loop through the HPO experiment data
for source_dir in "$SOURCE_DATA_DIR"/*; do
    if [ ! -d "$source_dir" ]; then
        continue
    fi

    # Check for necessary configuration files
    if [ ! -f "$source_dir/configuration.json" ] || [ ! -f "$source_dir/best-params.json" ]; then
        echo "Skipping $source_dir: Missing configuration.json or best-params.json" continue
    fi

    # Extract experiment settings from configuration.json
    source_experiment_name=$(jq -r '.experiment_name' "$source_dir/configuration.json")
    model_name=$(jq -r '.model_name' "$source_dir/configuration.json")
    dataset_name=$(jq -r '.dataset_name' "$source_dir/configuration.json")
    subset_size=$(jq -r '.subset_size' "$source_dir/configuration.json")
    dataset_label_field=$(jq -r '.dataset_label_field' "$source_dir/configuration.json")
    num_workers=$(jq -r '.num_workers' "$source_dir/configuration.json")
    physical_batch_size=$(jq -r '.physical_batch_size' "$source_dir/configuration.json")
    seed=$(jq -r '.seed' "$source_dir/configuration.json")

    # Extract best hyperparameters from best-params.json
    best_lr=$(jq -r '.learning_rate' "$source_dir/best-params.json")
    best_bs=$(jq -r '.batch_size' "$source_dir/best-params.json")
    best_grad=$(jq -r '.max_grad_norm' "$source_dir/best-params.json")

    # Loop over each target epsilon for training
    for target_epsilon in $TARGET_EPSILONS; do
        new_experiment_name="${source_experiment_name}_TargetEpsilon${target_epsilon}"
        new_experiment_dir="${LOG_BASE}/${new_experiment_name}"
        mkdir -p "$new_experiment_dir"

        if [ -f "$new_experiment_dir/runtime" ]; then
            echo "Experiment $new_experiment_name already completed. Skipping."
            continue
        fi

        if is_job_in_queue "$new_experiment_name"; then
            echo "Experiment $new_experiment_name is already in the queue. Skipping."
            continue
        fi

        echo "Submitting training job $new_experiment_name"

        sbatch -J "$new_experiment_name" run8-rocm.sh run.py train \
            --num-workers "$num_workers" \
            --model-name "$model_name" \
            --dataset-name "$dataset_name" \
            --subset-size "$subset_size" \
            --dataset-label-field "$dataset_label_field" \
            --batch-size "$best_bs" \
            --learning-rate "$best_lr" \
            --max-grad-norm "$best_grad" \
            --epochs "$EPOCHS" \
            --target-epsilon "$target_epsilon" \
            --seed "$seed" \
            --physical-batch-size "$physical_batch_size" \
            --experiment-name "$new_experiment_name" \
            --log-dir "$LOG_BASE" \
            $ZERO_HEAD \
            $PEFT \
            $PRIVACY \
            $USE_STEPS \
            $NORMALIZE_CLIPPING \
            $OVERWRITE_EXPERIMENT \
            $VALIDATION_FREQUENCY \
            $EVALUATION_MODE

        SBATCH_EXIT_CODE=$?
        if [ "$SBATCH_EXIT_CODE" -eq 0 ]; then
            echo "Job $new_experiment_name submitted successfully."
        else
            echo "Submission of $new_experiment_name failed with exit code $SBATCH_EXIT_CODE."
            exit 1
        fi
    done
done

