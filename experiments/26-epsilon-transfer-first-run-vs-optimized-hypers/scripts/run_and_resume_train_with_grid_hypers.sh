#!/bin/bash
set -euo pipefail

# CSV from the grid experiments
CSV_FILE="experiments/26-epsilon-transfer-first-run-vs-optimized-hypers/best_hyperparams_from_grid.csv"

# Base directories
EXPERIMENT_BASE="26-epsilon-transfer-first-run-vs-optimized-hypers"
LOG_DIR="/scratch/$PROJECT/experiments/$EXPERIMENT_BASE/data_train_with_grid_hypers"
mkdir -p "$LOG_DIR"

# Define datasets and their label fields (adjust as needed)
declare -A DATASET_LABEL_FIELDS=(
    ["cifar100"]="fine_label"
    ["dpdl-benchmark/svhn_cropped"]="label"
    ["dpdl-benchmark/svhn_cropped_balanced"]="label"
    ["dpdl-benchmark/sun397"]="label"
    ["dpdl-benchmark/patch_camelyon"]="label"
    ["dpdl-benchmark/cassava"]="label"
    ["dpdl-benchmark/imagenet397"]="label"
)

# Standard training parameters
EPOCHS=40
SEED=42
PHYSICAL_BATCH_SIZE=40
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
PEFT="--peft film"
OVERWRITE_EXPERIMENT="--overwrite-experiment"
VALIDATION_FREQUENCY="--validation-frequency 0"
EVALUATION_MODE="--evaluation-mode"
TARGET_EPSILONS=("1" "2" "4" "8")

# Function to check if a job is already in queue.
function is_job_in_queue() {
    local experiment_name="$1"
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0
    else
        return 1
    fi
}

# Loop through each row in the CSV (skipping header).
# CSV columns:
# - dataset
# - subset
# - hyperparameters.target_epsilo
# - hyperparameters.learning_rate
# - hyperparameters.batch_size
# - hyperparameters.max_grad_norm
# - test_metrics.MulticlassAccuracy
tail -n +2 "$CSV_FILE" | while IFS=, read -r dataset subset source_epsilon best_lr best_bs best_mgn best_acc; do
    # Trim whitespace.
    dataset=$(echo "$dataset" | xargs)
    subset=$(echo "$subset" | xargs)
    source_epsilon=$(echo "$source_epsilon" | xargs)
    best_lr=$(echo "$best_lr" | xargs)
    best_bs=$(echo "$best_bs" | xargs)
    best_mgn=$(echo "$best_mgn" | xargs)

    # Format source epsilon
    formatted_source_eps=$(printf "%.4f" "$source_epsilon")

    # Clean dataset name for the experiment directory by stripping "dpdl-benchmark/".
    clean_dataset=$(echo "$dataset" | sed 's|dpdl-benchmark/||')

    # Loop over each target epsilons
    for target_epsilon in "${TARGET_EPSILONS[@]}"; do
        formatted_target_eps=$(printf "%.4f" "$target_epsilon")
        exp_name="vit_base_patch16_224.augreg_in21k_${clean_dataset}_Subset${subset}_Epsilon${formatted_source_eps}_TargetEpsilon${formatted_target_eps}"
        exp_dir="$LOG_DIR/$exp_name"
        mkdir -p "$exp_dir"

        label_field="${DATASET_LABEL_FIELDS[$dataset]}"
        if [ -z "$label_field" ]; then
            echo "No label field defined for dataset $dataset. Skipping."
            continue
        fi

        if [ -f "$exp_dir/runtime" ]; then
            echo "Experiment $exp_name already completed. Skipping."
            continue
        fi

        if is_job_in_queue "$exp_name"; then
            echo "Experiment $exp_name is already in the queue. Skipping."
            continue
        fi

        echo "Submitting experiment $exp_name"
        echo "  Using best hyperparams from source epsilon $formatted_source_eps: LR=$best_lr, BS=$best_bs, MGN=$best_mgn, (Source Accuracy=$best_acc)"
        echo "  Training at target epsilon $formatted_target_eps"

        sbatch -J "$exp_name" run8-rocm.sh run.py train \
            --num-workers 7 \
            --model-name "vit_base_patch16_224.augreg_in21k" \
            --dataset-name "$dataset" \
            --subset-size "$subset" \
            --dataset-label-field "$label_field" \
            --epochs "$EPOCHS" \
            --seed "$SEED" \
            --physical-batch-size "$PHYSICAL_BATCH_SIZE" \
            --target-epsilon "$target_epsilon" \
            --batch-size "$best_bs" \
            --learning-rate "$best_lr" \
            --max-grad-norm "$best_mgn" \
            --experiment-name "$exp_name" \
            --log-dir "$LOG_DIR" \
            $PRIVACY \
            $USE_STEPS \
            $NORMALIZE_CLIPPING \
            $ZERO_HEAD \
            $PEFT \
            $OVERWRITE_EXPERIMENT \
            $VALIDATION_FREQUENCY \
            $EVALUATION_MODE

        SBATCH_EXIT_CODE=$?
        if [ "$SBATCH_EXIT_CODE" -ne 0 ]; then
            echo "Submission of $exp_name failed with exit code $SBATCH_EXIT_CODE."
            exit 1
        fi
    done
done
