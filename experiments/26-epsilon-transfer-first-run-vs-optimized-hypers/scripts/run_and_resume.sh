#!/bin/bash
###############################################################
### Experiment: ε to ε -transfer: First run vs Optimized hypers
###############################################################

set -euo pipefail

EXPERIMENT_BASE="26-epsilon-transfer-first-run-vs-optimized-hypers"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p "$LOG_DIR"

JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch "$JOB_STATUS_LOG"

MODELS=("vit_base_patch16_224.augreg_in21k")
DATASETS=(
    "cifar100"
    "dpdl-benchmark/sun397"
    "dpdl-benchmark/patch_camelyon"
    "dpdl-benchmark/cassava"
    "dpdl-benchmark/svhn_cropped"
    "dpdl-benchmark/svhn_cropped_balanced"
)
EPOCHS=40
EPSILONS="0.5 1 2 4 8 16"
OPTUNA_CONFIG="conf/optuna_hypers-epsilon-transfer-first-run-vs-optimized-hypers.md"

declare -A DATASET_LABEL_FIELDS=(
    ["cifar100"]="fine_label"
    ["dpdl-benchmark/sun397"]="label"
    ["dpdl-benchmark/patch_camelyon"]="label"
    ["dpdl-benchmark/cassava"]="label"
    ["dpdl-benchmark/svhn_cropped"]="label"
    ["dpdl-benchmark/svhn_cropped_balanced"]="label"
)

declare -A SUBSET_SIZES=(
    ["cifar100"]="0.1"
    ["dpdl-benchmark/sun397"]="0.1"
    ["dpdl-benchmark/patch_camelyon"]="0.02"
    ["dpdl-benchmark/cassava"]="1.0"
    ["dpdl-benchmark/svhn_cropped"]="0.1"
    ["dpdl-benchmark/svhn_cropped_balanced"]="0.1"
)

# Other settings
SEED=42
DEFAULT_N_TRIALS=20
N_TRIALS=-1
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
PEFT="--peft film"
OPTUNA_JOURNAL="$LOG_DIR/optuna.journal"

# Function to check if a job has been submitted
function is_job_submitted() {
    grep -Fxq "$1" "$JOB_STATUS_LOG"
}

# Function to check if a job is already in queue
function is_job_in_queue() {
    local experiment_name="$1"
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0
    else
        return 1
    fi
}

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

        for epsilon in $EPSILONS
        do
            rounded_epsilon=$(printf "%.4f" "$epsilon")

            EXPERIMENT_NAME="${model}_${clean_dataset_name}_Subset${subset_size}_Epsilon${rounded_epsilon}"
            EXPERIMENT_DIR="$LOG_DIR/$EXPERIMENT_NAME"
            mkdir -p "$EXPERIMENT_DIR"

            if [ -f "$EXPERIMENT_DIR/runtime" ]; then
                echo "Experiment $EXPERIMENT_NAME has completed. Skipping."
                continue
            fi

            if is_job_in_queue "$EXPERIMENT_NAME"; then
                echo "Experiment $EXPERIMENT_NAME is already in the queue."
                continue
            fi

            if is_job_submitted "$EXPERIMENT_NAME"; then
                N_TRIALS=$(python bin/get_n_trials.py --optuna-journal "$OPTUNA_JOURNAL" --study-name "$EXPERIMENT_NAME")
                if [ "$N_TRIALS" -eq -1 ]; then
                    REMAINING_TRIALS=$DEFAULT_N_TRIALS
                    OPTUNA_RESUME="--no-optuna-resume"
                    OVERWRITE_EXPERIMENT="--overwrite-experiment"
                    echo "New job: $EXPERIMENT_NAME"
                else
                    REMAINING_TRIALS=$((DEFAULT_N_TRIALS - N_TRIALS))
                    OPTUNA_RESUME="--optuna-resume"
                    OVERWRITE_EXPERIMENT="--no-overwrite-experiment"
                    echo "Resuming job: $EXPERIMENT_NAME with $REMAINING_TRIALS trials"
                fi
            else
                N_TRIALS=-1
                REMAINING_TRIALS=$DEFAULT_N_TRIALS
                OPTUNA_RESUME="--no-optuna-resume"
                OVERWRITE_EXPERIMENT="--overwrite-experiment"
                echo "New job: $EXPERIMENT_NAME"
            fi

            sbatch -J "$EXPERIMENT_NAME" run8-rocm.sh run.py optimize \
                --num-workers 7 \
                --model-name "$model" \
                --dataset-name "$dataset" \
                --subset-size "$subset_size" \
                --dataset-label-field "$label_field" \
                --target-hypers learning_rate \
                --target-hypers batch_size \
                --target-hypers max_grad_norm \
                --epochs "$EPOCHS" \
                --target-epsilon "$epsilon" \
                --n-trials "$REMAINING_TRIALS" \
                --seed "$SEED" \
                --physical-batch-size 40 \
                --optuna-config "$OPTUNA_CONFIG" \
                --optuna-target-metric MulticlassAccuracy \
                --optuna-direction maximize \
                --optuna-journal "$OPTUNA_JOURNAL" \
                --experiment-name "$EXPERIMENT_NAME" \
                --log-dir "$LOG_DIR" \
                $ZERO_HEAD \
                $PEFT \
                $PRIVACY \
                $USE_STEPS \
                $NORMALIZE_CLIPPING \
                $OVERWRITE_EXPERIMENT \
                $OPTUNA_RESUME

            SBATCH_EXIT_CODE=$?
            if [ "$SBATCH_EXIT_CODE" -eq 0 ]; then
                echo "Job $EXPERIMENT_NAME submitted."
                if [ "$N_TRIALS" -eq -1 ]; then
                    echo "$EXPERIMENT_NAME" >> "$JOB_STATUS_LOG"
                fi
            else
                echo "Submission of $EXPERIMENT_NAME failed with exit code $SBATCH_EXIT_CODE."
                exit 1
            fi
        done
    done
done

