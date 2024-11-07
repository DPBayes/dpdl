#!/bin/bash
##################################################
### Experiment: Alternative Approaches to HPO
##################################################
#
set -euo pipefail

# Base configurations
EXPERIMENT_BASE="18-hpo-alternatives"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# We keep track of submitted jobs here
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG  # Create the file if it doesn't exist

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k")
DATASETS=("dpdl-benchmark/svhn_cropped_balanced")
EPOCHS=40
EPSILON=4  # Fixed ε = 4.0
SEEDS=(43 44 45 46 47 48 49 50 51 52)
METHODS=("seeded_good" "seeded_avg" "seeded_bad")

OPTUNA_CONFIG="conf/optuna_hypers-hpo-alternatives.conf"

# These contain the hypers for the manual trials

declare -A DATASET_LABEL_FIELDS
DATASET_LABEL_FIELDS=(
    ["dpdl-benchmark/svhn_cropped_balanced"]="label"
)

declare -A SUBSET_SIZES
SUBSET_SIZES=(
    ["dpdl-benchmark/svhn_cropped_balanced"]="0.1"
)

DEFAULT_N_TRIALS=20
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
OPTUNA_JOURNAL="$LOG_DIR/optuna.journal"
PHYSICAL_BATCH_SIZE="40"
PEFT="film"

# Function to check if a job has been submitted
function is_job_submitted() {
    grep -Fxq "$1" $JOB_STATUS_LOG
}

# Function to check if a job is already in queue
function is_job_in_queue() {
    local experiment_name=$1
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0  # Job is in queue
    else
        return 1  # Job is not in queue
    fi
}

# Loop over configurations
for seed in "${SEEDS[@]}"
do
    for model in "${MODELS[@]}"
    do
        for dataset in "${DATASETS[@]}"
        do
            label_field=${DATASET_LABEL_FIELDS[$dataset]}
            subset_size=${SUBSET_SIZES[$dataset]}
            clean_dataset_name=${dataset#dpdl-benchmark/}

            for method in "${METHODS[@]}"
            do
                case $method in
                    "seeded_good")
                        MANUAL_TRIAL_FILE="conf/optuna_trials-svhn_cropped_balanced-good.conf"
                        N_TRIALS=$DEFAULT_N_TRIALS
                        ;;

                    "seeded_avg")
                        MANUAL_TRIAL_FILE=$"conf/optuna_trials-svhn_cropped_balanced-avg.conf"
                        N_TRIALS=$DEFAULT_N_TRIALS
                        ;;

                    "seeded_bad")
                        MANUAL_TRIAL_FILE=$"conf/optuna_trials-svhn_cropped_balanced-bad.conf"
                        N_TRIALS=$DEFAULT_N_TRIALS
                        ;;
                esac

                EXPERIMENT_NAME="${model}_${clean_dataset_name}_Subset${subset_size}_Epsilon${EPSILON}_${method}_Seed${seed}"
                EXPERIMENT_DIR="$LOG_DIR/$EXPERIMENT_NAME"
                mkdir -p "$EXPERIMENT_DIR"

                if [ -f "$EXPERIMENT_DIR/runtime" ]; then
                    echo "Experiment $EXPERIMENT_NAME has completed. Skipping submission."
                    continue
                fi

                if is_job_in_queue $EXPERIMENT_NAME; then
                    echo "Experiment $EXPERIMENT_NAME is already in the queue."
                    continue
                fi

                # If the jobs has already been started, then we need to calculate remaining trials
                if is_job_submitted $EXPERIMENT_NAME; then
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

                sbatch -J $EXPERIMENT_NAME run8-rocm.sh run.py optimize \
                    --num-workers 7 \
                    --model-name $model \
                    --peft $PEFT \
                    --dataset-name $dataset \
                    --subset-size $subset_size \
                    --dataset-label-field $label_field \
                    --target-hypers learning_rate \
                    --target-hypers batch_size \
                    --target-hypers max_grad_norm \
                    --epochs $EPOCHS \
                    --target-epsilon $EPSILON \
                    --n-trials $REMAINING_TRIALS \
                    --seed $seed \
                    --optuna-manual-trials $MANUAL_TRIAL_FILE \
                    --optuna-config $OPTUNA_CONFIG \
                    --experiment-name $EXPERIMENT_NAME \
                    --physical-batch-size $PHYSICAL_BATCH_SIZE \
                    --log-dir $LOG_DIR \
                    $PRIVACY \
                    $USE_STEPS \
                    $NORMALIZE_CLIPPING \
                    $ZERO_HEAD \
                    $OVERWRITE_EXPERIMENT \
                    $OPTUNA_RESUME \
                    --optuna-journal $OPTUNA_JOURNAL

                if [ $? -eq 0 ]; then
                    echo "Job $EXPERIMENT_NAME submitted successfully."
                    if [ "$N_TRIALS" -eq -1 ]; then
                        echo $EXPERIMENT_NAME >> $JOB_STATUS_LOG
                    fi
                else
                    echo "Submission of $EXPERIMENT_NAME failed."
                    exit 1
                fi
            done
        done
    done
done
