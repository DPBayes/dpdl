#!/bin/bash
##################################################
### Experiment: Comparison of us vs HyFreeDP
##################################################
#
set -euo pipefail

# Base configurations
EXPERIMENT_BASE="33-us-vs-hyfreedp-full"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# We keep track of submitted jobs here
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG  # Create the file if it doesn't exist

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k")
DATASETS=(
    "cifar100"
    "cifar10"
    "food101"
    "dpdl-benchmark/svhn_cropped"
    "dpdl-benchmark/gtsrb"
)
EPOCHS=40
EPSILONS="1 3 8"
OPTUNA_CONFIG="conf/optuna_hypers_ordered.conf"

declare -A DATASET_LABEL_FIELDS
DATASET_LABEL_FIELDS=(
    ["cifar10"]="label"
    ["cifar100"]="fine_label"
    ["food101"]="label"
    ["dpdl-benchmark/svhn_cropped"]="label"
    ["dpdl-benchmark/gtsrb"]="label"
)

declare -A SUBSET_SIZES
SUBSET_SIZES=(
    ["cifar10"]="1.0"
    ["cifar100"]="1.0"
    ["food101"]="1.0"
    ["dpdl-benchmark/svhn_cropped"]="1.0"
    ["dpdl-benchmark/gtsrb"]="1.0"
)

# Other settings
SEED=42
DEFAULT_N_TRIALS=20
N_TRIALS=-1 # Default to new experiment, this will be overridden
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
OPTUNA_JOURNAL="$LOG_DIR/optuna.journal"
PEFT="--peft film"

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

        if [ "$clean_dataset_name" = 'sun397' ]; then
            EPSILONS="$EPSILONS 17.211049379522922 18.513763796523516 19.915081431424372 21.42246561959156 23.04394460062014 24.788154276266493 26.664384204753418 28.682627076411382 30.853631934159846 33.18896142227813 35.701053368369394 38.40328702649928 41.310054334316945 44.436836563669 47.80028677294355 51.41831850028046 55.31020117002272 59.49666272053798 64.0"
        fi

        for epsilon in $EPSILONS
        do
            rounded_epsilon=$(printf "%.4f" $epsilon)

            EXPERIMENT_NAME="${model}_${clean_dataset_name}_Subset${subset_size}_Epsilon${rounded_epsilon}"
            EXPERIMENT_DIR="$LOG_DIR/$EXPERIMENT_NAME"
            mkdir -p "$EXPERIMENT_DIR"

            # If `runtime` file exists, then the job has been completed
            if [ -f "$EXPERIMENT_DIR/runtime" ]; then
                echo "Experiment $EXPERIMENT_NAME has completed. Skipping submission."
                continue
            fi

            # Determine if we should attempt to resume based on job submission status
            if is_job_submitted $EXPERIMENT_NAME; then
                N_TRIALS=$(python bin/get_n_trials.py --optuna-journal "$OPTUNA_JOURNAL" --study-name "$EXPERIMENT_NAME")

                if [ "$N_TRIALS" -eq -1 ]; then
                    # If N_TRIALS is -1, then the job has never been started
                    REMAINING_TRIALS=$DEFAULT_N_TRIALS
                    OPTUNA_RESUME="--no-optuna-resume"
                    OVERWRITE_EXPERIMENT="--overwrite-experiment"

                    echo "New job: $EXPERIMENT_NAME"
                else
                    # Otherwise, we want to resume it with the remaining trials
                    REMAINING_TRIALS=$((DEFAULT_N_TRIALS - N_TRIALS))
                    OPTUNA_RESUME="--optuna-resume"
                    OVERWRITE_EXPERIMENT="--no-overwrite-experiment"

                    echo "Resuming job: $EXPERIMENT_NAME with $REMAINING_TRIALS trials"
                fi
            else
                # Job not submitted, treat as new job
                N_TRIALS=-1
                REMAINING_TRIALS=$DEFAULT_N_TRIALS
                OPTUNA_RESUME="--no-optuna-resume"
                OVERWRITE_EXPERIMENT="--overwrite-experiment"
                echo "New job: $EXPERIMENT_NAME"
            fi

            if is_job_in_queue $EXPERIMENT_NAME; then
                echo "Experiment $EXPERIMENT_NAME is already in the queue."
                continue  # Skip to the next iteration
            fi

            # Submit the job
            sbatch -J $EXPERIMENT_NAME run8-rocm.sh run.py optimize \
                --num-workers 7 \
                --model-name $model \
                --dataset-name $dataset \
                --subset-size $subset_size \
                --dataset-label-field $label_field \
                --target-hypers learning_rate \
                --target-hypers batch_size \
                --target-hypers max_grad_norm \
                --epochs $EPOCHS \
                --target-epsilon $epsilon \
                --n-trials $REMAINING_TRIALS \
                --seed $SEED \
                --physical-batch-size 40 \
                --optuna-config $OPTUNA_CONFIG \
                --optuna-target-metric MulticlassAccuracy \
                --optuna-direction maximize \
                --experiment-name $EXPERIMENT_NAME \
                --log-dir $LOG_DIR \
                $ZERO_HEAD \
                $PEFT \
                $PRIVACY \
                $USE_STEPS \
                $NORMALIZE_CLIPPING \
                $OVERWRITE_EXPERIMENT \
                $OPTUNA_RESUME \
                --optuna-journal $OPTUNA_JOURNAL

            SBATCH_EXIT_CODE=$?

            if [ $SBATCH_EXIT_CODE -eq 0 ]; then
                echo "Job $EXPERIMENT_NAME submitted successfully."
                if [ "$N_TRIALS" -eq -1 ]; then
                    echo $EXPERIMENT_NAME >> $JOB_STATUS_LOG  # Log the submission for new experiments only
                fi
            else
                echo "Submission of $EXPERIMENT_NAME failed with exit code $SBATCH_EXIT_CODE."
                exit 1
            fi
        done
    done
done
