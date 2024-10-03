#!/bin/bash
##################################################
### Experiment: Privacy assistant evaluation
##################################################
#
set -euo pipefail

# Base configurations
EXPERIMENT_BASE="15-privacy-assistant"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# We keep track of submitted jobs here
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG  # Create the file if it doesn't exist

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k")
DATASETS=(
    "dpdl-benchmark/caltech_birds2011"
    "dpdl-benchmark/sun397"
    "dpdl-benchmark/eurosat"
    "dpdl-benchmark/oxford_iit_pet"
    "dpdl-benchmark/plant_village"
    "dpdl-benchmark/colorectal_histology"
    "dpdl-benchmark/cassava"
)
EPOCHS=40
EPSILONS="0.25 0.3149802624737183 0.39685026299204984 0.5 0.6299605249474366 0.7937005259840998 1.0 1.2599210498948732 1.5874010519681994 1.9999999999999998 2.5198420997897464 3.1748021039363983 4.0 5.039684199579491 6.3496042078727974 7.999999999999997 10.079368399158986 12.699208415745593 16.0"
OPTUNA_CONFIG="conf/optuna_hypers-privacy-assistant.conf"

declare -A DATASET_LABEL_FIELDS
DATASET_LABEL_FIELDS=(
    ["dpdl-benchmark/caltech_birds2011"]="label"
    ["dpdl-benchmark/sun397"]="label"
    ["dpdl-benchmark/eurosat"]="label"
    ["dpdl-benchmark/oxford_iiit_pet"]="label"
    ["dpdl-benchmark/plant_village"]="label"
    ["dpdl-benchmark/colorectal_histology"]="label"
    ["dpdl-benchmark/cassava"]="label"
)

declare -A SUBSET_SIZES
SUBSET_SIZES=(
    ["dpdl-benchmark/caltech_birds2011"]="1.0"
    ["dpdl-benchmark/sun397"]="0.1"
    ["dpdl-benchmark/eurosat"]="1.0"
    ["dpdl-benchmark/oxford_iiit_pet"]="1.0"
    ["dpdl-benchmark/plant_village"]="0.1"
    ["dpdl-benchmark/colorectal_histology"]="1.0"
    ["dpdl-benchmark/cassava"]="1.0"
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

        for epsilon in $EPSILONS
        do
            rounded_epsilon=$(printf "%.2f" $epsilon)

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
