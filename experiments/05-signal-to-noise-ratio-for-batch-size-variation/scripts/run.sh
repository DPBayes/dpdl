#!/bin/bash

set -euo pipefail

##################################################
### Experiment: Training with Recorded SNR
##################################################

# Base configurations
EXPERIMENT_BASE="05-signal-to-noise-ratio-for-batch-size-variation"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# Path to JSON file containing aggregated data
AGGREGATED_DATA_JSON="experiments/00-experiment-batch-size-variation/processed-data/aggregated_data.json"

# Job submission log file to track submitted experiments
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG  # Ensure the file exists

# Other settings
SEED=42
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna.journal"
RECORD_SNR="--record-snr"
PEFT="--peft film"

# these two are for easy requeing: if the jobs get stuck at the
# beginning, we can just `scontrol requeue <jobid>`
OPTUNA_RESUME="--no-optuna-resume"
OVERWRITE_EXPERIMENT="--overwrite-experiment"

# Function to check if a job has been submitted
function is_job_submitted() {
    grep -Fxq "$1" $JOB_STATUS_LOG
}

# Iterate through all experiments in the JSON file
for ID in $(jq -r 'keys[]' "$AGGREGATED_DATA_JSON")
do
    # Check if this job has already been submitted
    if is_job_submitted $ID; then
        echo "!!! Already submitted: $ID"
        continue  # Skip this iteration
    fi

    # Extract the whole experiment data block for the current ID
    DATA=$(jq ".\"$ID\"" "$AGGREGATED_DATA_JSON")

    # Directly extract parameters using jq
    MODEL=$(echo $DATA | jq -r '.configuration.model_name')
    DATASET_NAME=$(echo $DATA | jq -r '.configuration.dataset_name')
    SUBSET_SIZE=$(echo $DATA | jq -r '.configuration.subset_size')
    NUM_CLASSES=$(echo $DATA | jq -r '.configuration.num_classes')
    EPOCHS=$(echo $DATA | jq -r '.best_params.epochs')
    LEARNING_RATE=$(echo $DATA | jq -r '.best_params.learning_rate')
    MAX_GRAD_NORM=$(echo $DATA | jq -r '.best_params.max_grad_norm')
    BATCH_SIZE=$(echo $DATA | jq -r '.hyperparameters.batch_size')
    TARGET_EPSILON=$(echo $DATA | jq -r '.hyperparameters.target_epsilon')

    # Submit the training job with extracted parameters
    sbatch -J $ID run8.sh run.py train \
        --num-workers 7 \
        --model-name $MODEL \
        --dataset-name $DATASET_NAME \
        --subset-size $SUBSET_SIZE \
        --num-classes $NUM_CLASSES \
        --epochs $EPOCHS \
        --learning-rate $LEARNING_RATE \
        --max-grad-norm $MAX_GRAD_NORM \
        --batch-size $BATCH_SIZE \
        --target-epsilon $TARGET_EPSILON \
        --experiment-name $ID \
        --log-dir $LOG_DIR \
        --seed $SEED \
        --physical-batch-size 40 \
        $PEFT \
        $PRIVACY \
        $NORMALIZE_CLIPPING \
        $ZERO_HEAD \
        $USE_STEPS \
        $OVERWRITE_EXPERIMENT \
        $OPTUNA_RESUME \
        $OPTUNA_JOURNAL \
        $RECORD_SNR

    # Assume submission is successful for demonstration purposes
    # In practice, check submission success before logging
    echo $ID >> $JOB_STATUS_LOG
done
