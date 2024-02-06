#!/bin/bash

set -euo pipefail

### Experiment: Training with Bad Hyperparameters ###

# Base configurations
EXPERIMENT_BASE="05-signal-to-noise-ratio-for-batch-size-variation"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data-bad-hypers"
mkdir -p $LOG_DIR

# Path to JSON file containing bad hyperparameters
# NB: Use bin/find_bad_hypers.py to generate one.
BAD_HYPERS_JSON=${1:-"bad_hypers.json"}

# Job submission log file to track submitted experiments
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG

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

function parse_experiment_name() {
    local exp_name="$1"

    # Regular expression to parse the experiment name
    # Assumes format: <model_name>_cifar10|cifar100_Subset<subset_size>_Epsilon<epsilon>_BatchSize<batch_size>
    regex="^(.*)_((cifar10|cifar100)_Subset([0-9.]+)_Epsilon([0-9.]+)_BatchSize([0-9]+))$"

    if [[ $exp_name =~ $regex ]]; then
        # Extract captured groups
        MODEL="${BASH_REMATCH[1]}"
        DATASET_NAME="${BASH_REMATCH[3]}"
        SUBSET_SIZE="${BASH_REMATCH[4]}"
        TARGET_EPSILON="${BASH_REMATCH[5]}"
        BATCH_SIZE="${BASH_REMATCH[6]}"
    else
        echo "Failed to parse experiment name: $exp_name"
        # Handle the error case, e.g., by skipping this experiment or setting default values
    fi
}

# Iterate through all experiments in the JSON file
for ID in $(jq -r 'keys[]' "$BAD_HYPERS_JSON")
do
    # Check if this job has already been submitted
    if is_job_submitted $ID; then
        echo "!!! Already submitted: $ID"
        continue
    fi

    parse_experiment_name "$ID"

    # Adjust number of classes based on the dataset
    if [ "$DATASET_NAME" == "cifar100" ]; then
        NUM_CLASSES=100
    else
        NUM_CLASSES=10
    fi

    # Extract the bad hyperparameters for the current ID
    HYPERPARAMS=$(jq ".\"$ID\"" "$BAD_HYPERS_JSON")

    # Extract parameters directly using jq
    BATCH_SIZE=$(echo $HYPERPARAMS | jq -r '.batch_size')
    LEARNING_RATE=$(echo $HYPERPARAMS | jq -r '.learning_rate')
    EPOCHS=$(echo $HYPERPARAMS | jq -r '.epochs')
    MAX_GRAD_NORM=$(echo $HYPERPARAMS | jq -r '.max_grad_norm')

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

    echo $ID >> $JOB_STATUS_LOG
done
