#!/bin/bash
##################################################
### Experiment: Few-shot from scratch without DP
##################################################

set -euo pipefail

# Base configurations
EXPERIMENT_BASE="06-few-shot-from-scratch-non-dp"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# Experiment name to be resumed
RESUME_EXPERIMENT=$1

# We also have an option to set the number of trials to run
N_TRIALS=${2:-40}

# Experiment parameters
MODELS=("wrn-16-4" "koskela-net")
ALL_SHOTS=(1 5 10 25 50 100 250 500 1000 1500 2000 2500 3000 3500 4000 4500)
DATASET="cifar10"
NUM_CLASSES=10

# Other settings
SEED=42
PRIVACY="--no-privacy"
USE_STEPS="--use-steps"
PRETRAINED="--no-pretrained"
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna.journal"

OPTUNA_RESUME="--optuna-resume"
OVERWRITE_EXPERIMENT="--no-overwrite-experiment"

# Function to check if a job has been submitted
function is_job_submitted() {
    grep -Fxq "$1" $JOB_STATUS_LOG
}

for model in "${MODELS[@]}"
do
    for shots in "${ALL_SHOTS[@]}"
    do
        EXPERIMENT_NAME="${model}_${DATASET}_32x32_EPOCHFIX_Shots${shots}"

        OPTUNA_CONFIG="conf/optuna_hypers-shots${shots}.conf"

        # Check if this job has already been submitted
        if [ "$RESUME_EXPERIMENT" == "$EXPERIMENT_NAME" ]; then
            # Submit the job and capture its success or failure
            sbatch -J $EXPERIMENT_NAME run8.sh run.py optimize \
                --num-workers 7 \
                --model-name $model \
                --dataset-name $DATASET \
                --shots $shots \
                --num-classes $NUM_CLASSES \
                --target-hypers epochs \
                --target-hypers learning_rate \
                --target-hypers batch_size \
                --n-trials $N_TRIALS \
                --seed $SEED \
                --physical-batch-size 40 \
                --optuna-config $OPTUNA_CONFIG \
                --optuna-target-metric MulticlassAccuracy \
                --optuna-direction maximize \
                --experiment-name $EXPERIMENT_NAME \
                --log-dir $LOG_DIR \
                $PRIVACY \
                $USE_STEPS \
                $PRETRAINED \
                $OVERWRITE_EXPERIMENT \
                $OPTUNA_RESUME \
                $OPTUNA_JOURNAL
        fi
    done
done
