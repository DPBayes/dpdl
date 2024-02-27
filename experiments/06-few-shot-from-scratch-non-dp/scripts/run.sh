#!/bin/bash
##################################################
### Experiment: Few-shot from scratch without DP
##################################################

set -euo pipefail

# Base configurations
EXPERIMENT_BASE="06-few-shot-from-scratch-non-dp"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# We keep track of submitted jobs here
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG  # Create the file if it doesn't exist

# Experiment parameters
MODELS=("wrn-16-4" "koskela-net")
ALL_SHOTS=(1 5 10 25 50 100 250 500 1000 1500 2000)
DATASET="cifar10"
NUM_CLASSES=10

# Other settings
SEED=42
N_TRIALS=40
PRIVACY="--no-privacy"
USE_STEPS="--use-steps"
PRETRAINED="--no-pretrained"
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna.journal"

# these two are for easy requeuing: if the jobs get stuck at the
# beginning, we can just `scontrol requeue <jobid>`
OPTUNA_RESUME="--no-optuna-resume"
OVERWRITE_EXPERIMENT="--overwrite-experiment"

# Function to check if a job has been submitted
function is_job_submitted() {
    grep -Fxq "$1" $JOB_STATUS_LOG
}

for model in "${MODELS[@]}"
do
    for shots in "${ALL_SHOTS[@]}"
    do
        EXPERIMENT_NAME="${model}_${DATASET}_32x32_Shots${shots}"

        OPTUNA_CONFIG="conf/optuna_hypers-shots${shots}.conf"

        # Check if this job has already been submitted
        if is_job_submitted $EXPERIMENT_NAME; then
            echo "!!! Already submitted: $EXPERIMENT_NAME"
        else
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

            SBATCH_EXIT_CODE=$?

            if [ $SBATCH_EXIT_CODE -eq 0 ]; then
                echo "Job $EXPERIMENT_NAME submitted successfully."
                echo $EXPERIMENT_NAME >> $JOB_STATUS_LOG  # Log the submission
            else
                echo "Submission of $EXPERIMENT_NAME failed with exit code $SBATCH_EXIT_CODE."
                exit
            fi
        fi
    done
done
