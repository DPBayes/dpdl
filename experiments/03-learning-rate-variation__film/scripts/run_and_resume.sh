#!/bin/bash
##################################################
### Experiment: Learning Rate Variation
##################################################
#
set -euo pipefail

# Base configurations
EXPERIMENT_BASE="03-learning-rate-variation__film"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# We keep track of submitted jobs here
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG  # Create the file if it doesn't exist

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1" "1.0")
LEARNING_RATES=("0.000001" "0.000003" "0.00001" "0.000032" "0.0001" "0.000316" "0.001" "0.003162" "0.01" "0.031623" "0.1")

# Other settings
SEED=42
DEFAULT_N_TRIALS=20
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna.journal"
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
        # Adjust number of classes based on the dataset
        if [ "$dataset" == "cifar100" ]; then
            NUM_CLASSES=100
        else
            NUM_CLASSES=10
        fi

        for subset_size in "${SUBSET_SIZES[@]}"
        do
            OPTUNA_CONFIG="conf/optuna_hypers-subset${subset_size}.conf"

            if [ "$subset_size" == "1.0" ]; then
                EPSILONS="1"
            else
                EPSILONS="0.25 0.5 1 2 4 8"
            fi

            for learning_rate in "${LEARNING_RATES[@]}"
            do
                for epsilon in $EPSILONS
                do
                    EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_LearningRate${learning_rate}"
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
                    sbatch -J $EXPERIMENT_NAME run8.sh run.py optimize \
                        --num-workers 7 \
                        --model-name $model \
                        --dataset-name $dataset \
                        --subset-size $subset_size \
                        --num-classes $NUM_CLASSES \
                        --target-hypers epochs \
                        --target-hypers max_grad_norm \
                        --target-hypers batch_size \
                        --learning-rate $learning_rate \
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
                        $OPTUNA_JOURNAL

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
    done
done
