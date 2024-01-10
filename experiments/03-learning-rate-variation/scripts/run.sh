#!/bin/bash
##################################################
### Experiment: Learning Rate Variation
##################################################

# Base configurations
EXPERIMENT_BASE="03-learning-rate-variation"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# We keep track of submitted jobs here
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG  # Create the file if it doesn't exist

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1" "1.0")
LEARNING_RATES=("0.000001" "0.000003" "0.000010" "0.000032" "0.000100" "0.000316" "0.001000" "0.003162" "0.010000" "0.031623" "0.100000")

# Other settings
SEED=42
N_TRIALS=20
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna.journal"

# these two are for easy requeuing: if the jobs get stuck at the
# beginning, we can just `scontrol requeue <jobid>`
OPTUNA_RESUME="--no-optuna-resume"
OVERWRITE_EXPERIMENT="--overwrite-experiment"

# Function to check if a job has been submitted
function is_job_submitted() {
    grep -Fxq "$1" $JOB_STATUS_LOG
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
                    EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_LR${learning_rate}"

                    # Check if this job has already been submitted
                    if is_job_submitted $EXPERIMENT_NAME; then
                        echo "!!! Already submitted: $EXPERIMENT_NAME"
                    else
                        # Submit the job and capture its success or failure
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
                            --n-trials $N_TRIALS \
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
                            echo $EXPERIMENT_NAME >> $JOB_STATUS_LOG  # Log the submission
                        else
                            echo "Submission of $EXPERIMENT_NAME failed with exit code $SBATCH_EXIT_CODE."
                            exit
                        fi
                    fi
                done
            done
        done
    done
done

