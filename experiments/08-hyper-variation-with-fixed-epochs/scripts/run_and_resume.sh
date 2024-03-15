#!/bin/bash
##########################################################
### Experiment: Hyperparameter variation with fixed epochs
##########################################################

set -euo pipefail

if ! python -c "import optuna" &> /dev/null; then
    echo "Error: 'optuna' module not found. Please make sure you have correct environment activated."
    exit 1
fi

EXPERIMENT_BASES=("08-hyper-variation-with-fixed-epochs__batch_size_variation"
                  "08-hyper-variation-with-fixed-epochs__max_grad_norm_variation"
                  "08-hyper-variation-with-fixed-epochs__learning_rate_variation")

# Base configurations
EPOCH_VALUES=("10" "40")
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1" "1.0")
BATCH_SIZES_SUBSET_01=("256" "512" "1024" "2048" "4096" "-1")
BATCH_SIZES_SUBSET_1=("256" "512" "1024" "2048" "4096" "8192" "16384" "32768" "-1")
MAX_GRAD_NORMS=("0.1" "0.18" "0.32" "0.56" "1.0" "1.78" "3.16" "5.62" "10")
LEARNING_RATES=("0.000001" "0.000003" "0.00001" "0.000032" "0.0001" "0.000316" "0.001" "0.003162" "0.01" "0.031623" "0.1")

# Other settings
SEED=42
DEFAULT_N_TRIALS=20
OTHER_SETTINGS="--privacy --use-steps --normalize-clipping --zero-head --peft film"

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

function submit_experiment() {
    local experiment_base=$1
    local model=$2
    local epsilon=$3
    local dataset=$4
    local subset_size=$5
    local epoch=$6
    local hyper_value=$7
    local hyper_name=$8
    local hyper_name_experiment=$(echo $hyper_name | sed 's/_\([a-z]\)/\U\1/g;s/^./\U&/')

    local LOG_DIR="/projappl/$PROJECT/dpdl/experiments/${experiment_base}/data"
    mkdir -p "${LOG_DIR}"

    local JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
    touch $JOB_STATUS_LOG  # Create the file if it doesn't exist

    local EXPERIMENT_NAME
    if [ "$hyper_name" == "batch_size" ] && [ "$hyper_value" == "-1" ]; then
        EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_Epoch${epoch}_FullBatch"
    else
        local hyper_name_experiment=$(echo $hyper_name | sed 's/_\([a-z]\)/\U\1/g;s/^./\U&/')
        EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_Epoch${epoch}_${hyper_name_experiment}${hyper_value}"
    fi

    if is_job_in_queue $EXPERIMENT_NAME; then
        echo "Experiment $EXPERIMENT_NAME is already in the queue."
        return  # Skip to the next iteration
    fi

    # Determine target hyperparameters based on the fixed hyperparameter
    local TARGET_HYPERS=""
    case $hyper_name in
        "batch_size")
            TARGET_HYPERS="--target-hypers max_grad_norm --target-hypers learning_rate"
            ;;
        "max_grad_norm")
            TARGET_HYPERS="--target-hypers batch_size --target-hypers learning_rate"
            ;;
        "learning_rate")
            TARGET_HYPERS="--target-hypers batch_size --target-hypers max_grad_norm"
            ;;
    esac

    local CMD_PREFIX="sbatch -J ${EXPERIMENT_NAME} run8.sh run.py optimize --num-workers 7 --model-name ${model} --dataset-name ${dataset} --subset-size ${subset_size} --num-classes ${NUM_CLASSES} --epochs ${epoch} $TARGET_HYPERS --target-epsilon $epsilon --n-trials ${DEFAULT_N_TRIALS} --seed ${SEED} --physical-batch-size 40 --optuna-config conf/optuna_hypers-subset${subset_size}.conf --optuna-target-metric MulticlassAccuracy --optuna-direction maximize --experiment-name ${EXPERIMENT_NAME} --log-dir ${LOG_DIR} ${OTHER_SETTINGS} --${hyper_name} ${hyper_value}"

    # Check for and handle resuming of experiments
    local EXPERIMENT_DIR="${LOG_DIR}/${EXPERIMENT_NAME}"
    mkdir -p "${EXPERIMENT_DIR}"

    if [ -f "${EXPERIMENT_DIR}/runtime" ]; then
        echo "Experiment ${EXPERIMENT_NAME} has completed. Skipping submission."
        return
    fi

    if [ ! -f "${LOG_DIR}/optuna.journal" ]; then
        touch "${LOG_DIR}/optuna.journal"
    fi


    if is_job_submitted $EXPERIMENT_NAME; then
        local N_TRIALS=$(python bin/get_n_trials.py --optuna-journal "${LOG_DIR}/optuna.journal" --study-name "${EXPERIMENT_NAME}")

        if [ "$N_TRIALS" -eq -1 ]; then
            local REMAINING_TRIALS=$DEFAULT_N_TRIALS
            local OPTUNA_RESUME="--no-optuna-resume"
            local OVERWRITE_EXPERIMENT="--overwrite-experiment"
        else
            local REMAINING_TRIALS=$((DEFAULT_N_TRIALS - N_TRIALS))
            local OPTUNA_RESUME="--optuna-resume"
            local OVERWRITE_EXPERIMENT="--no-overwrite-experiment"
        fi

        if [ "$REMAINING_TRIALS" -le 0 ]; then
            echo "Experiment ${EXPERIMENT_NAME} has already completed the designated number of trials."
            return
        fi
    else
        local REMAINING_TRIALS=$DEFAULT_N_TRIALS
        local OPTUNA_RESUME="--no-optuna-resume"
        local OVERWRITE_EXPERIMENT="--overwrite-experiment"
        local N_TRIALS=-1  # Signal logging that this is a new experiment.
    fi

    # Adjust CMD_PREFIX based on remaining trials and whether to resume
    CMD_PREFIX+=" --n-trials ${REMAINING_TRIALS} ${OPTUNA_RESUME} $OVERWRITE_EXPERIMENT"

    echo "Submitting: ${EXPERIMENT_NAME} with ${REMAINING_TRIALS} trials"
    eval $CMD_PREFIX
    local SBATCH_EXIT_CODE=$?

    if [ $SBATCH_EXIT_CODE -eq 0 ]; then
        echo "Job $EXPERIMENT_NAME submitted successfully."
        if [ "$N_TRIALS" -eq -1 ]; then
            echo $EXPERIMENT_NAME >> $JOB_STATUS_LOG  # Log the submission for new experiments only
        fi
    else
        echo "Submission of $EXPERIMENT_NAME failed with exit code $SBATCH_EXIT_CODE."
        exit 1
    fi
}

# Main loop
for experiment_base in "${EXPERIMENT_BASES[@]}"
do
    for model in "${MODELS[@]}"
    do
        for dataset in "${DATASETS[@]}"
        do
            NUM_CLASSES=10
            [ "$dataset" == "cifar100" ] && NUM_CLASSES=100

            for subset_size in "${SUBSET_SIZES[@]}"
            do

                EPSILONS="0.25 0.5 1 2 4 8"
                [ "$subset_size" == "1.0" ] && EPSILONS="1"

                BATCH_SIZES=("${BATCH_SIZES_SUBSET_01[@]}")
                [ "$subset_size" == "1.0" ] && BATCH_SIZES=("${BATCH_SIZES_SUBSET_1[@]}")

                for epoch in "${EPOCH_VALUES[@]}"
                do
                    for epsilon in $EPSILONS
                    do

                        case $experiment_base in
                            *"batch_size_variation"*)
                                for batch_size in "${BATCH_SIZES[@]}"; do
                                    submit_experiment $experiment_base $model $epsilon $dataset $subset_size $epoch $batch_size "batch_size"
                                done
                                ;;
                            *"max_grad_norm_variation"*)
                                for max_grad_norm in "${MAX_GRAD_NORMS[@]}"; do
                                    submit_experiment $experiment_base $model $epsilon $dataset $subset_size $epoch $max_grad_norm "max_grad_norm"
                                done
                                ;;
                            *"learning_rate_variation"*)
                                for learning_rate in "${LEARNING_RATES[@]}"; do
                                    submit_experiment $experiment_base $model $epsilon $dataset $subset_size $epoch $learning_rate "learning_rate"
                                done
                                ;;
                        esac
                    done
                done
            done
        done
    done
done
