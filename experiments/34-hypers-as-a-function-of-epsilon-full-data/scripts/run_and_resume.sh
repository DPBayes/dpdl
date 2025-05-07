#!/bin/bash
####################################################################
### Experiment: Hypers as a function of epsilon
####################################################################

set -euo pipefail

# Base configurations
EXPERIMENT_BASE="24-hypers-as-a-function-of-epsilon"
LOG_BASE="/scratch/$PROJECT/experiments/$EXPERIMENT_BASE/"

# Experiment parameters
MODELS=(
    "vit_base_patch16_224.augreg_in21k"
#    "vit_tiny_patch16_224.augreg_in21k"
)

DATASETS=(
    "cifar100"
#    "dpdl-benchmark/sun397"
#    "dpdl-benchmark/patch_camelyon"
#    "dpdl-benchmark/cassava"
#    "dpdl-benchmark/svhn_cropped"
#    "dpdl-benchmark/svhn_cropped_balanced"
#    "dpdl-benchmark/hard-imagenet-medium"
#    "dpdl-benchmark/sun100"
#    "dpdl-benchmark/sun199"
#    "dpdl-benchmark/imagenet-mini-class-hard"
#    "dpdl-benchmark/imagenet-mini-sample-hard"
#    "dpdl-benchmark/places365-mini-class-hard"
#    "dpdl-benchmark/places365-mini-sample-hard"
#    "dpdl-benchmark/places365-mini-class-easy"
#    "dpdl-benchmark/places365-mini-sample-easy"
#    "dpdl-benchmark/cifar10_10pct_plus_cifar100_humans"
#    "dpdl-benchmark/imagenet397"
#    "dpdl-benchmark/places100-easy"
#    "dpdl-benchmark/places100-hard"
)

EPOCHS=40
LEARNING_RATES="1e-05 2.6826957952797274e-05 7.196856730011514e-05 0.00019306977288832496 0.0005179474679231213 0.0013894954943731374 0.003727593720314938 0.01"
BATCH_SIZES="256 1024 4096 16834 -1"
MAX_GRAD_NORMS="1e-5 1 10 50"
EPSILONS="1 3 8"
SEEDS="42"

declare -A DATASET_LABEL_FIELDS
DATASET_LABEL_FIELDS=(
    ["cifar100"]="fine_label"
    ["dpdl-benchmark/sun397"]="label"
    ["dpdl-benchmark/patch_camelyon"]="label"
    ["dpdl-benchmark/cassava"]="label"
    ["dpdl-benchmark/svhn_cropped"]="label"
    ["dpdl-benchmark/svhn_cropped_balanced"]="label"
    ["dpdl-benchmark/hard-imagenet-medium"]="label"
    ["dpdl-benchmark/sun100"]="label"
    ["dpdl-benchmark/sun199"]="label"
    ["dpdl-benchmark/sun199"]="label"
    ["dpdl-benchmark/imagenet-mini-class-hard"]="label"
    ["dpdl-benchmark/imagenet-mini-sample-hard"]="label"
    ["dpdl-benchmark/places365-mini-class-hard"]="label"
    ["dpdl-benchmark/places365-mini-sample-hard"]="label"
    ["dpdl-benchmark/places365-mini-class-easy"]="label"
    ["dpdl-benchmark/places365-mini-sample-easy"]="label"
    ["dpdl-benchmark/cifar10_10pct_plus_cifar100_humans"]="label"
    ["dpdl-benchmark/imagenet397"]="label"
    ["dpdl-benchmark/places100-easy"]="label"
    ["dpdl-benchmark/places100-hard"]="label"
)

declare -A SUBSET_SIZES
SUBSET_SIZES=(
    ["cifar100"]="1.0"
    ["dpdl-benchmark/sun397"]="0.1"
    ["dpdl-benchmark/patch_camelyon"]="0.02"
    ["dpdl-benchmark/cassava"]="1.0"
    ["dpdl-benchmark/svhn_cropped"]="0.1"
    ["dpdl-benchmark/svhn_cropped_balanced"]="0.1"
    ["dpdl-benchmark/hard-imagenet-medium"]="1.0"
    ["dpdl-benchmark/sun100"]="0.1"
    ["dpdl-benchmark/sun199"]="0.1"
    ["dpdl-benchmark/imagenet-mini-class-hard"]="1.0"
    ["dpdl-benchmark/imagenet-mini-sample-hard"]="1.0"
    ["dpdl-benchmark/places365-mini-class-hard"]="1.0"
    ["dpdl-benchmark/places365-mini-sample-hard"]="1.0"
    ["dpdl-benchmark/places365-mini-class-easy"]="1.0"
    ["dpdl-benchmark/places365-mini-sample-easy"]="1.0"
    ["dpdl-benchmark/cifar10_10pct_plus_cifar100_humans"]="1.0"
    ["dpdl-benchmark/imagenet397"]="0.1"
    ["dpdl-benchmark/places100-easy"]="0.1"
    ["dpdl-benchmark/places100-hard"]="0.1"
)

# Other settings
OVERWRITE_EXPERIMENT="--overwrite-experiment" # These are fast to run, overwrite by default
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
PEFT="--peft film"
RECORD_LOSS="--record-loss-by-step --record-loss-by-epoch"
RECORD_GRADIENTS=""
EVALUATION_MODE="--evaluation-mode"

# Function to check if a job is already in queue
function is_job_in_queue() {
    local experiment_name=$1
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0  # Job is in queue
    else
        return 1  # Job is not in queue
    fi
}

for seed in $SEEDS
do
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

            LOG_DIR="${LOG_BASE}/data_${clean_dataset_name}_Epochs${EPOCHS}_Seed${seed}"
            mkdir -p $LOG_DIR

            if  [ "$clean_dataset_name" = "sun100" ]; then
                BATCH_SIZES="192 512 1024 -1"
            fi

            if  [ "$clean_dataset_name" = "sun199" ]; then
                BATCH_SIZES="192 512 1024 2048 -1"
            fi

            for batch_size in $BATCH_SIZES
            do
                for learning_rate in $LEARNING_RATES
                do
                    rounded_learning_rate=$(printf "%.5f" $learning_rate)

                    for max_grad_norm in $MAX_GRAD_NORMS
                    do
                        rounded_max_grad_norm=$(printf "%.5f" $max_grad_norm)

                        for epsilon in $EPSILONS
                        do
                            rounded_epsilon=$(printf "%.2f" $epsilon)

                            EXPERIMENT_NAME="${model}_${clean_dataset_name}_Epochs${EPOCHS}_Subset${subset_size}_Epsilon${rounded_epsilon}_BatchSize${batch_size}_LearningRate${rounded_learning_rate}_MaxGradNorm${rounded_max_grad_norm}_Seed${seed}"

                            EXPERIMENT_DIR="$LOG_DIR/$EXPERIMENT_NAME"
                            mkdir -p "$EXPERIMENT_DIR"

                            # If `runtime` file exists, then the job has been completed
                            if [ -f "$EXPERIMENT_DIR/runtime" ]; then
                                echo "Experiment $EXPERIMENT_NAME has completed. Skipping submission."
                                continue
                            fi

                            if is_job_in_queue $EXPERIMENT_NAME; then
                                echo "Experiment $EXPERIMENT_NAME is already in the queue."
                                continue  # Skip to the next iteration
                            fi

                            # Submit the job
                            sbatch -J $EXPERIMENT_NAME run8.sh run.py train \
                                --num-workers 7 \
                                --model-name $model \
                                --dataset-name $dataset \
                                --subset-size $subset_size \
                                --dataset-label-field $label_field \
                                --batch-size $batch_size \
                                --learning-rate $learning_rate \
                                --max-grad-norm $max_grad_norm \
                                --epochs $EPOCHS \
                                --target-epsilon $epsilon \
                                --seed $seed \
                                --physical-batch-size 40 \
                                --experiment-name $EXPERIMENT_NAME \
                                --log-dir $LOG_DIR \
                                $ZERO_HEAD \
                                $PEFT \
                                $PRIVACY \
                                $USE_STEPS \
                                $NORMALIZE_CLIPPING \
                                $OVERWRITE_EXPERIMENT \
                                $RECORD_LOSS \
                                $RECORD_GRADIENTS \
                                $EVALUATION_MODE

                            SBATCH_EXIT_CODE=$?

                            if [ $SBATCH_EXIT_CODE -eq 0 ]; then
                                echo "Job $EXPERIMENT_NAME submitted successfully."
                            else
                                echo "Submission of $EXPERIMENT_NAME failed with exit code $SBATCH_EXIT_CODE."
                                exit 1
                            fi
                        done
                    done
                done
            done
        done
    done
done
