#!/bin/bash
##########################################################
### Pre-experiment: DPDL Benchmark without DP
##########################################################

set -euo pipefail

if ! python -c 'import optuna' &> /dev/null; then
    echo "Error: 'optuna' module not found. Please make sure you have correct environment activated."
    exit 1
fi

# Base configurations
EXPERIMENT_BASE='09-dpdl-benchmark-without-dp'
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p "$LOG_DIR"

# We keep track of submitted jobs here
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch "$JOB_STATUS_LOG"  # Create the file if it doesn't exist

# Function to check if a job has been submitted
is_job_submitted() {
    grep -Fxq "$1" "$JOB_STATUS_LOG"
}

# Function to check if a job is already in queue
is_job_in_queue() {
    local experiment_name=$1
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0  # Job is in queue
    else
        return 1  # Job is not in queue
    fi
}

# Define the datasets and labels that we will be using
declare -A DATASETS_LABELS=(
    ['dpdl-benchmark/omniglot']='label'
    ['dpdl-benchmark/fgvc_aircraft']='family variant'
    ['dpdl-benchmark/caltech_birds2011']='label'
    ['dpdl-benchmark/dtd']='label'
    ['dpdl-benchmark/quickdraw_bitmap']='label'
    ['dpdl-benchmark/oxford_flowers102']='label'
    ['dpdl-benchmark/gtsrb']='label'
    ['dpdl-benchmark/caltech101']='label'
    ['cifar100']='fine_label'
    ['dpdl-benchmark/clevr']='label_count label_distance'
    ['dpdl-benchmark/dsprites']='label_x_position_16 label_orientation_16'
    ['dpdl-benchmark/eurosat']='label'
    ['dpdl-benchmark/kitti']='label_distance'
    ['dpdl-benchmark/oxford_iiit_pet']='label'
    ['dpdl-benchmark/patch_camelyon']='label'
    ['dpdl-benchmark/resisc45']='label'
    ['dpdl-benchmark/smallnorb']='label_elevation label_azimuth'
    ['dpdl-benchmark/sun397']='label'
    ['dpdl-benchmark/svhn_cropped']='label'
    ['dpdl-benchmark/malaria']='label'
    ['dpdl-benchmark/plant_village']='label'
    ['dpdl-benchmark/cassava']='label'
    ['dpdl-benchmark/colorectal_histology']='label'
    ['dpdl-benchmark/uc_merced']='label'
)

SEED=42
ZERO_HEAD='--zero-head'
CACHE_FEATURES='--cache-features'
PEFT='--peft head-only'
MODEL='vit_base_patch16_224.augreg_in21k'
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna.journal"
OPTUNA_CONFIG='conf/optuna_hypers-dpdl-benchmark.conf'
PRIVACY='--no-privacy'
USE_STEPS="--use-steps"

DEFAULT_N_TRIALS=20

for SHOTS in 100 500; do
    for dataset in "${!DATASETS_LABELS[@]}"; do
        for label in ${DATASETS_LABELS[$dataset]}; do
            EXPERIMENT_NAME="${MODEL}_${dataset//\//_}_Shots${SHOTS}_Epsilon-1_Seed${SEED}_${label}"

            if is_job_in_queue "$EXPERIMENT_NAME"; then
                echo "Experiment $EXPERIMENT_NAME is already in the queue."
                continue
            fi

            EXPERIMENT_DIR="${LOG_DIR}/${EXPERIMENT_NAME}"
            mkdir -p "$EXPERIMENT_DIR"

            if [ -f "${EXPERIMENT_DIR}/runtime" ]; then
                echo "Experiment ${EXPERIMENT_NAME} has completed. Skipping submission."
                continue
            fi

            if is_job_submitted "$EXPERIMENT_NAME"; then
                N_TRIALS=$(python bin/get_n_trials.py --optuna-journal "$OPTUNA_JOURNAL" --study-name "${EXPERIMENT_NAME}")

                if [ "$N_TRIALS" -eq "-1" ]; then
                    REMAINING_TRIALS=$DEFAULT_N_TRIALS
                    OPTUNA_RESUME='--no-optuna-resume'
                    OVERWRITE_EXPERIMENT='--overwrite-experiment'
                else
                    REMAINING_TRIALS=$((DEFAULT_N_TRIALS - N_TRIALS))
                    OPTUNA_RESUME='--optuna-resume'
                    OVERWRITE_EXPERIMENT='--no-overwrite-experiment'
                fi

                if [ "$REMAINING_TRIALS" -le 0 ]; then
                    echo "Experiment ${EXPERIMENT_NAME} has already completed the designated number of trials. Running last training round."
                    REMAINING_TRIALS=0
                fi
            else
                REMAINING_TRIALS=$DEFAULT_N_TRIALS
                OPTUNA_RESUME='--no-optuna-resume'
                OVERWRITE_EXPERIMENT='--overwrite-experiment'
                N_TRIALS=-1  # Signal logging that this is a new experiment.
            fi

            echo sbatch -J "$EXPERIMENT_NAME" run8.sh run.py optimize \
                --num-workers 7 \
                --dataset-name "$dataset" \
                --dataset-label-field "$label" \
                --model-name "$MODEL" \
                --shots "$SHOTS" \
                --target-hypers learning_rate \
                --target-hypers batch_size \
                --target-hypers epochs \
                --n-trials "$REMAINING_TRIALS" \
                --seed "$SEED" \
                --physical-batch-size 40 \
                --optuna-config "$OPTUNA_CONFIG" \
                --optuna-target-metric MulticlassAccuracy \
                --optuna-direction maximize \
                --experiment-name "$EXPERIMENT_NAME" \
                --log-dir "$LOG_DIR" \
                $ZERO_HEAD \
                $PEFT \
                $PRIVACY \
                $USE_STEPS \
                $OVERWRITE_EXPERIMENT \
                $OPTUNA_RESUME \
                $OPTUNA_JOURNAL

            SBATCH_EXIT_CODE=$?

            if [ $SBATCH_EXIT_CODE -eq 0 ]; then
                echo "Job $EXPERIMENT_NAME submitted successfully."
                if [ "$N_TRIALS" -eq -1 ]; then
                    echo "$EXPERIMENT_NAME" >> "$JOB_STATUS_LOG"  # Log the submission for new experiments only
                fi
            else
                echo "Submission of $EXPERIMENT_NAME failed with exit code $SBATCH_EXIT_CODE."
                exit 1
            fi
        done
    done
done
