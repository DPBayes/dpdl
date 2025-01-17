#!/bin/bash
##################################################
### Experiment: Privacy assistant real-world evaluation
##################################################
#
set -euo pipefail

# Base configurations
EXPERIMENT_BASE="22-privacy-assistant-real-world-evaluation"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# We keep track of submitted jobs here
JOB_STATUS_LOG="$LOG_DIR/submitted_jobs.log"
touch $JOB_STATUS_LOG  # Create the file if it doesn't exist

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k")
DATASETS=(
#    "cifar100"
    "dpdl-benchmark/svhn_cropped"
#    "dpdl-benchmark/sun397"
#    "dpdl-benchmark/patch_camelyon"
#    "dpdl-benchmark/cassava"
)
EPOCHS=40
EPSILONS="0.01 0.010773699258659447 0.011607259571603908 0.01250531238416568 0.013472847476259053 0.014515240686700398 0.015638283782556744 0.016848216639483767 0.018151761911853994 0.019556162385310417 0.021069221219284253 0.022699345303073465 0.02445559196637773 0.02634771930382417 0.02838624039309777 0.030582481707924616 0.03294864605046334 0.03549788035277094 0.03824434872406301 0.04120331114963508 0.04439120827871378 0.04782575277233758 0.05152602771881634 0.055512592663587754 0.05980759784259593 0.06443490725389782 0.06942023125131089 0.07479126939682151 0.0805778643654735 0.08681216775786633 0.09352881874155437 0.10076513651391776 0.1085613276558714 0.1169607095285147 0.1260099509539641 0.13575933151764366 0.1462630209327739 0.15757938001927174 0.16977128496936428 0.18290647670161017 0.1970579372444148 0.2123042952403112 0.2287302628240756 0.2464271063020724 0.2654931532480229 0.2860343388327383 0.30816479442334166 0.33200748172236977 0.3576948759701685 0.3853697020066086 0.4151857272818411 0.44730861622223533 0.481916850698548 0.519202721710644 0.5593733977987932 0.6026520761178672 0.6492792225700643 0.6995139078866082 0.7536352470819924 0.8119439502786887 0.8747639935190532 0.9424444188478197 1.0153612736668485 1.0939197001376042 1.1785561862405467 1.2697409909988278 1.367980757341359 1.4738213271228982 1.5878507739420444 1.7107026706081225 1.8430596094117477 1.9856569947584508 2.139287129238106 2.304803615833227 2.4831261007258063 2.675245383054753 2.8822289200149087 3.105226757885143 3.345477921939666 3.6043173007462914 3.8831830631023725 4.183624648818498 4.507311377748522 4.856041724899652 5.2317533131570695 5.6365336791449465 6.072631872041288 6.5424709497922935 7.048661442157817 7.5940158553917145 8.181564299148176 8.814571322440727 9.496554052198045 10.23130173519855 11.022896791962967 11.875737499585098 12.794562429531425 13.784476776191466 14.850980722466229 16.0"
OPTUNA_CONFIG="conf/optuna_hypers-privacy-assistant-real-world.conf"

declare -A DATASET_LABEL_FIELDS
DATASET_LABEL_FIELDS=(
    ["cifar100"]="fine_label"
    ["dpdl-benchmark/svhn_cropped"]="label"
    ["dpdl-benchmark/sun397"]="label"
    ["dpdl-benchmark/patch_camelyon"]="label"
    ["dpdl-benchmark/cassava"]="label"
)

declare -A SUBSET_SIZES
SUBSET_SIZES=(
    ["cifar100"]="0.1"
    ["dpdl-benchmark/svhn_cropped"]="0.1"
    ["dpdl-benchmark/sun397"]="0.1"
    ["dpdl-benchmark/patch_camelyon"]="0.02"
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
            rounded_epsilon=$(printf "%.4f" $epsilon)

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
            echo sbatch -J $EXPERIMENT_NAME run8-rocm.sh run.py optimize \
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
            exit

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
