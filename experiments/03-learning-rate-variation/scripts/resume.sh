#!/bin/bash
##########################################################
### Resume experiments in: Learning Rate Variation
##########################################################

# Base configurations
EXPERIMENT_BASE="03-learning-rate-variation"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# Experiment name to be resumed
RESUME_EXPERIMENT=$1

# We also have an option to set the number of trials to run
N_TRIALS=${2:-20}

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1" "1.0")
LEARNING_RATES=("0.000001" "0.000003" "0.00001" "0.000032" "0.0001" "0.000316" "0.001" "0.003162" "0.01" "0.031623" "0.1")

# Other settings
SEED=42
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna.journal"

# we want to resume the optuna studies, so let's signal that
OPTUNA_RESUME="--optuna-resume"

# as we are resuming, we don't want to overwrite the experiment!
OVERWRITE_EXPERIMENT="--no-overwrite-experiment"

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

                    if [ "$RESUME_EXPERIMENT" == "$EXPERIMENT_NAME" ]; then
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
                    fi
                done
            done
        done
    done
done
