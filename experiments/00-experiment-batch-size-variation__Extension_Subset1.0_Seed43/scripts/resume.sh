#!/bin/bash
########################$$$$$$$$$$$##########################
### Resume experiments in: 00-experiment-batch-size-variation
#############################################################

# Experiment name to be resumed
RESUME_EXPERIMENT=$1

# We also have an option to set the number of trials to run
N_TRIALS=${2:-20}

# Base configurations
EXPERIMENT_BASE="00-experiment-batch-size-variation__Extension_Subset1.0_Seed43"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1" "1.0")
PEFT="--peft film"

# Other settings
SEED=43
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
            if [ "$subset_size" == "0.1" ]; then
                BATCH_SIZES="256 512 1024 2048 4096 -1"
                EPSILONS="0.25 0.5 1 2 4 8"
            else
                BATCH_SIZES="256 512 1024 2048 4096 8192 16384 32768 -1"
                EPSILONS="1"
            fi

            OPTUNA_CONFIG="conf/optuna_hypers-subset${subset_size}.conf"

            for epsilon in $EPSILONS
            do
                for batch_size in $BATCH_SIZES
                do
                    if [ "$batch_size" == "-1" ]; then
                        EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_Seed${SEED}_FullBatch"
                    else
                        EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_Seed${SEED}_BatchSize${batch_size}"
                    fi

                    if [ "$RESUME_EXPERIMENT" == "$EXPERIMENT_NAME" ]; then
                        echo sbatch -J $EXPERIMENT_NAME run8.sh run.py optimize \
                            --num-workers 7 \
                            --model-name $model \
                            --dataset-name $dataset \
                            --subset-size $subset_size \
                            --num-classes $NUM_CLASSES \
                            --batch-size $batch_size \
                            --target-hypers epochs \
                            --target-hypers learning_rate \
                            --target-hypers max_grad_norm \
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
