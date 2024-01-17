#!/bin/bash
##################################################
### Experiment: 00-experiment-batch-size-variation
##################################################

# Base configurations
EXPERIMENT_BASE="00-experiment-batch-size-variation"
mkdir -p $LOG_DIR

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1")
PEFT="--peft film"
BATCH_SIZES="256 512 1024 2048 4096 -1"
EPSILONS="0.25 0.5 1 2 4 8"

# Other settings
SEEDS=("43" "44")
N_TRIALS=20
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna.journal"

# these two are for easy requeing: if the jobs get stuck at the
# beginning, we can just `scontrol requeue <jobid>`
OPTUNA_RESUME="--no-optuna-resume"
OVERWRITE_EXPERIMENT="--overwrite-experiment"

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

            for epsilon in $EPSILONS
            do
                for batch_size in $BATCH_SIZES
                do
                    for seed in $SEEDS
                    do
                        LOG_DIR="/projappl/$PROJECT/dpdl/experiments/${EXPERIMENT_BASE}__Extension_Seed${seed}/data"

                        if [ "$batch_size" == "-1" ]; then
                            EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_FullBatch"
                        else
                            EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_BatchSize${batch_size}"
                        fi

                        sbatch -J $EXPERIMENT_NAME run8.sh run.py optimize \
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
                            --seed $seed \
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
                    done
                done
            done
        done
    done
done
