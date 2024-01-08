#!/bin/bash
##################################################
### Experiment: Maximum Gradient Norm Variation
##################################################

# Base configurations
EXPERIMENT_BASE="02-maximum-gradient-norm-variation"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/$EXPERIMENT_BASE/data"
mkdir -p $LOG_DIR

# Experiment parameters
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1" "1.0")
MAX_GRAD_NORMS=("0.1" "0.18" "0.32" "0.56" "1.0" "1.78" "3.16" "5.62" "10")

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

            for max_grad_norm in "${MAX_GRAD_NORMS[@]}"
            do
                for epsilon in $EPSILONS
                do

                    EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_Epsilon${epsilon}_MaxGradNorm${max_grad_norm}"

                    sbatch -J $EXPERIMENT_NAME run8.sh run.py optimize \
                        --num-workers 7 \
                        --model-name $model \
                        --dataset-name $dataset \
                        --subset-size $subset_size \
                        --num-classes $NUM_CLASSES \
                        --target-hypers epochs \
                        --target-hypers learning_rate \
                        --target-hypers batch_size \
                        --max-grad-norm $max_grad_norm \
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
                done
            done
        done
    done
done
