##########################################
### Experiment: 01-film-vs-lora-evaluation
##########################################

# Base configurations
EXPERIMENT_BASE="01-film-vs-lora-evaluation"
LOG_DIR="/projappl/$PROJECT/experiment-data/$EXPERIMENT_BASE"
mkdir -p $LOG_DIR

# Experiment parameters
EPSILONS="0.25 1.0 4.0"
PEFT_METHODS="film lora"
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1" "1.0")

# Other settings
SEED=42
N_TRIALS=50
PEFT="--peft"
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"

# Loop over configurations
for model in "${MODELS[@]}"
do
    for dataset in "${DATASETS[@]}"
    do
        for subset_size in "${SUBSET_SIZES[@]}"
        do
            for peft_method in $PEFT_METHODS
            do
                for epsilon in $EPSILONS
                do
                    EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_${peft_method}_Epsilon${epsilon}"
                    OPTUNA_CONFIG="conf/optuna_hypers-subset${subset_size}.conf"

                    sbatch -J $EXPERIMENT_NAME run8.sh run.py optimize \
                        --num-workers 7 \
                        --model-name $model \
                        --dataset-name $dataset \
                        --subset-size $subset_size \
                        --num-classes 10 \
                        --target-hypers epochs \
                        --target-hypers batch_size \
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
                        $PEFT $peft_method \
                        $PRIVACY \
                        --overwrite-experiment $USE_STEPS $NORMALIZE_CLIPPING
                done
            done
        done
    done
done
