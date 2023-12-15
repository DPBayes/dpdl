##########################################
### Experiment: 01-film-vs-lora-evaluation
### Extension: Subset: 0.1, Trials: 50
##########################################

# Base configurations
EXPERIMENT_BASE="01-film-vs-lora-evaluation__Extension_Subset0.1_Trials50"
LOG_DIR="/projappl/$PROJECT/dpdl/experiments/01-film-vs-lora-evaluation__Extension_Subset0.1_Trials50/data"
mkdir -p $LOG_DIR

# Experiment parameters
EPSILONS="0.25 1.0 4.0"
PEFT_METHODS="film lora"
MODELS=("vit_base_patch16_224.augreg_in21k" "resnetv2_50x1_bit.goog_in21k")
DATASETS=("cifar10" "cifar100")
SUBSET_SIZES=("0.1")

# Other settings
SEED=42
N_TRIALS=50
PEFT="--peft"
PRIVACY="--privacy"
USE_STEPS="--use-steps"
NORMALIZE_CLIPPING="--normalize-clipping"
ZERO_HEAD="--zero-head"
OPTUNA_JOURNAL="--optuna-journal $LOG_DIR/optuna_journal.log"
OPTUNA_RESUME="--no-optuna-resume" # should we resume the study?
OVERWRITE_EXPERIMENT="--overwrite-experiment"

# Loop over configurations
for model in "${MODELS[@]}"
do
    for dataset in "${DATASETS[@]}"
    do
        for subset_size in "${SUBSET_SIZES[@]}"
        do
            # Skip 100% subset for CIFAR-10
            if [ "$dataset" = "cifar10" ] && [ "$subset_size" = "1.0" ]; then
                continue
            fi

            # Set number of classes based on the dataset
            if [ "$dataset" = "cifar10" ]; then
                NUM_CLASSES=10
            elif [ "$dataset" = "cifar100" ]; then
                NUM_CLASSES=100
            fi

            for peft_method in $PEFT_METHODS
            do
                for epsilon in $EPSILONS
                do
                    EXPERIMENT_NAME="${model}_${dataset}_Subset${subset_size}_${peft_method}_Epsilon${epsilon}__Extension_Subset0.1_Trials50"
                    OPTUNA_CONFIG="conf/optuna_hypers-subset${subset_size}.conf"

                    echo sbatch -J $EXPERIMENT_NAME run8.sh run.py optimize \
                        --num-workers 7 \
                        --model-name $model \
                        --dataset-name $dataset \
                        --subset-size $subset_size \
                        --num-classes $NUM_CLASSES \
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
                        --log-dir $LOG_DIR \
                        $PEFT $peft_method \
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
