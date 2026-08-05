#!/bin/bash
####################################################################
### Experiment: clipping (C) sweep for DP DiseaseTask finetuning
###
### Covers experiment E3 (clipping & long-tail).
###
### Sweeps:
###   - model ∈ {Mistral-7B-Instruct-v0.3, OLMo-2-1124-7B-Instruct}
###   - C ∈ {1, 10, 50, 500}     (max-grad-norm, spans grad-norm tail)
###   - seeds: 42, 43, 44        (3 per cell, matrix floor)
###   = 24 jobs total (2 × 4 × 3)
###
### Each job: ε = 4 (mid-DP regime, fixed for E3), LoRA rank set
### below, full data. Records both pre-clip gradient-norm quantiles
### (--record-gradient-norms) and clipping-bias stats
### (--record-clipping) so we can analyze per-frequency-bin
### clipping bias post-hoc.
####################################################################

set -euo pipefail

# -------------------- Base configuration --------------------
PROJECT="project_462001244"
EXPERIMENT_BASE="DP_Clip_Sweep_$(date +%Y%m%d)"
LOG_BASE="/scratch/$PROJECT/rodr_temp/llm_project/test_repo/dpdl/experiments/$EXPERIMENT_BASE/"
mkdir -p "$LOG_BASE"

# -------------------- Fixed dataset / DP knobs --------------------
DATASET="sarus-tech/medical_v3"
LABEL_FIELD="disease"

# Parallel arrays: MODEL_NAMES[i] is the HF path, MODEL_TAGS[i] is the short
# tag used in the experiment name. Add another entry to both arrays to add
# a model to the sweep.
MODEL_NAMES=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "allenai/OLMo-2-1124-7B-Instruct"
)
MODEL_TAGS=(
    "mistral"
    "olmo2"
)

EPOCHS=25
BATCH_SIZE=1024
PHYSICAL_BATCH_SIZE=12
LORA_RANK=8                # E3 fixes rank=16 (overrides the matrix default of 8).
LEARNING_RATE="0.00026"
TARGET_EPSILON=4            # E3 fixes ε=4.
# δ is derived internally by the trainer (1/N when --target-epsilon is set).
MAX_LENGTH=185
VALIDATION_FREQUENCY=3

# -------------------- Sweep axes --------------------
# Clip values span the per-sample gradient-norm tail observed in the pilot
# (rough quantiles: q50 ≈ 470, q95 ≈ 1500). C=1 ≈ aggressive (modern DP-LoRA),
# C=500 ≈ only the tail gets clipped.
CLIP_NORMS=("1" "10" "50" "500")
SEEDS=(42 43 44)

# -------------------- Helpers --------------------
function is_job_in_queue() {
    local experiment_name=$1
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0
    fi
    return 1
}

# -------------------- Submit grid --------------------
for m in "${!MODEL_NAMES[@]}"; do
    MODEL="${MODEL_NAMES[$m]}"
    MODEL_TAG="${MODEL_TAGS[$m]}"

    for CLIP in "${CLIP_NORMS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            EXPERIMENT_NAME="SWEEP_${MODEL_TAG}_C${CLIP}_eps${TARGET_EPSILON}_seed${SEED}"
            EXPERIMENT_DIR="$LOG_BASE/$EXPERIMENT_NAME"
            mkdir -p "$EXPERIMENT_DIR"

            if [ -f "$EXPERIMENT_DIR/runtime" ]; then
                echo "Already completed: $EXPERIMENT_NAME"
                continue
            fi
            if is_job_in_queue "$EXPERIMENT_NAME"; then
                echo "Already in queue:  $EXPERIMENT_NAME"
                continue
            fi

            sbatch -J "$EXPERIMENT_NAME" run_script_distributed_lumi.sh run.py train \
                --llm \
                --task DiseaseTask \
                --num-workers 7 \
                --model-name "$MODEL" \
                --dataset-name "$DATASET" \
                --dataset-label-field "$LABEL_FIELD" \
                --batch-size "$BATCH_SIZE" \
                --physical-batch-size "$PHYSICAL_BATCH_SIZE" \
                --epochs "$EPOCHS" \
                --max-length "$MAX_LENGTH" \
                --seed "$SEED" \
                --learning-rate "$LEARNING_RATE" \
                --optimizer AdamW \
                --lora-rank "$LORA_RANK" \
                --peft lora \
                --privacy \
                --normalize-clipping \
                --target-epsilon "$TARGET_EPSILON" \
                --max-grad-norm "$CLIP" \
                # --record-gradient-norms triggers RecordBodyAndHeadGradientNormsPerClass
                # + RecordCosineSimilarity callbacks. The former called
                # torch.cuda.empty_cache() inside a per-class loop, a per-rank
                # blocking CUDA sync whose duration drifts across ranks and
                # eventually hangs NCCL. Root-cause fix landed in body_head_gradient.py;
                # keep this flag disabled until that fix is verified on one cell.
                # Per-class norms can be recovered post-hoc from per_sample_eval CSVs
                # by joining gradient norms to disease frequency bins.
                --record-clipping \
                --validation-frequency "$VALIDATION_FREQUENCY" \
                --record-loss-by-step \
                --record-loss-by-epoch \
                --overwrite-experiment \
                --save-model \
                --resume \
                --experiment-name "$EXPERIMENT_NAME" \
                --log-dir "$LOG_BASE"

            if [ $? -eq 0 ]; then
                echo "Submitted: $EXPERIMENT_NAME"
            else
                echo "FAILED to submit: $EXPERIMENT_NAME (exit $?)"
                exit 1
            fi
        done
    done
done

echo
echo "Submitted up to 24 jobs (model × C × seeds at ε=$TARGET_EPSILON) for E3."
echo "Logs will land in: $LOG_BASE"
echo "Check status with: squeue --me | grep SWEEP_"
