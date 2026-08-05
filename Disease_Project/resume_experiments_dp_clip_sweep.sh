#!/bin/bash
####################################################################
### Resume helper for the clip sweep (run_experiments_dp_clip_sweep.sh).
###
### For each cell it reports FINISHED / IN QUEUE / RESUMABLE / NOT STARTED
### and resubmits unfinished, non-queued cells with --resume. The DP
### trainer restores weights + optimizer + privacy accountant + epoch, so
### the continued run consumes exactly the remaining privacy budget.
###
### IMPORTANT: point LOG_BASE at the SAME directory the original sweep
### wrote to (the date is baked into the dir name — it will NOT match
### today's date if the sweep ran earlier).
###
### Usage:
###   LOG_BASE=/scratch/.../experiments/DP_Clip_Sweep_20260724/ \
###       ./resume_experiments_dp_clip_sweep.sh
####################################################################

set -euo pipefail

PROJECT="project_462001244"
: "${LOG_BASE:=/scratch/$PROJECT/rodr_temp/llm_project/test_repo/dpdl/experiments/DP_Clip_Sweep_REPLACE_ME/}"

if [ ! -d "$LOG_BASE" ]; then
    echo "ERROR: LOG_BASE does not exist: $LOG_BASE" >&2
    echo "Set it to the original sweep directory (see header)." >&2
    exit 1
fi

# -------------------- Fixed knobs (must match the original sweep) --------------------
DATASET="sarus-tech/medical_v3"
LABEL_FIELD="disease"

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
LORA_RANK=8
LEARNING_RATE="0.00026"
TARGET_EPSILON=4
MAX_LENGTH=185
VALIDATION_FREQUENCY=3

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

function latest_checkpoint_epoch() {
    local ckpt_dir=$1
    local latest=-1
    if [ -d "$ckpt_dir" ]; then
        for d in "$ckpt_dir"/checkpoint_epoch_*; do
            [ -d "$d" ] || continue
            [ -f "$d/training_state.pt" ] || continue
            local n="${d##*checkpoint_epoch_}"
            if [ "$n" -gt "$latest" ] 2>/dev/null; then
                latest="$n"
            fi
        done
    fi
    echo "$latest"
}

# -------------------- Scan + resubmit --------------------
n_finished=0; n_queued=0; n_resumed=0; n_fresh=0

for m in "${!MODEL_NAMES[@]}"; do
    MODEL="${MODEL_NAMES[$m]}"
    MODEL_TAG="${MODEL_TAGS[$m]}"

    for CLIP in "${CLIP_NORMS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            EXPERIMENT_NAME="SWEEP_${MODEL_TAG}_C${CLIP}_eps${TARGET_EPSILON}_seed${SEED}"
            EXPERIMENT_DIR="$LOG_BASE/$EXPERIMENT_NAME"

            if [ -f "$EXPERIMENT_DIR/runtime" ]; then
                echo "FINISHED    : $EXPERIMENT_NAME"
                n_finished=$((n_finished + 1))
                continue
            fi

            if is_job_in_queue "$EXPERIMENT_NAME"; then
                echo "IN QUEUE    : $EXPERIMENT_NAME"
                n_queued=$((n_queued + 1))
                continue
            fi

            last_epoch=$(latest_checkpoint_epoch "$EXPERIMENT_DIR/checkpoints")
            if [ "$last_epoch" -ge 0 ]; then
                echo "RESUMABLE   : $EXPERIMENT_NAME (from epoch $((last_epoch + 1)) of $EPOCHS)"
                n_resumed=$((n_resumed + 1))
            else
                echo "NOT STARTED : $EXPERIMENT_NAME (no checkpoint, restarts fresh)"
                n_fresh=$((n_fresh + 1))
            fi

            mkdir -p "$EXPERIMENT_DIR"

            # All clip-sweep cells are DP (ε=4 fixed), so every one is resumable
            # once it has run ≥1 epoch. No --overwrite-experiment (would delete
            # checkpoints). --record-gradient-norms stays OFF (see main script).
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
                --record-clipping \
                --validation-frequency "$VALIDATION_FREQUENCY" \
                --record-loss-by-step \
                --record-loss-by-epoch \
                --save-model \
                --resume \
                --experiment-name "$EXPERIMENT_NAME" \
                --log-dir "$LOG_BASE"

            if [ $? -ne 0 ]; then
                echo "FAILED to submit: $EXPERIMENT_NAME" >&2
                exit 1
            fi
        done
    done
done

echo
echo "Summary: finished=$n_finished  queued=$n_queued  resumed=$n_resumed  fresh=$n_fresh"
echo "Resubmitted $((n_resumed + n_fresh)) cell(s) with --resume."
echo "Logs: $LOG_BASE"
echo "Watch: squeue --me | grep SWEEP_"
