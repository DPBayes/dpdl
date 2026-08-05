#!/bin/bash
####################################################################
### Resume helper for the ε sweep (run_experiments_dp_eps_sweep.sh).
###
### For each cell in the sweep grid it reports one of:
###   FINISHED     - a `runtime` file exists (training completed)
###   IN QUEUE     - a job with this name is already queued/running
###   RESUMABLE    - a checkpoint_epoch_<N> exists; will resubmit with
###                  --resume to continue from epoch N+1 (DP cells only)
###   NOT STARTED  - no checkpoint; will (re)submit fresh
###
### Unfinished, non-queued cells are resubmitted with --resume. The DP
### trainer restores weights + optimizer + privacy accountant + epoch,
### so the continued run consumes exactly the remaining privacy budget
### (total ε across both jobs == target ε).
###
### IMPORTANT: point LOG_BASE at the SAME directory the original sweep
### wrote to (the date is in the dir name, so set it explicitly here —
### it will NOT match today's date if the sweep ran on an earlier day).
###
### Usage:
###   LOG_BASE=/scratch/.../experiments/DP_Eps_Sweep_20260724/ \
###       ./resume_experiments_dp_eps_sweep.sh
###   # or edit LOG_BASE below.
####################################################################

set -euo pipefail

# -------------------- Base configuration --------------------
PROJECT="project_462001244"

# Set this to the EXISTING sweep directory you want to resume. Override via
# environment: LOG_BASE=/path ./resume_experiments_dp_eps_sweep.sh
: "${LOG_BASE:=/scratch/$PROJECT/rodr_temp/llm_project/test_repo/dpdl/experiments/DP_Eps_Sweep_REPLACE_ME/}"

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
MAX_GRAD_NORM=1
MAX_LENGTH=185
VALIDATION_FREQUENCY=3

EPSILONS=("INF" "64" "16" "4" "1")
SEEDS=(42 43 44)

# -------------------- Helpers --------------------
function is_job_in_queue() {
    local experiment_name=$1
    if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
        return 0
    fi
    return 1
}

# Echo the highest completed epoch from checkpoint_epoch_<N> dirs, or -1 if none.
function latest_checkpoint_epoch() {
    local ckpt_dir=$1
    local latest=-1
    if [ -d "$ckpt_dir" ]; then
        for d in "$ckpt_dir"/checkpoint_epoch_*; do
            [ -d "$d" ] || continue
            # require the training_state.pt to count it as resumable
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
n_finished=0
n_queued=0
n_resumed=0
n_fresh=0

for m in "${!MODEL_NAMES[@]}"; do
    MODEL="${MODEL_NAMES[$m]}"
    MODEL_TAG="${MODEL_TAGS[$m]}"

    for EPS in "${EPSILONS[@]}"; do

        if [ "$EPS" = "INF" ]; then
            EPS_FLAGS="--no-privacy"
            EPS_TAG="inf"
        else
            EPS_FLAGS="--privacy --target-epsilon $EPS --max-grad-norm $MAX_GRAD_NORM --normalize-clipping"
            EPS_TAG="$EPS"
        fi

        for SEED in "${SEEDS[@]}"; do

            EXPERIMENT_NAME="SWEEP_${MODEL_TAG}_eps${EPS_TAG}_seed${SEED}"
            EXPERIMENT_DIR="$LOG_BASE/$EXPERIMENT_NAME"

            # 1. Finished?
            if [ -f "$EXPERIMENT_DIR/runtime" ]; then
                echo "FINISHED    : $EXPERIMENT_NAME"
                n_finished=$((n_finished + 1))
                continue
            fi

            # 2. Already queued/running?
            if is_job_in_queue "$EXPERIMENT_NAME"; then
                echo "IN QUEUE    : $EXPERIMENT_NAME"
                n_queued=$((n_queued + 1))
                continue
            fi

            # 3. Resumable? (DP cells write checkpoint_epoch_<N>; ε=∞ does not)
            last_epoch=$(latest_checkpoint_epoch "$EXPERIMENT_DIR/checkpoints")
            if [ "$last_epoch" -ge 0 ]; then
                echo "RESUMABLE   : $EXPERIMENT_NAME (from epoch $((last_epoch + 1)) of $EPOCHS)"
                n_resumed=$((n_resumed + 1))
            else
                if [ "$EPS" = "INF" ]; then
                    echo "NOT STARTED : $EXPERIMENT_NAME (non-DP: no checkpointing, restarts fresh)"
                else
                    echo "NOT STARTED : $EXPERIMENT_NAME (no checkpoint, restarts fresh)"
                fi
                n_fresh=$((n_fresh + 1))
            fi

            mkdir -p "$EXPERIMENT_DIR"

            # Resubmit with --resume. On a fresh cell this just starts training
            # and begins checkpointing; on a resumable cell it continues.
            # NOTE: no --overwrite-experiment here — it would delete the very
            # checkpoints we want to resume from. (--resume also suppresses it
            # in code as a belt-and-braces guard.)
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
                --validation-frequency "$VALIDATION_FREQUENCY" \
                --record-loss-by-step \
                --record-loss-by-epoch \
                --save-model \
                --resume \
                --experiment-name "$EXPERIMENT_NAME" \
                --log-dir "$LOG_BASE" \
                $EPS_FLAGS

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
