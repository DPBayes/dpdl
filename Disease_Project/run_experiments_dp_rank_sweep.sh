#!/bin/bash
####################################################################
### Experiment: LoRA rank sweep for DP DiseaseTask finetuning
###
### Covers experiment E4 (rank ablation). E5 (capability retention)
### reuses these checkpoints via lm-eval-harness; no extra training
### is launched here for E5 — once these jobs finish, point your
### eval harness at the saved adapters under $LOG_BASE.
###
### Sweeps:
###   - model ∈ {Mistral-7B-Instruct-v0.3, OLMo-2-1124-7B-Instruct}
###   - rank ∈ {4, 8, 16, 32}
###   - ε ∈ {inf, 4}            (no-DP vs. mid-DP regime — the
###                              memorization-vs-DP contrast)
###   - seeds: 42, 43, 44       (3 per cell, matrix floor)
###   = 48 jobs total (2 × 4 × 2 × 3)
###
### Note: rank=8 at ε=inf and ε=4 overlap with the eps-sweep script
### (which trains rank=8 at all five ε values, for BOTH models). If
### you've already run the eps-sweep, you can comment out rank=8
### here to avoid duplicating those 12 cells (6 per model). The
### idempotency check below also guards against duplicate
### submission inside the same LOG_BASE.
####################################################################

set -euo pipefail

# -------------------- Base configuration --------------------
PROJECT="project_462001244"
EXPERIMENT_BASE="DP_Rank_Sweep_$(date +%Y%m%d)"
LOG_BASE="/scratch/$PROJECT/rodr_temp/llm_project/test_repo/dpdl/experiments/$EXPERIMENT_BASE/"
mkdir -p "$LOG_BASE"

# -------------------- Fixed dataset / training knobs --------------------
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
LEARNING_RATE="0.00026"
MAX_GRAD_NORM=1
# δ is derived internally by the trainer (1/N when --target-epsilon is set).
MAX_LENGTH=185
VALIDATION_FREQUENCY=3

# -------------------- Sweep axes --------------------
LORA_RANKS=(4 8 16 32)
EPSILONS=("INF" "4")
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

    for RANK in "${LORA_RANKS[@]}"; do
        for EPS in "${EPSILONS[@]}"; do

            # ε=∞: strip all DP knobs and use --no-privacy.
            if [ "$EPS" = "INF" ]; then
                EPS_FLAGS="--no-privacy"
                EPS_TAG="inf"
            else
                EPS_FLAGS="--privacy --target-epsilon $EPS --max-grad-norm $MAX_GRAD_NORM --normalize-clipping"
                EPS_TAG="$EPS"
            fi

            for SEED in "${SEEDS[@]}"; do

                EXPERIMENT_NAME="SWEEP_${MODEL_TAG}_rank${RANK}_eps${EPS_TAG}_seed${SEED}"
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
                    --lora-rank "$RANK" \
                    --peft lora \
                    --validation-frequency "$VALIDATION_FREQUENCY" \
                    --record-loss-by-step \
                    --record-loss-by-epoch \
                    --overwrite-experiment \
                    --save-model \
                    --resume \
                    --experiment-name "$EXPERIMENT_NAME" \
                    --log-dir "$LOG_BASE" \
                    $EPS_FLAGS

                if [ $? -eq 0 ]; then
                    echo "Submitted: $EXPERIMENT_NAME"
                else
                    echo "FAILED to submit: $EXPERIMENT_NAME (exit $?)"
                    exit 1
                fi
            done
        done
    done
done

echo
echo "Submitted up to 48 jobs (model × rank × ε × seeds) for E4 (and downstream E5)."
echo "Logs will land in: $LOG_BASE"
echo "Check status with: squeue --me | grep SWEEP_"
echo
echo "Once jobs finish, point your lm-eval-harness at the saved LoRA"
echo "adapters under $LOG_BASE for the E5 capability-retention readout."
