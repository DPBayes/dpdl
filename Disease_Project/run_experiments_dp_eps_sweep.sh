#!/bin/bash
####################################################################
### Experiment: ε sweep for DP DiseaseTask finetuning
###
### Covers experiments E1 (headline privacy gap), E2 (concept phase
### diagram — reuses these checkpoints, no new training), and E6
### (similar-disease interference — reuses these checkpoints, no new
### training).
###
### Sweeps:
###   - model ∈ {Mistral-7B-Instruct-v0.3, OLMo-2-1124-7B-Instruct}
###   - ε ∈ {inf, 64, 16, 4, 1}    (inf = --no-privacy, no DP knobs)
###   - seeds: 42, 43, 44          (3 per cell, matrix floor)
###   = 30 jobs total (2 × 5 × 3)
###
### Each job: full data, LoRA rank = matrix default.
### All 15 jobs submitted at once; wallclock for the whole sweep ≈
### one job's runtime if the cluster has capacity.
####################################################################

set -euo pipefail

# -------------------- Base configuration --------------------
PROJECT="project_462001244"
EXPERIMENT_BASE="DP_Eps_Sweep_$(date +%Y%m%d)"
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

# Matrix defaults — change here if you re-baseline.
EPOCHS=25
BATCH_SIZE=1024
PHYSICAL_BATCH_SIZE=12
LORA_RANK=8                # matrix specifies rank=8 default. Flip to 16 if you
                           # want parity with the older lr-scheduler sweep.
LEARNING_RATE="0.00026"
MAX_GRAD_NORM=1            # used only when --privacy is enabled
# Note: δ is not a CLI flag — the trainer sets it to 1/N internally when
# --target-epsilon is passed. Matches the matrix spec (δ = min(1e-5, 1/N))
# whenever N ≥ 100k (true for DiseaseTask).
MAX_LENGTH=185
VALIDATION_FREQUENCY=3

# -------------------- Sweep axes --------------------
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

# -------------------- Submit grid --------------------
for m in "${!MODEL_NAMES[@]}"; do
    MODEL="${MODEL_NAMES[$m]}"
    MODEL_TAG="${MODEL_TAGS[$m]}"

    for EPS in "${EPSILONS[@]}"; do

        # ε=∞ means non-private: drop all DP knobs and use --no-privacy.
        # Anything else: build the full DP flag set for that ε.
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

echo
echo "Submitted up to 30 jobs (model × ε × seeds) for E1 + E2 + E6."
echo "Logs will land in: $LOG_BASE"
echo "Check status with: squeue --me | grep SWEEP_"
