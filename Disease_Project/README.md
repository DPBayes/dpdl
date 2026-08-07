# Disease_Project — running the DP DiseaseTask experiments on LUMI

This folder holds the batch launchers for the differentially-private
`DiseaseTask` fine-tuning study (privacy / memorization on
`sarus-tech/medical_v3`). Everything runs on **LUMI** (`standard-g`, 8× MI250 per node) through SLURM.

The workflow is always the same three steps:

1. **Launch a sweep** — `bash run_experiments_<sweep>.sh` submits one SLURM job
   per grid cell (model × knob × seed).
2. **Monitor** — `squeue --me`.
3. **Resume the unfinished cells** — `./resume_experiments_<sweep>.sh` resubmits
   anything that didn't finish, continuing from the last checkpoint with the DP
   accountant intact.

---

## 0. One-time setup (per clone / per LUMI account)

These must be in place before the first launch.

1. **Virtual environment** at the path the launcher expects:
   ```
   /scratch/project_x/venvs/dpdl-llms
   ```
   It must have this repo installed (`pip install -e .` from the repo root) plus
   `torch`, `opacus`, `peft`, `transformers`, `datasets`, `torchmetrics`.
   If your venv lives elsewhere, edit the `source .../bin/activate` line in
   [`run_script_distributed_lumi.sh`](run_script_distributed_lumi.sh).

2. **`run_wrapper.sh` next to the launcher.** `run_script_distributed_lumi.sh`
   calls `srun ./run_wrapper.sh …`, so a `run_wrapper.sh` must exist **in the
   directory you submit from** (this `Disease_Project/` folder). It sets the
   per-rank distributed env vars and execs `python3 run.py …`. If you don't have
   it here yet, copy the repo's wrapper and make sure it can reach `run.py`:
   ```bash
   cp ../examples/slurm_run_wrapper.sh ./run_wrapper.sh
   chmod +x run_wrapper.sh
   ```
   > NOTE: `run.py` lives in the **repo root**, one level up from here. The
   > wrapper must either `cd` to the repo root or reference `../run.py`. Confirm
   > your `run_wrapper.sh` resolves `run.py` before launching a full sweep —
   > the single-job smoke test in step 2 is the cheapest way to check.

3. **Datasets / model caches** are read from `/scratch/project_x/data`
   (`HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`). These are set in
   the launcher; the first job will populate them if the models aren't cached.

4. **Account / partition / paths.** If you are not on `project_x`, edit
   the `#SBATCH --account`, `PROJECT`, and the `LOG_BASE` variables (top of each
   `run_experiments_*.sh`).

---

## 1. The pieces in this folder

| File | What it is |
|------|------------|
| [`run_script_distributed_lumi.sh`](run_script_distributed_lumi.sh) | The **SLURM job script**. 8 GPUs, `standard-g`, 48 h, loads the CSC PyTorch stack, activates the venv, and runs `srun ./run_wrapper.sh $@`. You rarely call this by hand — the sweep scripts pass it to `sbatch`. |
| `run_experiments_dp_eps_sweep.sh`  | **ε sweep** (E1/E2/E6): ε ∈ {∞, 64, 16, 4, 1} × {mistral, olmo2} × seeds {42,43,44} = **30 jobs**. |
| `run_experiments_dp_clip_sweep.sh` | **Clipping sweep** (E3): max-grad-norm ∈ {1, 10, 50, 500} at ε=4 × 2 models × 3 seeds = **24 jobs**. |
| `run_experiments_dp_rank_sweep.sh` | **LoRA-rank sweep** (E4/E5): rank ∈ {4, 8, 16, 32} × ε ∈ {∞, 4} × 2 models × 3 seeds = **48 jobs**. |
| `resume_experiments_dp_*.sh`       | The **resume helper** for each sweep. Re-scans the grid and resubmits only the cells that didn't finish, with `--resume`. |
| `methodology.pdf`                  | The experiment matrix (E1–E6) these sweeps implement. |

All sweeps train `DiseaseTask` with LoRA on `sarus-tech/medical_v3`
(`--dataset-label-field disease`), 25 epochs, batch 1024 / physical 12,
`--max-length 185`, `--validation-frequency 3`, `--save-model`, `--resume`.
ε=∞ cells use `--no-privacy` (no DP knobs, no checkpointing); DP cells add
`--privacy --target-epsilon <ε> --max-grad-norm <C> --normalize-clipping`.

---

## 2. Launch a batch of experiments

From **inside this folder** on a LUMI login node:

```bash
cd /scratch/project_x/.../dpdl/Disease_Project     # wherever this repo lives

# Launch the ε sweep (30 jobs):
bash run_experiments_dp_eps_sweep.sh
```

Each sweep script:

- builds a dated output dir, e.g.
  `.../experiments/DP_Eps_Sweep_YYYYMMDD/` (`LOG_BASE`, set at the top of the
  script — **edit it to your scratch path**),
- loops over the grid and submits one job per cell with a descriptive name,
  e.g. `SWEEP_mistral_eps4_seed42`,
- **skips cells that are already done** (a `runtime` file exists in the cell dir)
  or already queued, so re-running the script is safe and idempotent.

You'll see one line per cell (`Submitted:` / `Already completed:` /
`Already in queue:`). All jobs go to the queue at once; with capacity the whole
sweep finishes in roughly one job's wall-clock.

The other sweeps are launched the same way:

```bash
bash run_experiments_dp_clip_sweep.sh    # 24 jobs, E3
bash run_experiments_dp_rank_sweep.sh    # 48 jobs, E4/E5
```

### Optional: one quick job first (recommended before a full sweep)

Submit a single cell by hand to confirm the env/wrapper/paths work end-to-end
before flooding the queue:

```bash
sbatch -J SmokeOne run_script_distributed_lumi.sh run.py train \
    --llm --task DiseaseTask --num-workers 7 \
    --model-name allenai/OLMo-2-1124-7B-Instruct \
    --dataset-name sarus-tech/medical_v3 --dataset-label-field disease \
    --batch-size 256 --physical-batch-size 12 \
    --epochs 1 --max-length 120 --seed 42 \
    --learning-rate 0.00026 --optimizer AdamW \
    --peft lora --lora-rank 8 --validation-frequency 1 \
    --privacy --target-epsilon 4 --max-grad-norm 1 --normalize-clipping \
    --save-model --overwrite-experiment \
    --experiment-name SmokeOne \
    --log-dir /scratch/project_462001244/rodr_temp/llm_project/smoke/
```

(There's also `../smoke_test_disease.sh` at the repo root — a 2-GPU, non-DP,
1-epoch smoke test you can `sbatch` directly.)

---

## 3. Resume the unfinished cells

DP jobs checkpoint at 25 %/50 %/75 % of training (`--resume` +
`checkpoint_fraction`), saving LoRA adapters + optimizer + **DP privacy
accountant** + epoch. If a job hits the 48 h wall or the queue kills it, the
resume helper continues it from the last checkpoint so the total privacy budget
across both runs equals the target ε (no double-counting).

Point `LOG_BASE` at the **same dated directory the original sweep wrote to**
(the date is baked into the dir name, so it won't match today's date), then run:

```bash
LOG_BASE=/scratch/project_x/.../experiments/DP_Eps_Sweep_20260807/ \
    ./resume_experiments_dp_eps_sweep.sh
```

For each cell it prints one of `FINISHED` / `IN QUEUE` / `RESUMABLE` /
`NOT STARTED` and resubmits the unfinished ones with `--resume`. It is
idempotent — run it as many times as needed until every cell says `FINISHED`.

> ε=∞ (non-private) cells do **not** checkpoint; if one didn't finish it simply
> restarts fresh.

---

## 4. Where the results land

Per cell, under `LOG_BASE/<EXPERIMENT_NAME>/`:

| File | Contents |
|------|----------|
| `test_metrics` | final test loss + token-level accuracy/perplexity + the disease/PII CustomAccuracyLog metrics |
| `per_disease_accuracy.csv` | per-disease correct / true_count / accuracy |
| `disease_confusion_matrix.csv` | truth × predicted disease counts (last col = no-prediction) |
| `per_sample_eval_test.csv` | one row per test sample: truth, prediction, answer log-prob, generated text, prompt, per-PII-field truth + match — drives the memorization analyses |
| `epoch_learning_rate.csv` | per-epoch LR (from `--record-learning-rate`) |
| `train_loss_by_step.csv`, epoch loss/acc | training curves |
| `final_model.pt` / LoRA adapter dir | saved model (`--save-model`) |
| `checkpoints/checkpoint_epoch_<N>/` | resume checkpoints (DP cells; only the latest is kept) |
| `runtime` | written on completion — its presence marks the cell **done** |

---

## Quick reference

```bash
# launch
bash run_experiments_dp_eps_sweep.sh
bash run_experiments_dp_clip_sweep.sh
bash run_experiments_dp_rank_sweep.sh

# monitor
squeue --me | grep SWEEP_

# resume (LOG_BASE = the ORIGINAL dated sweep dir)
LOG_BASE=/scratch/.../experiments/DP_Eps_Sweep_YYYYMMDD/ ./resume_experiments_dp_eps_sweep.sh
```

**Before a sweep, double check:** account (`--account`), `LOG_BASE` path, the
venv path in `run_script_distributed_lumi.sh`, and that `run_wrapper.sh` is
present here and resolves `run.py`.
