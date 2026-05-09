# CPT Experiments Branch Plan

## Context

The `helios_optimized` branch has a proven SLURM pipeline for LoRA fine-tuning 1.7B models on Kyrgyz. We now need a new branch (`cpt_experiments`) that scales this to **Continued Pre-Training (CPT) on 8B-scale base models** across 3 language variants (FT-KY, FT-KZ, FT-PL), following the research pipeline from the PDF.

The key differences from the existing pipeline:
- 8B models (Llama-3.1-8B, Qwen3-8B-Base, Gemma-4-E4B) → 4×GH200 GPUs
- RSLoRA with `target_modules="all-linear"` (not a fixed 7-module list)
- `weight_decay=0.1`, `gradient_clipping=1.0`, `warmup_ratio=0.05` (not warmup_steps)
- **4 predefined hyperparameter runs** (A/B/C/D) per model on FT-KY only, then best fixed for all variants
- Two experiments: **Exp A** (equal words, 100M) and **Exp B** (equal tokens, 100M)
- **English in first 10% of steps for ALL variants** (anti-catastrophic-forgetting, per paper: "including English in the first 10% training steps in a curriculum learning fashion retains over 94% of original performance")
- English is not a separate variant; it is a uniform ~10% Phase 1 token mix for FT-KY, FT-KZ, and FT-PL
- All CPT datasets already on HuggingFace — load by dataset ID, no re-cleaning
- **No 4-bit QLoRA** — full bf16 LoRA (4×GH200 = 384GB VRAM, 8B model in bf16 ≈ 16GB)
- **Sequence packing** in data prep (ConstantLengthDataset) to eliminate padding waste (~30-50% compute savings)
- **Shuffle before budget cutoff** to avoid domain confound between Exp A and B

---

## Branch & Folder Structure

New branch: `cpt_experiments` (branched from `helios_optimized`)

New folder in repo root:
```
cpt/
├── PLAN.md                           ← this file
├── README.md                         ← run guide for humans
├── scripts/
│   ├── prepare_cpt_data.py           # Load HF dataset, shuffle, count words/tokens, pack, save
│   └── train_cpt.py                  # CPT training: RSLoRA, all-linear, EN curriculum mixing
├── jobs/
│   ├── prepare_cpt_data.sh           # CPU SLURM job (data prep)
│   ├── grid_search.sh                # SLURM array job (4 runs A/B/C/D, FT-KY only)
│   ├── pick_best_grid.sh             # Collector job (afterany:ARRAY_ID, picks winner)
│   └── train_cpt.sh                  # GPU SLURM job (4×GH200, main CPT run)
├── configs/
│   ├── llama_cpt.yaml                # Frozen hyperparams for Llama-3.1-8B (post grid search)
│   ├── qwen_cpt.yaml                 # Frozen hyperparams for Qwen3-8B-Base
│   └── gemma_cpt.yaml                # Frozen hyperparams for Gemma-4-E4B
├── submit_cpt_pipeline.sh            # Single model × variant orchestrator
└── submit_cpt_matrix.sh              # Full 3 models × 3 variants launcher
```

No changes to existing files in `helios/`, `training/`, or `data/`. The `cpt/` folder is self-contained and only calls:
- `helios/accelerate_config.yaml` (reused for multi-GPU config)

This CPT folder stops at trained LoRA adapters. Merging, HuggingFace upload, inference, and benchmarking live outside `cpt/`.

---

## File-by-File Implementation Plan

### 1. `cpt/scripts/prepare_cpt_data.py`

Replaces `data/prepare_kyrgyz.py` for CPT. Loads from HuggingFace Hub by dataset ID. **Does not stream into training** — downloads the slice, saves to disk as a map-style Dataset (streaming IterableDataset + DDP causes rank sync deadlocks).

**Key arguments:**
```
--dataset_id            HF dataset ID (e.g. "user/kyrgyz-cpt-100m")
--tokenizer_id          HF model ID for token counting (must match training model family)
--lang_variant          FT-KY | FT-KZ | FT-PL
--experiment            words | tokens  (Exp A or Exp B)
--word_budget           100_000_000 (Exp A cutoff)
--token_budget          100_000_000 (Exp B cutoff)
--english_dataset_id    HF dataset ID for English mix (required for all variants)
--english_ratio         Fraction of Phase 1 tokens that are English (default: 0.1 for all variants)
--output_dir            Save path (written to data/cpt_processed/<name>/)
--seq_len               2048 (packing chunk size)
--seed                  42
```

**Key logic:**
1. `ds.shuffle(seed=42)` **before** any budget cutoff — ensures Exp A and Exp B see the same document distribution, avoiding domain confound
2. Stream record-by-record, maintaining `word_count` (whitespace split) and `token_count` (tokenizer, no truncation); stop at `word_budget` (Exp A) or `token_budget` (Exp B)
3. Compute and report `tokens_per_word` ratio — key paper metric (KY ~3.0, KZ ~2.5, PL ~1.25)
4. **Sequence packing**: concatenate all documents end-to-end (with EOS token separator), chunk into fixed `seq_len=2048` windows — eliminates padding waste
5. Save as standard map-style `Dataset` to disk via `save_to_disk()` — train-ready, DDP-safe
6. **ALL variants** produce two splits:
   - `train_phase1`: English interleaved with target-language data, packed (first 10% of steps)
     - FT-KY/KZ/PL: ~10% English tokens, 90% target language (anti-forgetting minimal injection)
   - `train_phase2`: target-language only, packed (remaining 90% of steps)
7. Write `data_stats.json`: `{lang_variant, experiment, total_docs, total_words, total_tokens, tokens_per_word, packed_sequences, english_ratio}`

**Saves to:** `data/cpt_processed/<output_dir>/` as HuggingFace `DatasetDict`

---

### 2. `cpt/scripts/train_cpt.py`

New training script; **do not modify `training/train_lora.py`**. Core differences:

| Parameter | `training/train_lora.py` | `cpt/scripts/train_cpt.py` |
|---|---|---|
| `target_modules` | Hardcoded 7-module list | `"all-linear"` |
| RSLoRA | Not used | `use_rslora=True` in `LoraConfig` |
| 4-bit QLoRA | Default on | **Off** — full bf16 LoRA (GH200 has 96GB, no reason to quantize) |
| `weight_decay` | Not set | `0.1` |
| `max_grad_norm` | Not set | `1.0` |
| Warmup | `warmup_steps` (int) | `warmup_ratio=0.05` |
| EN curriculum | None | Two sequential `trainer.train()` calls for all variants (DDP-safe) |
| Instruction format | Optional `"Суроо: ... Жооп: ..."` | Never — raw CPT text only |
| Grid search output | None | Writes `grid_search_result.json` after training |
| Checkpoint resume | Not handled | Auto-detects existing checkpoint, passes `resume_from_checkpoint` |

**Key arguments** (additive to existing ones in `train_lora.py`):
```
--use_rslora            Flag (always True for CPT runs)
--weight_decay          0.1
--max_grad_norm         1.0
--warmup_ratio          0.05
--lang_variant          FT-KY | FT-KZ | FT-PL
--lora_alpha            Optional override (default: 2 * lora_r, auto-computed)
```

**No `--use_4bit` flag** — model loaded in bf16 directly:
```python
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, ...)
# No BitsAndBytesConfig, no prepare_model_for_kbit_training
```

**RSLoRA config:**
```python
LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,     # 2 * r
    target_modules="all-linear",
    lora_dropout=0.05,
    use_rslora=True,
    bias="none",
    task_type="CAUSAL_LM"
)
```
Print `model.print_trainable_parameters()` in logs — verify `lm_head` is not included (not expanding vocabulary).

**Curriculum mixing — ALL variants (DDP-safe approach):**
- All variants have `train_phase1` and `train_phase2` splits (produced by `prepare_cpt_data.py`)
- Phase 1 (first 10% of steps): call `trainer.train()` on `train_phase1` split
  - FT-KY/KZ/PL: English is a small anti-forgetting injection (~10% English tokens)
- Phase 2 (remaining 90%): call `trainer.train(resume_from_checkpoint=True)` on `train_phase2` split (target-language only)
- Two sequential `trainer.train()` calls is the DDP-safe approach — custom IterableDataset causes rank sync deadlocks in multi-GPU DDP

**Checkpoint resume:** at start of training, check if `output_dir` already contains a checkpoint; if so, pass `resume_from_checkpoint=latest` to `trainer.train()`. This handles the 24h SLURM wall-time limit for long runs.

**Post-training:** saves `grid_search_result.json` to `output_dir` with `{run_label, lora_r, learning_rate, final_train_loss}`. Copy `_patch_accelerate_unwrap_model()` from `training/train_lora.py` verbatim.

---

### 3. `cpt/jobs/prepare_cpt_data.sh`

CPU SLURM job wrapping `prepare_cpt_data.py`.

```
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=cpt/logs/prepare-cpt-%j.log
#SBATCH --error=cpt/logs/prepare-cpt-%j.err
```

Arguments: `$1=DATASET_ID  $2=TOKENIZER_ID  $3=LANG_VARIANT  $4=EXPERIMENT  $5=BUDGET  $6=OUTPUT_NAME  $7=ENGLISH_DATASET_ID`

Activates venv from `${SCRATCH}/kyrgyzLLM/venv`, sets `HF_HOME`, loads `.env` token — identical preamble to `helios/jobs/prepare_data.sh`.

---

### 4. `cpt/jobs/grid_search.sh`

SLURM array job: **4 tasks** (runs A/B/C/D — predefined combos, not a full grid), on FT-KY only per model.

| Run | LR | Rank | Notes |
|---|---|---|---|
| A | 1e-4 | 64 | low-capacity baseline |
| B | 1e-4 | 128 | strong default |
| C | 1e-4 | 256 | literature prior |
| D | 2e-4 | 128 | higher LR stress test |

```
#SBATCH --array=0-3
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=cpt/logs/grid-search-%A-%a.log
#SBATCH --error=cpt/logs/grid-search-%A-%a.err
```

Arguments: `$1=MODEL  $2=FT-KY-DATASET  $3=GRID_MAX_STEPS (default: 500)`

Array index → predefined run config:
```bash
LR_VALUES=(1e-4 1e-4 1e-4 2e-4)
RANK_VALUES=(64 128 256 128)
RUN_LABELS=(A B C D)

LR=${LR_VALUES[$SLURM_ARRAY_TASK_ID]}
LORA_R=${RANK_VALUES[$SLURM_ARRAY_TASK_ID]}
LORA_ALPHA=$((LORA_R * 2))
RUN_LABEL=${RUN_LABELS[$SLURM_ARRAY_TASK_ID]}
```

Each task runs `accelerate launch --num_processes 4 cpt/scripts/train_cpt.py` and writes `grid_search_result.json` with `{run_label, lora_r, learning_rate, final_train_loss}`.

**200-500 steps only** — enough to see loss divergence; not a full training run.

---

### 5. `cpt/jobs/pick_best_grid.sh`

Collector job submitted with `--dependency=afterany:ARRAY_ID` (not `afterok` — one crashed task won't block picking a winner).

```
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:0
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=cpt/logs/grid-winner-%j.log
```

Arguments: `$1=MODEL_SHORT` (used to glob checkpoints)

```bash
python -c "
import json, glob, sys, os
pattern = '${SCRATCH}/kyrgyzLLM/checkpoints/grid_${MODEL_SHORT}_*/grid_search_result.json'
files = glob.glob(pattern)
if not files:
    print('ERROR: no grid_search_result.json found'); sys.exit(1)
results = [json.load(open(f)) for f in files if os.path.exists(f)]
best = min(results, key=lambda x: x['final_train_loss'])
print(json.dumps(best, indent=2))
" > cpt/logs/grid_winner_${MODEL_SHORT}.txt
cat cpt/logs/grid_winner_${MODEL_SHORT}.txt
```

User reads `cpt/logs/grid_winner_<model>.txt`, updates `cpt/configs/<model>_cpt.yaml` with winning `lora_r` and `learning_rate`, then submits full matrix.

---

### 6. `cpt/jobs/train_cpt.sh`

Main GPU training job for full CPT runs.

```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --account=plgunhype-gpu-gh200
#SBATCH --output=cpt/logs/train-cpt-%j.log
#SBATCH --error=cpt/logs/train-cpt-%j.err
```

Arguments: `$1=MODEL  $2=DATASET  $3=LANG_VARIANT  $4=LORA_R  $5=LR  $6=MAX_STEPS  $7=RUN_ID`

Always uses accelerate (never single-GPU):
```bash
accelerate launch \
    --config_file helios/accelerate_config.yaml \
    --num_processes 4 \
    --mixed_precision bf16 \
    cpt/scripts/train_cpt.py \
    --model "${MODEL}" \
    --data_path "data/cpt_processed/${DATASET}" \
    --lang_variant "${LANG_VARIANT}" \
    --lora_r "${LORA_R}" \
    --learning_rate "${LR}" \
    --warmup_ratio 0.05 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --use_rslora \
    --lora_dropout 0.05 \
    --max_steps "${MAX_STEPS}" \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_length 2048 \
    --bf16 --gradient_checkpointing \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${MODEL_SHORT}_${LANG_VARIANT}"
```

Note: no `--use_4bit` flag — model loads in bf16 directly.

**Effective batch size:** 4 GPUs × 2 × 4 = 32 (appropriate for 8B CPT)

---

### 7. `cpt/configs/{llama,qwen,gemma}_cpt.yaml`

Template (values marked `# SET AFTER GRID SEARCH`):
```yaml
model: "meta-llama/Llama-3.1-8B"   # (or Qwen/Qwen3-8B-Base, google/gemma-4-e4b)
lora:
  r: 128                    # SET AFTER GRID SEARCH (grid tested: 64, 128, 256)
  alpha: 256                # 2 * r, auto-computed
  dropout: 0.05
  use_rslora: true
  target_modules: "all-linear"
training:
  learning_rate: 1e-4       # SET AFTER GRID SEARCH (grid tested: 1e-4, 2e-4)
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  weight_decay: 0.1
  max_grad_norm: 1.0
  max_length: 2048
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  bf16: true
  gradient_checkpointing: true
compute:
  num_gpus: 4
  partition: plgrid-gpu-gh200
  account: plgunhype-gpu-gh200
  time_limit: "24:00:00"
grid_search:
  runs:
    A: {lr: 1e-4, rank: 64}
    B: {lr: 1e-4, rank: 128}
    C: {lr: 1e-4, rank: 256}
    D: {lr: 2e-4, rank: 128}
  grid_max_steps: 500
  full_max_steps: 20000       # adjust per dataset size after data prep
```

---

### 8. `cpt/submit_cpt_pipeline.sh`

Single-model × single-variant orchestrator. Chain: `prepare_cpt_data → [optional grid_search + pick_best →] train_cpt`.

Arguments: `$1=MODEL  $2=DATASET_ID  $3=LANG_VARIANT  $4=EXPERIMENT  $5=CONFIG_FILE  $6=MAX_STEPS  $7=SKIP_GRID_SEARCH(true|false)  $8=ENGLISH_DATASET_ID`

SLURM dependency chain:
```
PREP_JOB_ID   = sbatch prepare_cpt_data.sh ...
if not SKIP_GRID_SEARCH:
  GRID_JOB_ID = sbatch --dependency=afterok:PREP_JOB_ID grid_search.sh ...
  PICK_JOB_ID = sbatch --dependency=afterany:GRID_JOB_ID pick_best_grid.sh ...
  # User must update config manually after reading grid_winner_<model>.txt
  # User updates config manually after reading grid_winner_<model>.txt,
  # then reruns with SKIP_GRID_SEARCH=true for full training.
else:
  TRAIN_JOB_ID = sbatch --dependency=afterok:PREP_JOB_ID train_cpt.sh ...
```

Reads `lora_r` and `learning_rate` from `CONFIG_FILE` using a Python one-liner to parse the YAML.

---

### 9. `cpt/submit_cpt_matrix.sh`

Full experiment matrix launcher.

```bash
MODELS=("meta-llama/Llama-3.1-8B" "Qwen/Qwen3-8B-Base" "google/gemma-4-e4b")
LANG_VARIANTS=("FT-KY" "FT-KZ" "FT-PL")

# HF dataset IDs per language (placeholders — user fills in)
declare -A DATASET_IDS=(
    [FT-KY]="TBD/kyrgyz-cpt-100m"
    [FT-KZ]="TBD/kazakh-cpt-100m"
    [FT-PL]="TBD/polish-cpt-100m"
)
ENGLISH_DATASET_ID="TBD/english-cpt-100m-tokens"

EXPERIMENT=${1:-"words"}   # words | tokens
```

Per model: submit FT-KY with `SKIP_GRID_SEARCH=false`, then FT-KZ/FT-PL with `SKIP_GRID_SEARCH=true` (reusing the config frozen after grid search).

---

### 10. `cpt/README.md`

Documents:
- Experiment A vs B (word vs token budget, why it matters for agglutinative languages)
- Grid search protocol (4 runs A/B/C/D on FT-KY per model family; fix params for all variants)
- English curriculum strategy (first 10% of steps for ALL variants; cite paper: "retains over 94% of original performance")
- HF dataset IDs (fill in when ready)
- Step-by-step run guide: grid search → update configs → full matrix
- Expected compute: 12 full runs × ~24h × 4 GPUs = ~1152 GPU-hours

---

## Workflow Summary

```
Step 1 — Create branch
  git checkout -b cpt_experiments

Step 2 — Implement cpt/ folder (all files above)

Step 3 — Fill HF dataset IDs in submit_cpt_matrix.sh and ENGLISH_DATASET_ID

Step 4 — Grid search (once per model family, FT-KY only)
  bash cpt/submit_cpt_pipeline.sh meta-llama/Llama-3.1-8B TBD/kyrgyz-cpt-100m \
       FT-KY words cpt/configs/llama_cpt.yaml 500 false TBD/english-cpt-100m
  [wait for pick_best_grid.sh to write cpt/logs/grid_winner_llama.txt]
  # Read winner, update cpt/configs/llama_cpt.yaml manually
  (repeat for Qwen, Gemma)

Step 5 — Full matrix, Experiment A (equal words)
  bash cpt/submit_cpt_matrix.sh words

Step 6 — Full matrix, Experiment B (equal tokens)
  bash cpt/submit_cpt_matrix.sh tokens

Step 7 — Collect adapters
  ls ${SCRATCH}/kyrgyzLLM/checkpoints/cpt_*/final/
```

---

## Critical Existing Files (Bases for New Code)

| New file | Based on |
|---|---|
| `cpt/scripts/train_cpt.py` | `training/train_lora.py` — extend, do not modify |
| `cpt/jobs/train_cpt.sh` | `helios/jobs/train.sh` — same SLURM preamble pattern |
| `cpt/jobs/prepare_cpt_data.sh` | `helios/jobs/prepare_data.sh` — same env preamble |
| `cpt/submit_cpt_pipeline.sh` | `helios/submit_full_pipeline.sh` — same dependency chain pattern |
| Multi-GPU launcher | `helios/accelerate_config.yaml` — reused as-is, override `--num_processes 4` |

---

## Known Placeholders (Fill Before Running)

| Placeholder | Where | What to fill |
|---|---|---|
| `TBD/kyrgyz-cpt-100m` | `submit_cpt_matrix.sh` | HF dataset ID for Kyrgyz |
| `TBD/kazakh-cpt-100m` | `submit_cpt_matrix.sh` | HF dataset ID for Kazakh |
| `TBD/polish-cpt-100m` | `submit_cpt_matrix.sh` | HF dataset ID for Polish |
| `TBD/english-cpt-100m-tokens` | `submit_cpt_matrix.sh` | HF dataset ID for English |
| `google/gemma-4-e4b` | configs, matrix | Verify exact HF model ID |
| `r: 128` / `learning_rate: 1e-4` | `cpt/configs/*.yaml` | Update after grid search |
| `full_max_steps: 20000` | `cpt/configs/*.yaml` | Adjust after data prep reports token count |

---

## Verification

**Local smoke tests (before Helios submission):**
```bash
# 1. Data prep — tiny slice, CPU only
python cpt/scripts/prepare_cpt_data.py \
  --dataset_id TBD/kyrgyz-cpt-100m \
  --tokenizer_id meta-llama/Llama-3.1-8B \
  --lang_variant FT-KY --experiment words \
  --word_budget 1000 \
  --english_dataset_id TBD/english-cpt-100m-tokens \
  --output_dir data/cpt_processed/smoke_test

# 2. Training — 5 steps, CPU mode (verifies two-phase curriculum runs without DDP crash)
python cpt/scripts/train_cpt.py \
  --model meta-llama/Llama-3.1-8B \
  --data_path data/cpt_processed/smoke_test \
  --lang_variant FT-KY \
  --lora_r 16 --learning_rate 5e-5 \
  --max_steps 5 --cpu \
  --output_dir /tmp/cpt_smoke_test
```

**On Helios:** submit a 10-step sanity run via `submit_cpt_pipeline.sh` before launching the full matrix to confirm SLURM job chaining, GPU allocation, and checkpoint saving all work correctly.
