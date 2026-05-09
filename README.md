# CPT Experiments

Continued Pre-Training pipeline for 8B-scale LLMs on Helios (PLGrid GH200).

## What This Runs

- **Models**: `meta-llama/Llama-3.1-8B`, `Qwen/Qwen3-8B-Base`, `google/gemma-4-e4b`
- **Languages**: `FT-KY` (Kyrgyz), `FT-KZ` (Kazakh), `FT-PL` (Polish)
- **Experiment A**: equal word budget — 100M words per language
- **Experiment B**: equal token budget — 100M tokens per language
- **LoRA**: bf16 RSLoRA, `target_modules="all-linear"`, no quantization
- **Epochs**: 3 over Phase 2 data; `max_steps` auto-computed at runtime from dataset size
- **Curriculum**: Phase 1 = 100% English (first 10% of steps); Phase 2 = 100% target language (remaining 90%)

English data is a fixed 100M-word dataset — the same across all languages and both experiments.

## Curriculum Strategy

The model trains on 100% English for the first 10% of steps, then switches to 100% target language. This anchors reasoning and in-context learning before adaptation, preventing catastrophic forgetting without needing English in Phase 2. (Per paper: retains over 94% of original performance.)

## Repository Layout

```
setup_and_submit.sh           # entry point: one-shot cluster setup + full pipeline submission
submit_cpt_matrix.sh          # submits all 3 model pipelines for one experiment
submit_cpt_pipeline.sh        # submits one model's full automated chain
scripts/
  prepare_cpt_data.py         # downloads HF dataset, packs sequences, saves train_phase1/phase2
  train_cpt.py                # RSLoRA training with two-phase curriculum; auto-resumes
jobs/
  prepare_cpt_data.sh         # SLURM wrapper — CPU-only data prep
  grid_search.sh              # SLURM array (4 runs) — FT-KY hyperparameter search
  pick_best_grid.sh           # selects lowest-loss grid run, writes winner JSON
  apply_winner_and_train.sh   # patches config YAML with winner, submits all 3 training chains
  train_cpt.sh                # SLURM wrapper — full CPT training job
configs/
  llama_cpt.yaml              # Llama-3.1-8B hyperparameters
  qwen_cpt.yaml               # Qwen3-8B-Base hyperparameters
  gemma_cpt.yaml              # Gemma-4-E4B hyperparameters
.env.example                  # template for cluster secrets and dataset IDs
```

## Automated Pipeline

Each model runs an independent SLURM chain fully submitted by `setup_and_submit.sh`:

```
FT-KY data prep
  → grid search (4-run array: lr ∈ {1e-4, 2e-4}, rank ∈ {64, 128, 256})
  → pick winner (lowest final training loss)
  → patch configs/<model>_cpt.yaml with winning lora.r + learning_rate   ← automated
  → FT-KY prep + train                                                    ← automated
  → FT-KZ prep + train                                                    ← automated
  → FT-PL prep + train                                                    ← automated
```

No manual steps between grid search and full training.

## Cluster Setup (Helios)

**1. Clone the repo**

```bash
cd /net/scratch/hscra/plgrid/plgadiletbaimyrza
git clone git@github.com:adiletbaimyrza/cpt_experiments.git
cd cpt_experiments
```

**2. Create `.env`**

```bash
cp .env.example /net/scratch/hscra/plgrid/plgadiletbaimyrza/cpt_experiments/.env
nano /net/scratch/hscra/plgrid/plgadiletbaimyrza/cpt_experiments/.env
```

Fill in `HF_TOKEN` and all `CPT_DATASET_*` variables. See `.env.example` for the full list.

**3. Submit**

```bash
bash setup_and_submit.sh words    # Experiment A — equal word budget
bash setup_and_submit.sh tokens   # Experiment B — equal token budget
```

`setup_and_submit.sh` is safe to re-run. It creates `logs/YYYY-MM-DD/`, `data/cpt_processed/`, and `checkpoints/` if they don't exist; uses a shared HF cache at `$SCRATCH/hf_home`; builds the venv once; validates `HF_TOKEN` and all dataset IDs; then submits the matrix.

**Monitor jobs**

```bash
squeue -u $(whoami)
```

## Resource Allocation

| Job | CPUs | Mem | GPU | Time |
|-----|------|-----|-----|------|
| `prepare_cpt_data.sh` | 8 | 64 GB | 0 | 4 h |
| `grid_search.sh` | 8 | — | 1 | 4 h |
| `train_cpt.sh` | 8 | — | 1 | 24 h |
| `pick_best_grid.sh` | 4 | 8 GB | 0 | 10 m |
| `apply_winner_and_train.sh` | 2 | 4 GB | 0 | 10 m |

24 h is the partition wall-time limit. Worst-case full training (Kyrgyz words, ~13.7k steps) takes ~4 h on a GH200.

## Restart Behavior

- **Data prep**: skips if `data_stats.json` exists. Set `FORCE_PREP=true` to rebuild.
- **Training**: saves checkpoints every 250 steps. If the job hits the wall-time limit, resubmit the same command — it resumes from the latest checkpoint automatically.
- **Phase 1 done, Phase 2 not**: `phase1_final/` is detected and Phase 1 is skipped on resume.
- **Already complete**: if `final/` exists, the job exits immediately.
- **Run ID**: defaults to `resume` so resubmitting the same model/variant/experiment always targets the same checkpoint directory. Pass `CPT_RUN_ID=<id>` or a ninth argument to start fresh.

## Logs

All SLURM output for a run lands in `logs/YYYY-MM-DD/` (date of submission). Each training output directory also contains structured metrics:

| File | Contents |
|------|----------|
| `metrics/train_metrics.jsonl` | one JSON object per Trainer log event |
| `metrics/trainer_log_history.json` | combined Trainer log history for both phases |
| `metrics/trainer_log_history.csv` | flattened CSV for pandas/spreadsheets |
| `grid_search_result.json` | grid run summary with final training loss |

Every JSONL row includes: model, dataset ID, experiment, language variant, phase, LoRA rank, learning rate, `global_step`, continuous `total_step`, epoch, loss, grad norm, tokens-per-word.

## Smoke Checks

Run these locally or on the login node before submitting full jobs.

**Data prep (tiny budget):**

```bash
python scripts/prepare_cpt_data.py \
  --dataset_id YOUR_DATASET_ID \
  --tokenizer_id meta-llama/Llama-3.1-8B \
  --lang_variant FT-KY \
  --experiment words \
  --word_budget 1000 \
  --english_dataset_id YOUR_ENGLISH_DATASET_ID \
  --english_word_budget 100 \
  --output_dir data/cpt_processed/smoke_test
```

**CPU training (5 steps):**

```bash
python scripts/train_cpt.py \
  --model meta-llama/Llama-3.1-8B \
  --data_path data/cpt_processed/smoke_test \
  --lang_variant FT-KY \
  --lora_r 16 \
  --learning_rate 5e-5 \
  --epochs 0 --max_steps 5 \
  --cpu \
  --output_dir /tmp/cpt_smoke_test
```

## Manual Submission

To submit a single model's full pipeline directly:

```bash
bash submit_cpt_pipeline.sh \
  meta-llama/Llama-3.1-8B \
  words \
  configs/llama_cpt.yaml \
  ENGLISH_DATASET_ID \
  DATASET_FT_KY \
  DATASET_FT_KZ \
  DATASET_FT_PL
```
