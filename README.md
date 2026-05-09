# CPT Experiments

This folder contains the Continued Pre-Training pipeline for 8B-scale models on Helios.
It is self-contained and reuses only `helios/accelerate_config.yaml`.

## What This Runs

- Models: `meta-llama/Llama-3.1-8B`, `Qwen/Qwen3-8B-Base`, `google/gemma-4-e4b`
- Variants: `FT-KY`, `FT-KZ`, `FT-PL`
- Experiment A: equal word budget — 100M words per language
- Experiment B: equal token budget — 100M tokens per language (pre-sized datasets)
- LoRA: bf16 RSLoRA, `target_modules="all-linear"`, no 4-bit quantization
- Epochs: 3 over Phase 2 (target language data); `max_steps` auto-computed at runtime
- Curriculum: first 10% of steps train on 100% English, remaining 90% on target language only

## Curriculum Strategy

The model trains on 100% English for the first 10% of steps, then switches to 100% target
language for the remaining 90%. This anchors reasoning and in-context learning capabilities
before adaptation, preventing catastrophic forgetting without needing English in Phase 2.
(Per paper: retains over 94% of original performance.)

English data is a fixed 100M word dataset — the same across all languages and both experiments.

## Files

- `scripts/prepare_cpt_data.py`: downloads HF datasets, shuffles before cutoff, counts words/tokens, packs fixed-length causal LM sequences, saves `train_phase1` (English only) and `train_phase2` (target only)
- `scripts/train_cpt.py`: trains RSLoRA adapters with the two-phase CPT schedule; auto-computes `max_steps` from dataset size
- `jobs/prepare_cpt_data.sh`: CPU SLURM wrapper for data prep
- `jobs/grid_search.sh`: 4-run FT-KY grid array
- `jobs/pick_best_grid.sh`: selects the lowest final training loss grid result
- `jobs/train_cpt.sh`: full 4-GPU CPT job (3 epochs, max_steps auto-computed)
- `configs/*.yaml`: model-specific hyperparameter templates
- `submit_cpt_pipeline.sh`: one model x one variant pipeline
- `submit_cpt_matrix.sh`: 3 models x 3 variants launcher

## Restart Behavior

Pipeline outputs are resumable by default. `submit_cpt_pipeline.sh` uses `RUN_ID=resume` unless you pass an eighth argument or set `CPT_RUN_ID`, so re-submitting the same model/variant/experiment/dataset targets the same checkpoint directory.

- Data prep skips an existing processed dataset when `data_stats.json` exists. Set `FORCE_PREP=true` to rebuild.
- Training saves regular checkpoints, a `phase1_final` adapter after the English warm-up, and the final adapter under `final`.
- If `final` already exists, the train job exits successfully without retraining.
- If Phase 1 already finished, reruns load `phase1_final` and continue/resume Phase 2.
- This CPT directory stops at trained LoRA adapters. HuggingFace upload, inference, merging, and benchmarking live outside `cpt/`.

## Metrics Logs

Each training output directory contains structured metrics for plots and tables:

- `metrics/train_metrics.jsonl`: one JSON object per Trainer log event
- `metrics/trainer_log_history.json`: final combined Trainer history
- `metrics/trainer_log_history.csv`: flattened CSV for pandas/spreadsheets
- `grid_search_result.json`: final grid-selection summary with final training loss

Every JSONL row includes run metadata such as model, dataset ID, experiment, language variant, phase, LoRA rank, learning rate, epochs, `global_step`, continuous `total_step`, epoch, loss, learning rate, grad norm, and tokens-per-word.

## Required Edits Before Full Runs

Fill dataset IDs in `submit_cpt_matrix.sh`:

```bash
DATASET_IDS_WORDS[FT-KY]    # 100M word Kyrgyz dataset
DATASET_IDS_WORDS[FT-KZ]    # 100M word Kazakh dataset
DATASET_IDS_WORDS[FT-PL]    # 100M word Polish dataset
DATASET_IDS_TOKENS[FT-KY]   # 100M token Kyrgyz dataset
DATASET_IDS_TOKENS[FT-KZ]   # 100M token Kazakh dataset
DATASET_IDS_TOKENS[FT-PL]   # 100M token Polish dataset
ENGLISH_DATASET_ID           # 100M word English dataset (same for all)
```

Verify the exact Gemma model ID before submitting full jobs. `google/gemma-4-e4b` is kept as the plan placeholder.

After each model-family grid search, update the matching config:

```yaml
lora:
  r: 128
  alpha: 256
training:
  learning_rate: 1.0e-4
```

## Smoke Checks

Tiny data prep:

```bash
python cpt/scripts/prepare_cpt_data.py \
  --dataset_id DATASET_ID \
  --tokenizer_id meta-llama/Llama-3.1-8B \
  --lang_variant FT-KY \
  --experiment words \
  --word_budget 1000 \
  --english_dataset_id ENGLISH_DATASET_ID \
  --english_word_budget 100 \
  --output_dir data/cpt_processed/smoke_test
```

Tiny CPU training:

```bash
python cpt/scripts/train_cpt.py \
  --model meta-llama/Llama-3.1-8B \
  --data_path data/cpt_processed/smoke_test \
  --lang_variant FT-KY \
  --lora_r 16 \
  --learning_rate 5e-5 \
  --epochs 0 --max_steps 5 \
  --cpu \
  --output_dir /tmp/cpt_smoke_test
```

## Submission

Single model and variant (full training, grid search skipped):

```bash
bash cpt/submit_cpt_pipeline.sh \
  meta-llama/Llama-3.1-8B \
  DATASET_ID \
  FT-KY \
  words \
  cpt/configs/llama_cpt.yaml \
  true \
  ENGLISH_DATASET_ID
```

Run FT-KY grid search before the full matrix:

```bash
bash cpt/submit_cpt_pipeline.sh \
  meta-llama/Llama-3.1-8B \
  DATASET_ID \
  FT-KY \
  words \
  cpt/configs/llama_cpt.yaml \
  false \
  ENGLISH_DATASET_ID
```

This submits data prep, the 4-run grid array, and winner selection only. The winner is written to `cpt/logs/grid_winner_<model>.txt`. Update the matching config, then rerun with `true` for full training.

Full matrix after dataset IDs and configs are ready:

```bash
bash cpt/submit_cpt_matrix.sh words true
bash cpt/submit_cpt_matrix.sh tokens true
```

To submit FT-KY grid jobs for all model families, pass `false` as the second argument. After updating configs from the winner files, launch the matrix with grid skipped.
