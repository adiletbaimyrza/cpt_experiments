# CPT Experiments

This folder contains the Continued Pre-Training pipeline for 8B-scale models on Helios.
It is self-contained and reuses only `helios/accelerate_config.yaml`.

## What This Runs

- Models: `meta-llama/Llama-3.1-8B`, `Qwen/Qwen3-8B-Base`, `google/gemma-4-e4b`
- Variants: `FT-KY`, `FT-KZ`, `FT-PL`
- Experiment A: equal word budget, default `100000000` words
- Experiment B: equal token budget, default `100000000` tokens
- LoRA: bf16 RSLoRA, `target_modules="all-linear"`, no 4-bit quantization
- Curriculum: first 10% of steps use `train_phase1` with English mixed in, remaining 90% use target-only `train_phase2`

English is not a separate variant. It is a uniform 10% phase-1 curriculum mix for every language variant.

## Files

- `scripts/prepare_cpt_data.py`: downloads HF datasets, shuffles before cutoff, counts words/tokens, packs fixed-length causal LM sequences, saves `train_phase1` and `train_phase2`
- `scripts/train_cpt.py`: trains RSLoRA adapters with the two-phase CPT schedule
- `jobs/prepare_cpt_data.sh`: CPU SLURM wrapper for data prep
- `jobs/grid_search.sh`: 4-run FT-KY grid array
- `jobs/pick_best_grid.sh`: selects the lowest final training loss grid result
- `jobs/train_cpt.sh`: full 4-GPU CPT job
- `configs/*.yaml`: model-specific hyperparameter templates
- `submit_cpt_pipeline.sh`: one model x one variant pipeline
- `submit_cpt_matrix.sh`: 3 models x 3 variants launcher

## Restart Behavior

Pipeline outputs are resumable by default. `submit_cpt_pipeline.sh` uses `RUN_ID=resume` unless you pass a ninth argument or set `CPT_RUN_ID`, so re-submitting the same model/variant/experiment/dataset targets the same checkpoint directory.

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

Every JSONL row includes run metadata such as model, dataset ID, experiment, language variant, phase, LoRA rank, learning rate, max steps, `global_step`, continuous `total_step`, epoch, loss, learning rate, grad norm, tokens-per-word, and English phase-1 ratio when available.

## Required Edits Before Full Runs

Fill dataset IDs in `cpt/submit_cpt_matrix.sh`:

```bash
DATASET_IDS[FT-KY]
DATASET_IDS[FT-KZ]
DATASET_IDS[FT-PL]
ENGLISH_DATASET_ID
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
  --max_steps 5 \
  --cpu \
  --output_dir /tmp/cpt_smoke_test
```

## Submission

Single model and variant:

```bash
bash cpt/submit_cpt_pipeline.sh \
  meta-llama/Llama-3.1-8B \
  DATASET_ID \
  FT-KY \
  words \
  cpt/configs/llama_cpt.yaml \
  20000 \
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
  500 \
  false \
  ENGLISH_DATASET_ID
```

This submits data prep, the 4-run grid array, and winner selection only. The winner is written to `cpt/logs/grid_winner_<model>.txt`. Update the matching config, then rerun with `SKIP_GRID_SEARCH=true` for full training.

Full matrix after dataset IDs and configs are ready:

```bash
bash cpt/submit_cpt_matrix.sh words 20000 true
bash cpt/submit_cpt_matrix.sh tokens 20000 true
```

To submit FT-KY grid jobs for all model families, pass `false` as the third argument. That mode submits grid jobs only. After updating configs from the winner files, launch the matrix with grid skipped.
