# CPT Pipeline Documentation

This directory contains the Continued Pre-Training pipeline for KyrgyzLLM experiments on Helios.

The active experiment matrix is:

- 3 base model families: Llama, Qwen, Gemma
- 3 language variants: `FT-KY`, `FT-KZ`, `FT-PL`
- 2 budget experiments: `words` and `tokens`
- Uniform English curriculum: about 10% English tokens during the first 10% of training steps for every variant

English is not a separate model variant. It is only an anti-forgetting curriculum mix during Phase 1.

## Pipeline Flow

1. Prepare packed CPT data with `jobs/prepare_cpt_data.sh`.
2. Optionally run FT-KY grid search with `jobs/grid_search.sh`.
3. Select the best grid run with `jobs/pick_best_grid.sh`.
4. Train a resumable CPT adapter with `jobs/train_cpt.sh`.
5. Use the final adapter output in downstream projects.

For normal use, submit through:

- [submit_cpt_pipeline.md](submit_cpt_pipeline.md) for one model and variant
- [submit_cpt_matrix.md](submit_cpt_matrix.md) for the full model x variant matrix

## Root Files

- [README.md](README.md): run guide and high-level notes
- [PLAN.md](PLAN.md): implementation plan and research rationale
- [submit_cpt_pipeline.md](submit_cpt_pipeline.md): one model x one variant orchestrator
- [submit_cpt_matrix.md](submit_cpt_matrix.md): full matrix launcher

## Python Scripts

- [scripts/prepare_cpt_data.md](scripts/prepare_cpt_data.md): HF dataset loading, budget cutoff, English token mixing, sequence packing
- [scripts/train_cpt.md](scripts/train_cpt.md): RSLoRA CPT training, two-phase curriculum, resume/checkpoint behavior, structured metrics

## SLURM Jobs

- [jobs/prepare_cpt_data.md](jobs/prepare_cpt_data.md): CPU data preparation job
- [jobs/grid_search.md](jobs/grid_search.md): FT-KY 4-run grid search array
- [jobs/pick_best_grid.md](jobs/pick_best_grid.md): grid winner collector
- [jobs/train_cpt.md](jobs/train_cpt.md): 4-GPU CPT training job

## Resumability

The pipeline is designed so re-submitting the same command does not throw away progress.

- `submit_cpt_pipeline.sh` defaults to `RUN_ID=resume`, so the same model/variant/dataset targets the same output directory.
- Data prep skips an existing processed dataset when `data_stats.json` exists.
- Training saves checkpoints, `phase1_final`, and `final`.
- If `final` exists, training exits successfully without retraining.
- This directory stops at CPT adapter training.

Use a custom ninth argument or `CPT_RUN_ID` only when you intentionally want a fresh run.

Downstream HuggingFace upload, inference, merging, and benchmarking live outside `cpt/`.

## Metrics

Each training output directory writes structured logs:

- `metrics/train_metrics.jsonl`
- `metrics/trainer_log_history.json`
- `metrics/trainer_log_history.csv`
- `grid_search_result.json`

Use `total_step` for continuous plots across Phase 1 and Phase 2.
