# train_cpt.py

Runs resumable CPT training with bf16 RSLoRA.

## Purpose

This script trains LoRA adapters on the packed datasets produced by `prepare_cpt_data.py`.

It differs from the older supervised LoRA training path by using:

- bf16 model loading, no 4-bit QLoRA
- RSLoRA
- `target_modules="all-linear"`
- `warmup_ratio`
- `weight_decay=0.1`
- `max_grad_norm=1.0`
- two sequential curriculum phases

## Training Phases

Phase 1:

- First 10% of optimizer steps
- Uses `train_phase1`
- About 90% target-language tokens and 10% English tokens

Phase 2:

- Remaining 90% of optimizer steps
- Uses `train_phase2`
- Target-language only

## Resume Behavior

The script is designed for repeated SLURM submissions.

- If `final/` contains a complete adapter, training exits successfully.
- If `phase1_final/` exists, Phase 1 is skipped and Phase 2 starts from that adapter.
- If `phase1/checkpoint-*` exists, Phase 1 resumes from the latest checkpoint.
- If `phase2/checkpoint-*` exists, Phase 2 resumes from the latest checkpoint.

Final adapter writes are rank-gated so only the main distributed process writes shared files.

## Structured Metrics

Metrics are written under `OUTPUT_DIR/metrics/`.

- `train_metrics.jsonl`: appended during training, one JSON row per log event
- `trainer_log_history.json`: final combined Trainer history
- `trainer_log_history.csv`: flattened CSV for plots and tables

Important fields:

- `phase`
- `global_step`
- `total_step`
- `loss`
- `learning_rate`
- `grad_norm`
- `tokens_per_word`
- `english_ratio_actual_phase1`

Use `total_step` as the x-axis for continuous plots across both phases.

## Key Arguments

- `--model`: base model ID
- `--data_path`: packed DatasetDict directory
- `--lang_variant`: `FT-KY`, `FT-KZ`, or `FT-PL`
- `--output_dir`: checkpoint and adapter directory
- `--max_steps`: total optimizer steps across both phases
- `--lora_r`: LoRA rank
- `--lora_alpha`: defaults to `2 * lora_r`
- `--learning_rate`: optimizer learning rate
- `--save_steps`: checkpoint interval
- `--save_total_limit`: retained checkpoint count
