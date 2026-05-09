# prepare_cpt_data.py

Prepares CPT datasets from HuggingFace Hub data.

## Purpose

This script creates DDP-safe, packed HuggingFace `DatasetDict` outputs for CPT training. It handles:

- Loading a target-language HF dataset
- Shuffling before budget cutoff
- Cutting by word budget or token budget
- Adding about 10% English tokens to Phase 1
- Packing text into fixed `seq_len` causal LM chunks
- Writing `data_stats.json`

## Outputs

The output directory contains:

- `train_phase1`: target language plus English warm-up mix
- `train_phase2`: target language only
- `data_stats.json`: dataset, token, word, and English-ratio metadata

## Phase Semantics

Phase 1 data is used for the first 10% of CPT training steps. English is mixed by token budget, not by document count.

Phase 2 data is target-language only and is used for the remaining 90% of training steps.

## Key Arguments

- `--dataset_id`: target-language HF dataset ID
- `--tokenizer_id`: model tokenizer used for token counting and packing
- `--lang_variant`: `FT-KY`, `FT-KZ`, or `FT-PL`
- `--experiment`: `words` or `tokens`
- `--word_budget`: target word budget for `words`
- `--token_budget`: target token budget for `tokens`
- `--english_dataset_id`: English HF dataset ID
- `--english_ratio`: target Phase 1 English token ratio, default `0.1`
- `--output_dir`: destination directory
- `--seq_len`: packed sequence length, default `2048`
- `--text_column`: dataset text field, default `text`

## Notes

The script does not stream into training. It saves map-style datasets with `save_to_disk()` because map-style datasets are safer with multi-GPU DDP.

This CPT directory stops at data preparation and adapter training.
