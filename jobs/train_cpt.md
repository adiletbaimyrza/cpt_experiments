# train_cpt.sh

Main 4-GPU SLURM job for CPT training.

## Purpose

Runs `cpt/scripts/train_cpt.py` through `accelerate launch` using 4 GH200 GPUs.

This job trains one model, one language variant, and one prepared dataset.

## Arguments

```text
$1 MODEL
$2 DATASET
$3 LANG_VARIANT
$4 LORA_R
$5 LR
$6 MAX_STEPS
$7 RUN_ID
```

`DATASET` is the name under `data/cpt_processed/`.

## Output Directory

```text
${SCRATCH}/kyrgyzLLM/checkpoints/cpt_<MODEL_SHORT>_<LANG_VARIANT>_<DATASET>_<RUN_ID>/
```

## Checkpoints

The wrapper passes:

- `--save_steps 250`
- `--save_total_limit 4`
The Python script additionally writes:

- `phase1/checkpoint-*`
- `phase1_final/`
- `phase2/checkpoint-*`
- `final/`
- `metrics/train_metrics.jsonl`
- `metrics/trainer_log_history.csv`
- `grid_search_result.json`

## Resume Behavior

Re-submit the same pipeline command with the same `RUN_ID` to resume.

The default orchestrator run ID is `resume`, so repeated submissions target the same output directory unless you override it.

## Final Artifact

The CPT output artifact is the LoRA adapter under:

```text
final/
```

Downstream merging, HuggingFace upload, inference, and benchmarking live outside `cpt/`.
