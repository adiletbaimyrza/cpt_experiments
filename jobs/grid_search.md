# grid_search.sh

SLURM array job for CPT hyperparameter grid search.

## Purpose

Runs four predefined short CPT runs on `FT-KY` only. The goal is to choose a stable LoRA rank and learning rate per model family before launching the full matrix.

## Array Runs

| Array ID | Run | Learning Rate | LoRA Rank |
|---:|---|---:|---:|
| 0 | A | `1e-4` | 64 |
| 1 | B | `1e-4` | 128 |
| 2 | C | `1e-4` | 256 |
| 3 | D | `2e-4` | 128 |

## Arguments

```text
$1 MODEL
$2 FT_KY_DATASET
$3 GRID_MAX_STEPS
```

`FT_KY_DATASET` is the prepared dataset name under `data/cpt_processed/`.

## Outputs

Each run writes under:

```text
${SCRATCH}/kyrgyzLLM/checkpoints/grid_<MODEL_SHORT>_r<RANK>_lr<LR>_<SLURM_ARRAY_JOB_ID>/
```

Important outputs:

- `grid_search_result.json`
- `metrics/train_metrics.jsonl`
- phase checkpoints and final adapter artifacts

## Follow-Up

Use `pick_best_grid.sh` to select the winner from the same SLURM array job ID.
