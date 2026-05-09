# submit_cpt_matrix.sh

Submits the full CPT experiment matrix.

## Purpose

This script launches the model x language-variant matrix through `submit_cpt_pipeline.sh`.
It stops at trained CPT adapters.

The active matrix is:

```text
3 model families x 3 language variants
```

Language variants:

- `FT-KY`
- `FT-KZ`
- `FT-PL`

## Arguments

```text
$1 EXPERIMENT
$2 MAX_STEPS
$3 SKIP_GRID_SEARCH
```

Defaults:

```text
EXPERIMENT=words
MAX_STEPS=20000
SKIP_GRID_SEARCH=true
```

## Dataset IDs

Before running, fill these placeholders in the script:

- `DATASET_IDS[FT-KY]`
- `DATASET_IDS[FT-KZ]`
- `DATASET_IDS[FT-PL]`
- `ENGLISH_DATASET_ID`

## Grid Search Mode

If `SKIP_GRID_SEARCH=false`, this script submits only FT-KY grid-search chains for each model family.

After those finish:

1. Read `cpt/logs/grid_winner_<MODEL_SHORT>.txt`.
2. Update the matching config in `cpt/configs/`.
3. Re-run the matrix with `SKIP_GRID_SEARCH=true`.

## Final Output Count

Per experiment:

```text
3 models x 3 variants = 9 final outputs
```

Across both `words` and `tokens`:

```text
18 final outputs
```
