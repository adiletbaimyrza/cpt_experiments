# submit_cpt_pipeline.sh

Orchestrates one CPT pipeline for one model and one language variant.

## Purpose

This is the primary entrypoint for targeted CPT runs.

With grid search skipped, it submits:

```text
prepare_cpt_data -> train_cpt
```

With grid search enabled, it submits:

```text
prepare_cpt_data -> grid_search -> pick_best_grid
```

and then stops so the config can be updated manually.

## Arguments

```text
$1 MODEL
$2 DATASET_ID
$3 LANG_VARIANT
$4 EXPERIMENT
$5 CONFIG_FILE
$6 MAX_STEPS
$7 SKIP_GRID_SEARCH
$8 ENGLISH_DATASET_ID
$9 RUN_ID
```

`RUN_ID` is optional. If omitted, the script uses:

```text
${CPT_RUN_ID:-resume}
```

## Valid Values

- `LANG_VARIANT`: `FT-KY`, `FT-KZ`, `FT-PL`
- `EXPERIMENT`: `words`, `tokens`
- `SKIP_GRID_SEARCH`: `true`, `false`

## Resumability

The default `RUN_ID=resume` makes repeated submissions target the same training output directory.

Pass a custom ninth argument only when you intentionally want a fresh output directory.

## Scope

This pipeline stops after CPT adapter training. Downstream merging, HuggingFace upload, inference, and benchmarking live outside `cpt/`.

## Outputs

Prepared data:

```text
data/cpt_processed/<LANG_VARIANT>_<EXPERIMENT>_<MODEL_SHORT>_<DATASET_ID_SLUG>/
```

Training output:

```text
checkpoints/cpt_<MODEL_SHORT>_<LANG_VARIANT>_<DATASET_SAFE>_<RUN_ID>/
```

Final adapter:

```text
checkpoints/cpt_<MODEL_SHORT>_<LANG_VARIANT>_<DATASET_SAFE>_<RUN_ID>/final/
```
