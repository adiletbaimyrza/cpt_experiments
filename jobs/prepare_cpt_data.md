# prepare_cpt_data.sh

SLURM CPU wrapper for CPT data preparation.

## Purpose

Submits a CPU job that calls `cpt/scripts/prepare_cpt_data.py` on Helios.

It prepares packed map-style datasets under:

```text
data/cpt_processed/<OUTPUT_NAME>/
```

## Arguments

```text
$1 DATASET_ID
$2 TOKENIZER_ID
$3 LANG_VARIANT
$4 EXPERIMENT
$5 BUDGET
$6 OUTPUT_NAME
$7 ENGLISH_DATASET_ID
```

Valid values:

- `LANG_VARIANT`: `FT-KY`, `FT-KZ`, `FT-PL`
- `EXPERIMENT`: `words` or `tokens`

## Resume Behavior

If the output directory already contains `data_stats.json`, the job exits successfully without rebuilding.

Set:

```bash
FORCE_PREP=true
```

to force a rebuild.

## Environment

The job follows the working Helios environment pattern:

- loads `ML-bundle/24.06a`
- enters `${SCRATCH}/kyrgyzLLM`
- activates `${SCRATCH}/kyrgyzLLM/venv`
- uses `${SCRATCH}/kyrgyzLLM/cache` as `HF_HOME`
- reads `${SCRATCH}/kyrgyzLLM/.env` for `HF_TOKEN`
