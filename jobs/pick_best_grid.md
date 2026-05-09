# pick_best_grid.sh

Collector job for CPT grid search results.

## Purpose

Reads `grid_search_result.json` files from a grid-search array and selects the run with the lowest final training loss.

## Arguments

```text
$1 MODEL_SHORT
$2 GRID_JOB_ID
```

`GRID_JOB_ID` scopes the search to the current grid array, which avoids accidentally selecting stale results from older grid searches.

## Output

Writes:

```text
cpt/logs/grid_winner_<MODEL_SHORT>.txt
```

The winner JSON includes:

- `run_label`
- `lora_r`
- `lora_alpha`
- `learning_rate`
- `final_train_loss`
- `checkpoint_dir`

## Human Step

After the winner file is written, update the matching config file in `cpt/configs/` with the selected rank and learning rate before launching the full matrix.
