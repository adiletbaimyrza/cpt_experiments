"""
Upload trained CPT adapters to the HuggingFace Hub.

For each adapter under $SCRATCH/cpt_experiments/checkpoints/cpt_*/final/:
  1. Generate a model card (base model, language, training config, losses)
  2. Create the HF repo (if needed)
  3. Upload the adapter folder

Usage:
  python scripts/upload_adapters.py --org <hf_username_or_org>
  python scripts/upload_adapters.py --org <org> --private
  python scripts/upload_adapters.py --org <org> --experiment words --dry-run
  python scripts/upload_adapters.py --org <org> --only Llama-3.1-8B-FT-KY

Repo naming:
  <org>/cpt-<model_short>-<variant>-<experiment>
  e.g. adiletbaimyrza/cpt-Llama-3.1-8B-FT-KY-words

Requires HF_TOKEN in env with WRITE scope (https://huggingface.co/settings/tokens).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo


# Filename produced by jobs/train_cpt.sh:
#   cpt_<MODEL_SHORT>_<VARIANT>_<DATASET_SAFE>_<RUN_ID>
# DATASET_SAFE is f"{VARIANT}_{EXPERIMENT}_{MODEL_SHORT}_{dataset_id_safe}"
# So we can recover MODEL_SHORT, VARIANT, EXPERIMENT from the path itself.
ADAPTER_DIR_RE = re.compile(
    r"cpt_(?P<model>[^/]+?)_(?P<variant>FT-(?:KY|KZ|PL))_"
    r"(?P=variant)_(?P<experiment>words|tokens)_(?P=model)_"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org", required=True,
                   help="HF username or organization to push under")
    p.add_argument("--checkpoints_root", default=None,
                   help="Directory containing cpt_*/ adapter dirs "
                        "(default: $SCRATCH/cpt_experiments/checkpoints)")
    p.add_argument("--experiment", choices=["words", "tokens", "both"], default="both",
                   help="Filter by experiment (default: both)")
    p.add_argument("--only", action="append", default=[],
                   help="Only upload adapters matching this token (e.g. 'Llama-3.1-8B' "
                        "or 'FT-KY'). Repeatable; matches if ANY appears in the dir name.")
    p.add_argument("--private", action="store_true",
                   help="Create private repos (default: public)")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be uploaded without touching the Hub")
    p.add_argument("--commit_message", default="Upload CPT adapter")
    return p.parse_args()


def find_adapters(root: Path, experiment_filter: str, only_tokens: list) -> list[Path]:
    """Return final/ paths under cpt_*/ matching the filters."""
    if not root.is_dir():
        sys.exit(f"ERROR: checkpoints root does not exist: {root}")

    adapters = []
    for adapter_root in sorted(root.glob("cpt_*")):
        final_dir = adapter_root / "final"
        if not (final_dir / "adapter_config.json").is_file():
            continue
        name = adapter_root.name
        if experiment_filter != "both":
            if f"_{experiment_filter}_" not in name:
                continue
        if only_tokens:
            if not any(tok in name for tok in only_tokens):
                continue
        adapters.append(final_dir)
    return adapters


def parse_metadata(final_dir: Path) -> dict:
    """Recover model / variant / experiment / metrics from the run's artifacts."""
    parent = final_dir.parent  # cpt_<MODEL>_<VARIANT>_..._<RUN_ID>
    name = parent.name

    # Recover MODEL_SHORT, VARIANT, EXPERIMENT from the directory name.
    # Format: cpt_<MODEL>_<VARIANT>_<VARIANT>_<EXPERIMENT>_<MODEL>_<dataset>...
    parts = name.split("_")
    if len(parts) < 6 or parts[0] != "cpt":
        return {"adapter_dir_name": name}

    # parts[1]: MODEL_SHORT (no underscores in our model names: "Llama-3.1-8B" etc.)
    # However, "Qwen3-8B-Base" has hyphens not underscores, so split by _ is safe.
    model_short = parts[1]
    variant = parts[2]
    # parts[3] should equal parts[2] (DATASET_SAFE starts with VARIANT)
    experiment = parts[4] if parts[4] in ("words", "tokens") else None

    info = {
        "adapter_dir_name": name,
        "model_short": model_short,
        "variant": variant,
        "experiment": experiment,
    }

    # Pull training metadata from JSONL if available.
    metrics_jsonl = parent / "metrics" / "train_metrics.jsonl"
    if metrics_jsonl.is_file():
        last_p1 = None
        last_p2 = None
        with open(metrics_jsonl) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("phase") == "phase1":
                    last_p1 = rec
                elif rec.get("phase") == "phase2":
                    last_p2 = rec
        last = last_p2 or last_p1
        if last:
            info["base_model"] = last.get("model")
            info["dataset_id"] = last.get("dataset_id")
            info["lora_r"] = last.get("lora_r")
            info["lora_alpha"] = last.get("lora_alpha")
            info["learning_rate"] = last.get("learning_rate")
            info["max_steps"] = last.get("max_steps")
            info["phase1_steps"] = last.get("phase1_steps")
            info["phase2_steps"] = last.get("phase2_steps")
            info["epochs"] = last.get("epochs")
            info["total_words"] = last.get("total_words")
            info["total_tokens"] = last.get("total_tokens")

    grid_result = parent / "grid_search_result.json"
    if grid_result.is_file():
        with open(grid_result) as f:
            data = json.load(f)
        info["final_train_loss"] = data.get("final_train_loss")
        info["final_eval_loss"] = data.get("final_eval_loss")

    return info


def repo_id_for(org: str, info: dict) -> str:
    """Construct the HF repo ID (org/name)."""
    name = f"cpt-{info['model_short']}-{info['variant']}-{info.get('experiment','x')}"
    return f"{org}/{name}"


def make_model_card(info: dict, repo_id: str) -> str:
    base = info.get("base_model") or "unknown"
    variant = info.get("variant", "?")
    experiment = info.get("experiment", "?")
    lang = {"FT-KY": "Kyrgyz (kir, ky)", "FT-KZ": "Kazakh (kaz, kk)", "FT-PL": "Polish (pol, pl)"}.get(variant, variant)
    train_loss = info.get("final_train_loss")
    eval_loss = info.get("final_eval_loss")
    lora_r = info.get("lora_r")
    lora_alpha = info.get("lora_alpha")
    lr = info.get("learning_rate")
    total_words = info.get("total_words")
    total_tokens = info.get("total_tokens")

    train_loss_s = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else "n/a"
    eval_loss_s = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else "n/a"

    body = f"""---
license: other
library_name: peft
base_model: {base}
tags:
  - continued-pretraining
  - cpt
  - lora
  - rslora
  - {variant.lower()}
language:
  - {variant.split('-')[1].lower()}
---

# {repo_id.split('/', 1)[1]}

LoRA + RSLoRA adapter for **continued pretraining** of `{base}` on **{lang}**.

## Training

- **Recipe**: RSLoRA (`use_rslora=True`), `target_modules="all-linear"`, `modules_to_save=["embed_tokens","lm_head"]`
- **LoRA**: rank={lora_r}, alpha={lora_alpha}, dropout=0.05
- **Optimizer**: AdamW with cosine schedule, warmup_ratio=0.05, weight_decay=0.1, max_grad_norm=1.0
- **Learning rate**: {lr}
- **Precision**: bf16, gradient checkpointing
- **Curriculum**: Phase 1 (10% steps) = English warmup, Phase 2 (90% steps) = target language only
- **Budget**: experiment={experiment}, total_words={total_words}, total_tokens={total_tokens}

## Final losses

| Metric | Value |
|---|---|
| Phase 2 final train loss | {train_loss_s} |
| Phase 2 final eval loss  | {eval_loss_s} |

## Adapter contents

This adapter contains:
- LoRA delta weights for all `nn.Linear` layers
- **Full fine-tuned `embed_tokens` and `lm_head` tensors** (saved via PEFT `modules_to_save`)

The embedding layer override is necessary because the base tokenizer fragments
{lang} into many subword pieces with sparsely trained embeddings.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype="bfloat16", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{base}")
model = PeftModel.from_pretrained(base, "{repo_id}")
model.eval()
```

## Notes

- Adapter size is larger than typical LoRA (~2–4 GB vs ~500 MB) due to the
  full embedding/lm_head tensors.
- Generated by the [cpt_experiments](https://github.com/adiletbaimyrza/cpt_experiments) pipeline.
"""
    return body


def upload_one(api: HfApi, final_dir: Path, info: dict, repo_id: str,
               private: bool, dry_run: bool, commit_message: str):
    print(f"\n→ {final_dir}")
    print(f"  repo: {repo_id}")
    print(f"  base: {info.get('base_model','?')}  variant: {info.get('variant','?')}  exp: {info.get('experiment','?')}")
    print(f"  losses: train={info.get('final_train_loss')} eval={info.get('final_eval_loss')}")

    # Write the model card alongside the adapter (no harm if it ends up locally too).
    card_path = final_dir / "README.md"
    card_path.write_text(make_model_card(info, repo_id))

    if dry_run:
        print(f"  [dry-run] would create {repo_id} (private={private}) and upload {final_dir}")
        return

    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(final_dir),
        commit_message=commit_message,
        repo_type="model",
    )
    print(f"  ✓ pushed → https://huggingface.co/{repo_id}")


def main():
    args = parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("ERROR: HF_TOKEN not set. Add a write-scope token to your .env.")

    root = Path(args.checkpoints_root) if args.checkpoints_root else \
           Path(os.environ.get("SCRATCH", "")) / "cpt_experiments" / "checkpoints"

    adapters = find_adapters(root, args.experiment, args.only)
    if not adapters:
        sys.exit(f"No adapters matched under {root} (experiment={args.experiment}, only={args.only})")

    print(f"Found {len(adapters)} adapter(s) to upload from {root}\n")

    api = HfApi(token=os.environ["HF_TOKEN"])

    failures = []
    for final_dir in adapters:
        info = parse_metadata(final_dir)
        if not info.get("model_short") or not info.get("variant"):
            print(f"  SKIP: could not parse {final_dir.parent.name}")
            failures.append(final_dir)
            continue
        repo_id = repo_id_for(args.org, info)
        try:
            upload_one(api, final_dir, info, repo_id,
                       private=args.private, dry_run=args.dry_run,
                       commit_message=args.commit_message)
        except Exception as e:
            print(f"  ✗ failed: {e}")
            failures.append(final_dir)

    print(f"\nDone. {len(adapters) - len(failures)} succeeded, {len(failures)} failed.")
    if failures and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
