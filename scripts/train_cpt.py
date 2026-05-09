"""
CPT training script for 8B models on Helios (1×GH200, 96GB).

  - RSLoRA (use_rslora=True), target_modules="all-linear", no QLoRA
  - Two sequential Trainer.train() calls (two-phase curriculum):
      Phase 1 (10% of steps): train_phase1 split — 100% English
      Phase 2 (90% of steps): train_phase2 split — 100% target language
  - max_steps auto-computed from dataset size and --epochs (default 3)
  - Auto-detects latest checkpoint for resume (handles 24h SLURM wall-time limit)
  - Writes grid_search_result.json on completion (used by pick_best_grid.sh)
"""

import argparse
import glob
import inspect
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class JsonlMetricsCallback(TrainerCallback):
    """Write Trainer logs as machine-readable JSONL for later tables/plots."""

    def __init__(self, output_path: Path, run_metadata: dict):
        self.output_path = output_path
        self.run_metadata = run_metadata

    def on_log(self, args, state, control, logs=None, **kwargs):
        is_world_process_zero = getattr(state, "is_world_process_zero", True)
        if not logs or not is_world_process_zero:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            **self.run_metadata,
            "global_step": state.global_step,
            "total_step": self.run_metadata.get("phase_start_step", 0) + state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        with open(self.output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _patch_accelerate_unwrap_model():
    """Compatibility shim for newer Transformers Trainer keep_torch_compile kwarg."""
    sig = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" in sig.parameters:
        return
    original = Accelerator.unwrap_model

    def compat(self, model, *args, **kwargs):
        return original(self, model)

    Accelerator.unwrap_model = compat
    print("Patched Accelerator.unwrap_model for keep_torch_compile compatibility")


def parse_args():
    p = argparse.ArgumentParser(description="CPT training with RSLoRA for 8B models")

    # Model
    p.add_argument("--model", required=True)
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=None,
                   help="Default: 2 * lora_r")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--use_rslora", action="store_true", default=True,
                   help="Use RSLoRA scaling (default: True)")
    p.add_argument("--no_rslora", dest="use_rslora", action="store_false",
                   help="Disable RSLoRA, use vanilla LoRA")

    # Data
    p.add_argument("--data_path", required=True,
                   help="Path to DatasetDict produced by prepare_cpt_data.py")
    p.add_argument("--lang_variant", required=True,
                   choices=["FT-KY", "FT-KZ", "FT-PL"])

    # Training
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of epochs over phase2 (target language) data. "
                        "max_steps is computed automatically from dataset size. "
                        "Set to 0 to use --max_steps directly instead.")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override max_steps directly. Ignored when --epochs > 0.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Compute
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU — for smoke tests only")

    # Grid search
    p.add_argument("--run_label", type=str, default=None,
                   help="Run label (A/B/C/D) written into grid_search_result.json")
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


def load_base_model_and_tokenizer(model_name: str, cpu: bool):
    """Load 8B base model in bf16 (no quantization)."""
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float32 if cpu else torch.bfloat16,
        "trust_remote_code": True,
    }
    if cpu:
        load_kwargs["device_map"] = "cpu"
    elif torch.cuda.is_available():
        # Load shards directly to GPU to avoid the host-RAM peak during deserialization.
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.config.use_cache = False
    return model, tokenizer


def apply_lora(model, lora_r: int, lora_alpha: int, lora_dropout: float, use_rslora: bool):
    """Apply a fresh RSLoRA adapter to a base model.

    Also fully fine-tunes embed_tokens and lm_head via modules_to_save: 'all-linear'
    matches only nn.Linear, so token embeddings (nn.Embedding) are otherwise never
    updated. For cross-lingual CPT (e.g. Cyrillic on a Latin-pretrained tokenizer)
    this caps quality on the new-language side.
    """
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_lora_for_training(model, adapter_dir: Path):
    """Load an existing adapter for continued training."""
    print(f"Loading existing adapter for resume: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    model.print_trainable_parameters()
    return model


def adapter_is_complete(adapter_dir: Path):
    """Return True when a PEFT adapter directory looks complete enough to reuse."""
    if not adapter_dir.is_dir():
        return False
    if not (adapter_dir / "adapter_config.json").is_file():
        return False
    return (adapter_dir / "adapter_model.safetensors").is_file() or (adapter_dir / "adapter_model.bin").is_file()


def find_latest_checkpoint(output_dir: str):
    """Return the latest checkpoint path inside output_dir, or None."""
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not ckpts:
        return None
    def checkpoint_step(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1

    return max(ckpts, key=checkpoint_step)


def make_trainer(model, tokenizer, dataset, training_args, eval_dataset=None):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def add_metrics_callback(trainer, metrics_path: Path, metadata: dict):
    trainer.add_callback(JsonlMetricsCallback(metrics_path, metadata))
    return trainer


def write_metrics_summary(output_dir: Path, phase1_trainer, phase2_trainer, metadata: dict):
    """Write final trainer log history to JSON and flattened CSV."""
    if phase2_trainer is None or not phase2_trainer.is_world_process_zero():
        return

    records = []
    for phase, trainer in (("phase1", phase1_trainer), ("phase2", phase2_trainer)):
        if trainer is None:
            continue
        for entry in trainer.state.log_history:
            phase_start_step = 0 if phase == "phase1" else metadata.get("phase1_steps", 0)
            step = entry.get("step", entry.get("global_step", 0))
            record = {
                **metadata,
                "phase": phase,
                "phase_start_step": phase_start_step,
                "total_step": phase_start_step + step,
                **entry,
            }
            records.append(record)

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history_path = metrics_dir / "trainer_log_history.json"
    with open(history_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if not records:
        return

    import csv
    preferred = [
        "run_name", "model", "lang_variant", "phase", "max_steps",
        "lora_r", "lora_alpha", "learning_rate", "phase_start_step", "step", "global_step", "total_step",
        "epoch", "loss", "grad_norm",
    ]
    keys = []
    for key in preferred:
        if any(key in record for record in records) and key not in keys:
            keys.append(key)
    for record in records:
        for key in record.keys():
            if key not in keys:
                keys.append(key)

    csv_path = metrics_dir / "trainer_log_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def make_training_args(args, output_dir: str, max_steps: int, run_name: str,
                        with_eval: bool = False):
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    training_arg_params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "bf16": args.bf16 and not args.cpu,
        "gradient_checkpointing": args.gradient_checkpointing and not args.cpu,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "ddp_find_unused_parameters": False,
        "report_to": "wandb" if use_wandb else "none",
        "run_name": run_name,
    }

    if with_eval:
        eval_steps = max(args.logging_steps, max_steps // 10) if max_steps else args.logging_steps
        # Newer Transformers renamed evaluation_strategy → eval_strategy.
        eval_strategy_key = "eval_strategy" if "eval_strategy" in training_arg_params else "evaluation_strategy"
        kwargs[eval_strategy_key] = "steps"
        kwargs["eval_steps"] = eval_steps
        kwargs["per_device_eval_batch_size"] = args.batch_size

    if args.cpu and "no_cuda" in training_arg_params:
        kwargs["no_cuda"] = True

    return TrainingArguments(**kwargs)


def main():
    args = parse_args()
    _patch_accelerate_unwrap_model()

    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"\n{'='*60}")
    print(f"CPT Training: {args.run_name or args.model}")
    print(f"  lang_variant : {args.lang_variant}")
    print(f"  lora_r       : {args.lora_r}")
    print(f"  lora_alpha   : {args.lora_alpha}")
    print(f"  use_rslora   : {args.use_rslora}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  epochs       : {args.epochs} (0 = use --max_steps directly)")
    print(f"  max_steps    : {args.max_steps or 'auto (computed from epochs)'}")
    print(f"  warmup_ratio : {args.warmup_ratio}")
    print(f"  weight_decay : {args.weight_decay}")
    print(f"  max_grad_norm: {args.max_grad_norm}")
    print(f"{'='*60}\n")

    print(f"Loading dataset from: {args.data_path}")
    dataset_dict = load_from_disk(args.data_path)
    data_stats = {}
    data_stats_path = Path(args.data_path) / "data_stats.json"
    if data_stats_path.is_file():
        with open(data_stats_path) as f:
            data_stats = json.load(f)

    phase1_ds = dataset_dict["train_phase1"]
    phase2_ds = dataset_dict["train_phase2"]
    # eval_target was added later — gracefully handle older preprocessed datasets.
    eval_ds = dataset_dict.get("eval_target") if hasattr(dataset_dict, "get") else None
    if eval_ds is None and "eval_target" in dataset_dict:
        eval_ds = dataset_dict["eval_target"]
    if eval_ds is not None and len(eval_ds) == 0:
        eval_ds = None  # too small to evaluate (smoke datasets); skip eval

    if len(phase1_ds) == 0 or len(phase2_ds) == 0:
        raise ValueError(
            "Both train_phase1 and train_phase2 must contain at least one packed sequence. "
            "Increase the data budget or lower --seq_len during data preparation."
        )

    if args.epochs > 0:
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
        phase2_steps = max(1, (len(phase2_ds) * args.epochs) // effective_batch)
        phase1_steps = max(1, phase2_steps // 9)
        args.max_steps = phase1_steps + phase2_steps
        print(f"Auto-computed max_steps from {args.epochs} epochs over {len(phase2_ds)} phase2 sequences:")
        print(f"  effective_batch : {effective_batch} (batch={args.batch_size}, grad_accum={args.gradient_accumulation_steps}, world_size={world_size})")
    else:
        if args.max_steps is None:
            raise ValueError("Either --epochs > 0 or --max_steps must be set.")
        phase2_steps = max(1, int(args.max_steps * 0.90))
        phase1_steps = args.max_steps - phase2_steps

    run_metadata = {
        "run_name": args.run_name,
        "model": args.model,
        "dataset_path": args.data_path,
        "dataset_id": data_stats.get("dataset_id"),
        "experiment": data_stats.get("experiment"),
        "lang_variant": args.lang_variant,
        "max_steps": args.max_steps,
        "phase1_steps": phase1_steps,
        "phase2_steps": phase2_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "total_words": data_stats.get("total_words"),
        "total_tokens": data_stats.get("total_tokens"),
        "tokens_per_word": data_stats.get("tokens_per_word"),
        "epochs": args.epochs,
        "english_ratio_requested": data_stats.get("english_ratio_requested"),
        "english_ratio_actual": data_stats.get("english_ratio_actual"),
    }

    print(f"\nCurriculum schedule:")
    print(f"  Phase 1 (English only): {phase1_steps} steps ({len(phase1_ds)} packed seqs)")
    print(f"  Phase 2 (target only):  {phase2_steps} steps ({len(phase2_ds)} packed seqs)")
    print(f"  Total max_steps:        {args.max_steps}")
    print()

    output_dir = args.output_dir
    phase1_dir = os.path.join(output_dir, "phase1")
    phase2_dir = os.path.join(output_dir, "phase2")
    phase1_final_dir = Path(output_dir) / "phase1_final"
    final_dir = Path(output_dir) / "final"

    if adapter_is_complete(final_dir):
        print(f"Final adapter already exists, skipping training: {final_dir}")
        return

    base_model, tokenizer = load_base_model_and_tokenizer(args.model, args.cpu)

    if adapter_is_complete(phase1_final_dir):
        model = load_lora_for_training(base_model, phase1_final_dir)
        skip_phase1 = True
    else:
        model = apply_lora(
            base_model, args.lora_r, args.lora_alpha,
            args.lora_dropout, args.use_rslora,
        )
        skip_phase1 = False

    # ------------------------------------------------------------------
    # Phase 1 training (English anti-forgetting injection)
    # ------------------------------------------------------------------
    if skip_phase1:
        print(f"Phase 1 final adapter already exists, skipping Phase 1: {phase1_final_dir}")
        p1_trainer = None
    else:
        print(f"{'='*60}")
        print(f"Phase 1: English + target language ({phase1_steps} steps)")
        print(f"{'='*60}")

        resume_p1 = find_latest_checkpoint(phase1_dir)
        if resume_p1:
            print(f"Resuming Phase 1 from checkpoint: {resume_p1}")

        p1_args = make_training_args(
            args, phase1_dir, phase1_steps,
            run_name=f"{args.run_name or 'cpt'}_phase1",
        )
        p1_trainer = make_trainer(model, tokenizer, phase1_ds, p1_args)
        add_metrics_callback(
            p1_trainer,
            Path(output_dir) / "metrics" / "train_metrics.jsonl",
            {**run_metadata, "phase": "phase1", "phase_steps": phase1_steps, "phase_start_step": 0},
        )
        p1_trainer.train(resume_from_checkpoint=resume_p1)

        if p1_trainer.is_world_process_zero():
            model.save_pretrained(phase1_final_dir, safe_serialization=True, max_shard_size="2GB")
            tokenizer.save_pretrained(phase1_final_dir)
            print(f"Phase 1 final adapter saved to: {phase1_final_dir}")
        if hasattr(p1_trainer, "accelerator"):
            p1_trainer.accelerator.wait_for_everyone()

        print(f"\nPhase 1 complete.\n")

    # ------------------------------------------------------------------
    # Phase 2 training (target language only)
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"Phase 2: Target language only ({phase2_steps} steps)")
    print(f"{'='*60}")

    resume_p2 = find_latest_checkpoint(phase2_dir)
    if resume_p2:
        print(f"Resuming Phase 2 from checkpoint: {resume_p2}")

    p2_args = make_training_args(
        args, phase2_dir, phase2_steps,
        run_name=f"{args.run_name or 'cpt'}_phase2",
        with_eval=eval_ds is not None,
    )
    p2_trainer = make_trainer(model, tokenizer, phase2_ds, p2_args, eval_dataset=eval_ds)
    add_metrics_callback(
        p2_trainer,
        Path(output_dir) / "metrics" / "train_metrics.jsonl",
        {**run_metadata, "phase": "phase2", "phase_steps": phase2_steps, "phase_start_step": phase1_steps},
    )

    p2_trainer.train(resume_from_checkpoint=resume_p2)

    print(f"\nPhase 2 complete.\n")

    # ------------------------------------------------------------------
    # Save final adapter
    # ------------------------------------------------------------------
    if p2_trainer.is_world_process_zero():
        model.save_pretrained(final_dir, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(final_dir)
        print(f"Final adapter saved to: {final_dir}")
    if hasattr(p2_trainer, "accelerator"):
        p2_trainer.accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # Write grid_search_result.json (used by pick_best_grid.sh)
    # ------------------------------------------------------------------
    # Prefer the most recent intermediate "loss" log; fall back to the final
    # summary's "train_loss" if the run was too short to hit logging_steps.
    final_train_loss = None
    final_eval_loss = None
    if hasattr(p2_trainer, "state") and p2_trainer.state.log_history:
        for entry in reversed(p2_trainer.state.log_history):
            if "loss" in entry:
                final_train_loss = entry["loss"]
                break
        if final_train_loss is None:
            for entry in reversed(p2_trainer.state.log_history):
                if "train_loss" in entry:
                    final_train_loss = entry["train_loss"]
                    break
        for entry in reversed(p2_trainer.state.log_history):
            if "eval_loss" in entry:
                final_eval_loss = entry["eval_loss"]
                break

    result = {
        "run_label": args.run_label,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "final_train_loss": final_train_loss,
        "final_eval_loss": final_eval_loss,
        "lang_variant": args.lang_variant,
        "max_steps": args.max_steps,
    }
    result_path = Path(output_dir) / "grid_search_result.json"
    if p2_trainer.is_world_process_zero():
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Grid search result written to: {result_path}")
        print(json.dumps(result, indent=2))
    if hasattr(p2_trainer, "accelerator"):
        p2_trainer.accelerator.wait_for_everyone()

    write_metrics_summary(Path(output_dir), p1_trainer, p2_trainer, run_metadata)

    print(f"\n{'='*60}")
    print(f"Training complete! Adapter: {final_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
