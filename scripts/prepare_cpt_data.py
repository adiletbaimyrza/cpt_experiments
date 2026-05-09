"""
CPT data preparation script.

Loads a HuggingFace dataset, shuffles before budget cutoff (to avoid domain confound
between Exp A and Exp B), counts words/tokens until budget, sequence-packs into 2048-token
chunks, and saves three splits for ALL lang variants:
  train_phase1  — English only (first 10% of training steps; anchors model capabilities)
  train_phase2  — target-language only (remaining 90% of training steps)
  eval_target   — held-out 5% of target texts for grid winner selection / early stopping

Curriculum strategy (per paper):
  The model trains on 100% English for the first 10% of steps, then switches to 100%
  target language. This anchors reasoning and in-context learning capabilities before
  adaptation, preventing catastrophic forgetting without needing English in Phase 2.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", required=True)
    p.add_argument("--tokenizer_id", required=True)
    p.add_argument("--lang_variant", required=True,
                   choices=["FT-KY", "FT-KZ", "FT-PL"])
    p.add_argument("--experiment", required=True, choices=["words", "tokens"])
    p.add_argument("--word_budget", type=int, default=100_000_000)
    p.add_argument("--token_budget", type=int, default=100_000_000)
    p.add_argument("--english_dataset_id", required=True,
                   help="HF dataset ID for English Phase 1 (same dataset for all languages and experiments)")
    p.add_argument("--english_word_budget", type=int, default=100_000_000,
                   help="Word budget for English Phase 1 data (default: 100M words). "
                        "Fixed across all languages and experiments.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text_column", type=str, default="text")
    return p.parse_args()


def collect_tokens_until_budget(ds_iter, tokenizer, experiment, word_budget, token_budget, text_col):
    """Iterate shuffled dataset, collect texts until word or token budget is hit."""
    texts = []
    total_words = 0
    total_tokens = 0

    for record in ds_iter:
        text = record.get(text_col, "")
        if not text or not text.strip():
            continue

        words = len(text.split())
        toks = len(tokenizer.encode(text, add_special_tokens=False))

        if experiment == "words":
            if total_words + words > word_budget:
                break
            total_words += words
            total_tokens += toks
        else:
            if total_tokens + toks > token_budget:
                break
            total_words += words
            total_tokens += toks

        texts.append(text)

    return texts, total_words, total_tokens


def chunks_to_dataset(chunks):
    """Build a Dataset with stable features, including for empty splits."""
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
        "labels": Sequence(Value("int32")),
    })
    if chunks:
        return Dataset.from_list(chunks, features=features)
    return Dataset.from_dict(
        {"input_ids": [], "attention_mask": [], "labels": []},
        features=features,
    )


def pack_texts(texts, tokenizer, seq_len):
    """
    Concatenate all texts with EOS separator, chunk into fixed seq_len windows.
    Returns list of dicts with keys: input_ids, attention_mask, labels.
    """
    eos = tokenizer.eos_token_id
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(eos)

    chunks = []
    for start in range(0, len(all_ids) - seq_len + 1, seq_len):
        chunk = all_ids[start:start + seq_len]
        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1] * seq_len,
            "labels": chunk,
        })

    return chunks


def main():
    args = parse_args()

    print(f"CPT Data Preparation")
    print(f"  lang_variant      : {args.lang_variant}")
    print(f"  experiment        : {args.experiment}")
    print(f"  dataset_id        : {args.dataset_id}")
    print(f"  english_dataset   : {args.english_dataset_id}")
    print(f"  english_word_budget: {args.english_word_budget:,} words (fixed, same for all languages/experiments)")
    print(f"  seq_len           : {args.seq_len}")
    print(f"  seed              : {args.seed}")
    print()

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------------------------------------------------
    # Load and shuffle target dataset BEFORE budget cutoff
    # (ensures Exp A and Exp B see the same document distribution)
    # -------------------------------------------------------------------------
    print(f"Loading target dataset: {args.dataset_id}")
    ds = load_dataset(args.dataset_id, split="train")
    print(f"  Total records (pre-filter): {len(ds)}")
    ds = ds.shuffle(seed=args.seed)

    print(f"Collecting target texts until {args.experiment} budget...")
    target_texts, total_words, total_tokens = collect_tokens_until_budget(
        iter(ds), tokenizer, args.experiment,
        args.word_budget, args.token_budget, args.text_column
    )
    tokens_per_word = total_tokens / max(total_words, 1)
    print(f"  Collected docs   : {len(target_texts)}")
    print(f"  Total words      : {total_words:,}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  tokens_per_word  : {tokens_per_word:.3f}  (KY~3.0, KZ~2.5, PL~1.25)")
    print()

    # -------------------------------------------------------------------------
    # Hold out 5% of target texts (capped at 5K docs) for eval / grid selection.
    # Texts are already shuffled, so a deterministic prefix slice is sufficient.
    # -------------------------------------------------------------------------
    if len(target_texts) > 1:
        n_eval = min(max(1, len(target_texts) // 20), 5000)
    else:
        n_eval = 0
    eval_texts = target_texts[:n_eval]
    train_texts = target_texts[n_eval:]

    if not train_texts:
        raise ValueError(
            "No training texts were collected. Increase the budget or check the text column."
        )

    print(f"Held-out eval split:")
    print(f"  eval docs:  {len(eval_texts)}")
    print(f"  train docs: {len(train_texts)}")
    print()

    # -------------------------------------------------------------------------
    # Load English data up to fixed word budget (same for all languages/experiments)
    # -------------------------------------------------------------------------
    print(f"Loading English dataset: {args.english_dataset_id}")
    en_ds = load_dataset(args.english_dataset_id, split="train")
    print(f"  Total records (pre-filter): {len(en_ds)}")
    en_ds = en_ds.shuffle(seed=args.seed + 1)
    english_texts, english_words, english_tokens = collect_tokens_until_budget(
        iter(en_ds), tokenizer, "words",
        args.english_word_budget, args.english_word_budget, args.text_column
    )
    print(f"  English docs used   : {len(english_texts):,}")
    print(f"  English words used  : {english_words:,}")
    print(f"  English tokens used : {english_tokens:,}")
    print()

    # Phase 1: English only (100% English — anchors model capabilities)
    phase1_texts = english_texts
    # Phase 2: target language only (100% target)
    phase2_texts = train_texts

    print(f"Split sizes (texts before packing):")
    print(f"  train_phase1: {len(phase1_texts)} docs (100% English)")
    print(f"  train_phase2: {len(phase2_texts)} docs (100% target-lang)")
    print()

    # -------------------------------------------------------------------------
    # Sequence packing: eliminates padding waste (~30-50% compute savings)
    # -------------------------------------------------------------------------
    print(f"Packing sequences into {args.seq_len}-token chunks...")
    p1_chunks = pack_texts(phase1_texts, tokenizer, args.seq_len)
    p2_chunks = pack_texts(phase2_texts, tokenizer, args.seq_len)
    eval_chunks = pack_texts(eval_texts, tokenizer, args.seq_len)

    print(f"  train_phase1 packed sequences: {len(p1_chunks)}")
    print(f"  train_phase2 packed sequences: {len(p2_chunks)}")
    print(f"  eval_target  packed sequences: {len(eval_chunks)}")
    print()

    # -------------------------------------------------------------------------
    # Save as HuggingFace DatasetDict — map-style, DDP-safe
    # -------------------------------------------------------------------------
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict({
        "train_phase1": chunks_to_dataset(p1_chunks),
        "train_phase2": chunks_to_dataset(p2_chunks),
        "eval_target":  chunks_to_dataset(eval_chunks),
    })
    dataset_dict.save_to_disk(str(output_path))
    print(f"Saved DatasetDict to: {output_path}")

    # -------------------------------------------------------------------------
    # Write data_stats.json
    # -------------------------------------------------------------------------
    stats = {
        "lang_variant": args.lang_variant,
        "experiment": args.experiment,
        "dataset_id": args.dataset_id,
        "english_dataset_id": args.english_dataset_id,
        "english_word_budget": args.english_word_budget,
        "english_words_phase1": english_words,
        "english_tokens_phase1": english_tokens,
        "phase1_strategy": "english_only",
        "phase2_strategy": "target_only",
        "total_docs": len(target_texts),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_word": round(tokens_per_word, 4),
        "packed_sequences": {
            "train_phase1": len(p1_chunks),
            "train_phase2": len(p2_chunks),
            "eval_target":  len(eval_chunks),
        },
        "eval_target_docs": len(eval_texts),
        "train_target_docs": len(train_texts),
        "seq_len": args.seq_len,
        "seed": args.seed,
    }
    stats_path = output_path / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to: {stats_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
