"""
CPT data preparation script.

Loads a HuggingFace dataset, shuffles before budget cutoff (to avoid domain confound
between Exp A and Exp B), counts words/tokens until budget, sequence-packs into 2048-token
chunks, and saves three splits for ALL lang variants:
  train_phase1  — target-language + English interleaved (anti-forgetting injection)
  train_phase2  — target-language only (remaining 90% of CPT)

English ratio in Phase 1:
  FT-KY / FT-KZ / FT-PL : ~10%  (minimal anti-forgetting injection)
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
                   help="HF dataset ID for English mix-in (required for all variants)")
    p.add_argument("--english_ratio", type=float, default=None,
                   help="Fraction of Phase 1 tokens that are English. "
                        "Default: 0.1 for all variants")
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


def collect_tokens_until_token_budget(ds_iter, tokenizer, token_budget, text_col):
    """Collect non-empty texts until the next record would exceed token_budget."""
    texts = []
    total_tokens = 0

    for record in ds_iter:
        text = record.get(text_col, "")
        if not text or not text.strip():
            continue

        toks = len(tokenizer.encode(text, add_special_tokens=False)) + 1
        if total_tokens + toks > token_budget:
            break
        total_tokens += toks
        texts.append(text)

    return texts, total_tokens


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


def interleave_texts(target_texts, english_texts, seed):
    """
    Interleave target-language and English texts.
    The English amount should already have been selected by token budget.
    """
    import random
    rng = random.Random(seed)

    combined = target_texts + english_texts
    rng.shuffle(combined)
    return combined


def main():
    args = parse_args()

    if args.english_ratio is None:
        args.english_ratio = 0.1

    print(f"CPT Data Preparation")
    print(f"  lang_variant   : {args.lang_variant}")
    print(f"  experiment     : {args.experiment}")
    print(f"  dataset_id     : {args.dataset_id}")
    print(f"  english_dataset: {args.english_dataset_id}")
    print(f"  english_ratio  : {args.english_ratio} (Phase 1)")
    print(f"  seq_len        : {args.seq_len}")
    print(f"  seed           : {args.seed}")
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
    # Use all target texts for CPT training.
    # -------------------------------------------------------------------------
    train_texts = target_texts

    if not train_texts:
        raise ValueError(
            "No training texts were collected. Increase the budget or check the text column."
        )

    train_token_count = sum(len(tokenizer.encode(text, add_special_tokens=False)) + 1 for text in train_texts)
    english_token_budget = int(train_token_count * args.english_ratio / max(1 - args.english_ratio, 1e-6))

    # -------------------------------------------------------------------------
    # Load only enough English data to approximate the requested Phase 1 token mix
    # -------------------------------------------------------------------------
    print(f"Loading English dataset: {args.english_dataset_id}")
    en_ds = load_dataset(args.english_dataset_id, split="train")
    en_ds = en_ds.shuffle(seed=args.seed + 1)
    english_texts, english_tokens = collect_tokens_until_token_budget(
        iter(en_ds), tokenizer, english_token_budget, args.text_column
    )
    phase1_total_tokens = train_token_count + english_tokens
    actual_english_ratio = english_tokens / max(phase1_total_tokens, 1)
    print(f"  English token budget: {english_token_budget:,}")
    print(f"  English docs used   : {len(english_texts):,}")
    print(f"  English tokens used : {english_tokens:,}")
    print(f"  Actual phase1 ratio : {actual_english_ratio:.4f}")
    print()

    # Phase 1: interleave English into target texts
    phase1_texts = interleave_texts(train_texts, english_texts, args.seed)
    # Phase 2: target language only
    phase2_texts = train_texts

    print(f"Split sizes (texts before packing):")
    print(f"  train_phase1: {len(phase1_texts)} (English token ratio {actual_english_ratio:.4f})")
    print(f"  train_phase2: {len(phase2_texts)} (target-lang only)")
    print()

    # -------------------------------------------------------------------------
    # Sequence packing: eliminates padding waste (~30-50% compute savings)
    # -------------------------------------------------------------------------
    print(f"Packing sequences into {args.seq_len}-token chunks...")
    p1_chunks = pack_texts(phase1_texts, tokenizer, args.seq_len)
    p2_chunks = pack_texts(phase2_texts, tokenizer, args.seq_len)

    print(f"  train_phase1 packed sequences: {len(p1_chunks)}")
    print(f"  train_phase2 packed sequences: {len(p2_chunks)}")
    print()

    # -------------------------------------------------------------------------
    # Save as HuggingFace DatasetDict — map-style, DDP-safe
    # -------------------------------------------------------------------------
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict({
        "train_phase1": chunks_to_dataset(p1_chunks),
        "train_phase2": chunks_to_dataset(p2_chunks),
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
        "english_ratio_requested": args.english_ratio,
        "english_ratio_actual_phase1": round(actual_english_ratio, 4),
        "english_tokens_phase1": english_tokens,
        "total_docs": len(target_texts),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_word": round(tokens_per_word, 4),
        "packed_sequences": {
            "train_phase1": len(p1_chunks),
            "train_phase2": len(p2_chunks),
        },
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
