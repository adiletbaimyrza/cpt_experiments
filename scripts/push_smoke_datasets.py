"""
Create and push tiny smoke-test datasets to HuggingFace Hub.

Run this once locally before testing the full pipeline on Helios.
Each dataset has ~500 samples of ~100 words → ~50K words total.
That produces ~50-150 packed sequences per language, enough for the
grid search (500 steps, cycles the data) and a fast full-training run
(~7-14 steps with --epochs 3).

Usage:
    python scripts/push_smoke_datasets.py --hf_org YOUR_HF_USERNAME
    python scripts/push_smoke_datasets.py --hf_org YOUR_HF_USERNAME --private

After running, copy the printed .env block into your .env on Helios.
"""

import argparse

from datasets import Dataset

N_SAMPLES = 500

# Enough variety that packing doesn't collapse to identical sequences
TEMPLATES = {
    "ky": (
        "Бул {i}-номерлүү сыноо текст. Кыргыз тилиндеги сүйлөмдөр машиналык үйрөнүү "
        "изилдөөсүн текшерүү үчүн колдонулат. Модель {i} санын жана тилдин структурасын "
        "үйрөнөт. Улантуу үчүн дагы бир нече сүйлөм кошулат. "
    ),
    "kz": (
        "Бұл {i}-нөмірлі сынақ мәтін. Қазақ тіліндегі сөйлемдер машиналық оқытуды "
        "тексеру үшін қолданылады. Модель {i} санын және тіл құрылымын үйренеді. "
        "Жалғастыру үшін бірнеше сөйлем қосылады. "
    ),
    "pl": (
        "To jest tekst testowy numer {i}. Zdania w języku polskim służą do testowania "
        "uczenia maszynowego. Model uczy się liczby {i} i struktury języka. "
        "Dodajemy kilka zdań dla kontynuacji. "
    ),
    "en": (
        "This is test document number {i}. English sentences are used to verify the "
        "machine learning pipeline. The model learns the number {i} and language structure. "
        "Adding more sentences for continuity and length. "
    ),
}

DATASET_NAMES = {
    "ky": "cpt-smoke-ky",
    "kz": "cpt-smoke-kz",
    "pl": "cpt-smoke-pl",
    "en": "cpt-smoke-en",
}


def make_samples(lang: str, n: int) -> list[str]:
    template = TEMPLATES[lang]
    samples = []
    for i in range(n):
        # Repeat template a few times per sample to get ~100 words
        text = (template * 4).format(i=i)
        samples.append(text.strip())
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_org", required=True,
                        help="HuggingFace username or org to push datasets under")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES,
                        help=f"Samples per dataset (default: {N_SAMPLES})")
    parser.add_argument("--private", action="store_true",
                        help="Push as private datasets")
    args = parser.parse_args()

    repo_ids = {}
    for lang, name in DATASET_NAMES.items():
        repo_id = f"{args.hf_org}/{name}"
        print(f"Pushing {repo_id} ({args.n_samples} samples, private={args.private})...")
        samples = make_samples(lang, args.n_samples)
        ds = Dataset.from_dict({"text": samples})
        ds.push_to_hub(repo_id, private=args.private)
        repo_ids[lang] = repo_id
        print(f"  Done — {len(samples)} samples\n")

    print("=" * 50)
    print("Add to your .env on Helios:")
    print("=" * 50)
    print(f"CPT_DATASET_FT_KY_WORDS={repo_ids['ky']}")
    print(f"CPT_DATASET_FT_KZ_WORDS={repo_ids['kz']}")
    print(f"CPT_DATASET_FT_PL_WORDS={repo_ids['pl']}")
    print(f"CPT_DATASET_FT_KY_TOKENS={repo_ids['ky']}")
    print(f"CPT_DATASET_FT_KZ_TOKENS={repo_ids['kz']}")
    print(f"CPT_DATASET_FT_PL_TOKENS={repo_ids['pl']}")
    print(f"CPT_DATASET_ENGLISH={repo_ids['en']}")


if __name__ == "__main__":
    main()
