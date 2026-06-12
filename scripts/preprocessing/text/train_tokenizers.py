#!/usr/bin/env python3
"""
Train BPE, Unigram LM, and Morfessor (Baseline + FlatCat) on the Wake corpus.

Writes model files to data/tokenizers/ (gitignored, reproducible from this script).
Loaders live in shared.tokenizers: load_bpe(), load_unigram(), load_morfessor(),
load_flatcat().

Usage:
    python scripts/preprocessing/train_tokenizers.py
    python scripts/preprocessing/train_tokenizers.py --vocab-sizes 4000 8000
    python scripts/preprocessing/train_tokenizers.py --only bpe
    python scripts/preprocessing/train_tokenizers.py --only unigram
    python scripts/preprocessing/train_tokenizers.py --only morfessor
    python scripts/preprocessing/train_tokenizers.py --only flatcat
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.tokenizers import (  # noqa: E402
    DEFAULT_VOCAB_SIZES,
    FLATCAT_PATH,
    MORFESSOR_PATH,
    bpe_path,
    train_bpe,
    train_flatcat,
    train_morfessor,
    train_unigram,
    unigram_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train BPE, Unigram LM, and Morfessor on the Wake"
    )
    parser.add_argument(
        "--only",
        choices=["bpe", "unigram", "morfessor", "flatcat", "superbpe"],
        help=(
            "Train only one (default: train all in-venv tokenizers). "
            "FlatCat requires Morfessor first. SuperBPE requires a parallel "
            "venv with the upstream fork — see shared.tokenizers.train_superbpe."
        ),
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_VOCAB_SIZES),
        help=f"BPE/Unigram vocab sizes to sweep (default: {list(DEFAULT_VOCAB_SIZES)})",
    )
    args = parser.parse_args()

    if args.only in (None, "bpe"):
        for vocab_size in args.vocab_sizes:
            print(f"Training BPE (vocab_size={vocab_size})...", flush=True)
            train_bpe(vocab_size=vocab_size)
            print(f"  → {bpe_path(vocab_size)}")

    if args.only in (None, "unigram"):
        for vocab_size in args.vocab_sizes:
            print(f"Training Unigram LM (vocab_size={vocab_size})...", flush=True)
            train_unigram(vocab_size=vocab_size)
            print(f"  → {unigram_path(vocab_size)}")

    if args.only in (None, "morfessor"):
        print("Training Morfessor Baseline...", flush=True)
        train_morfessor()
        print(f"  → {MORFESSOR_PATH}")

    if args.only in (None, "flatcat"):
        print("Training Morfessor FlatCat (HMM over Baseline segmentations)...", flush=True)
        train_flatcat()
        print(f"  → {FLATCAT_PATH}")

    if args.only == "superbpe":
        # Deliberately NOT in the default (no-flag) run — requires a parallel venv.
        from shared.tokenizers import train_superbpe  # noqa: E402
        for vocab_size in args.vocab_sizes:
            train_superbpe(vocab_size=vocab_size)


if __name__ == "__main__":
    main()
