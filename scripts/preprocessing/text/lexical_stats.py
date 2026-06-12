#!/usr/bin/env python3
"""
Compute lexical statistics over the Finnegans Wake corpus.

Outputs:
  - Total tokens (whitespace-split, punctuation-preserving)
  - Vocabulary size (unique types)
  - Type-to-token ratio
  - Hapax legomena count and rate
  - Shannon entropy over unigram distribution (bits per token)
  - Top-N most frequent tokens

No lowercasing, stemming, or punctuation stripping — consistent with
preprocessing decisions documented in scripts/preprocessing/README.md.

Usage:
    python scripts/preprocessing/lexical_stats.py
    python scripts/preprocessing/lexical_stats.py --top 30
    python scripts/preprocessing/lexical_stats.py --json stats.json
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.corpus import iter_lines  # noqa: E402


def collect_tokens() -> list[str]:
    """Collect tokens, rejoining words split by line-break hyphens in main text.

    If the last token of a line's text ends with a single '-', it is a
    line-break hyphen: the token is carried forward and prepended to the first
    token of the next line. Double-hyphens ('--', Joyce's em-dash) are not
    treated as line breaks. Margin fields are tokenized independently.
    """
    tokens: list[str] = []
    carry = ""

    for line in iter_lines():
        if line.text:
            toks = line.text.split()
            if toks:
                if carry:
                    toks[0] = carry + toks[0]
                    carry = ""
                last = toks[-1]
                if last.endswith("-") and not last.endswith("--"):
                    # Drop the typographic line-break hyphen.
                    carry = last[:-1]
                    toks = toks[:-1]
                tokens.extend(toks)

        if line.left_margin:
            tokens.extend(line.left_margin.split())
        if line.right_margin:
            tokens.extend(line.right_margin.split())

    if carry:
        tokens.append(carry)

    return tokens


def shannon_entropy(counts: Counter) -> float:
    """Shannon entropy in bits over the unigram distribution."""
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexical statistics for Finnegans Wake corpus")
    parser.add_argument("--top", type=int, default=20, help="Show N most frequent tokens")
    parser.add_argument("--json", metavar="PATH", help="Write full stats to JSON file")
    args = parser.parse_args()

    print("Loading corpus...", flush=True)
    tokens = collect_tokens()
    counts = Counter(tokens)

    total_tokens = len(tokens)
    vocab_size = len(counts)
    hapaxes = [w for w, c in counts.items() if c == 1]
    hapax_count = len(hapaxes)
    entropy = shannon_entropy(counts)

    print(f"\n{'='*50}")
    print("FINNEGANS WAKE — LEXICAL STATISTICS")
    print(f"{'='*50}")
    print(f"Total tokens      : {total_tokens:>10,}")
    print(f"Vocabulary (types): {vocab_size:>10,}")
    print(f"Type/token ratio  : {vocab_size/total_tokens:>10.4f}")
    print(f"Hapax legomena    : {hapax_count:>10,}  ({100*hapax_count/vocab_size:.1f}% of types)")
    print(f"Shannon entropy   : {entropy:>10.4f} bits/token")
    print(f"\nTop {args.top} most frequent tokens:")
    for rank, (word, count) in enumerate(counts.most_common(args.top), 1):
        print(f"  {rank:>3}. {word:<30} {count:>6}")

    if args.json:
        out = {
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "type_token_ratio": round(vocab_size / total_tokens, 6),
            "hapax_count": hapax_count,
            "hapax_rate_of_types": round(hapax_count / vocab_size, 6),
            "shannon_entropy_bits": round(entropy, 6),
            "top_tokens": counts.most_common(args.top),
        }
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nFull stats written to {args.json}")


if __name__ == "__main__":
    main()
