#!/usr/bin/env python3
"""Build Tier-1 multilingual phonotactic-attribution lexicon for idea 2.

For one FW chapter (default I.6), extracts FWEET-tagged tokens from the
joyce-derived IPA artifact for the Top-N target languages and emits:

  training_words: one per (token, language) instance. Foreign-language
                  instances are emitted when FWEET tags the token with that
                  language at HIGH confidence (orthographic-similarity match
                  from generate_joyce_derived_ipa). Tokens with no high-conf
                  foreign tag get an English (en-us) instance by default.
  probe_lines:    held-out (val + test) page-lines with their tokens and a
                  multi-label gold language set (any FWEET-tagged language
                  active on the page-line, restricted to the target set).

Phoneme source for BOTH training and probe: en-us-baseline.
This is deliberate. Using per-language IPA at probe time would leak the
language label; using it at train time would mean CONTEXT_L learns a
native-phonology template that wouldn't match the English-realized probe.
The CONTEXT_L assembly therefore learns: "Joyce-style English-pronounced
phonotactic patterns indicate foreign source L." That's the same competence
an English-speaking Wake reader exercises (one doesn't need to know French
to hear 'pas encore' inside 'passencore').

Split: deterministic 80/10/10 via shared.corpus.line_split. Training =
train; probe lines = val + test combined.

Consumed by: src/replicate_multilingual_context.py (Day 3-5).

Usage:
    python scripts/preprocessing/text/build_multilingual_lexicon.py \\
        --book 1 --episode 6 \\
        --output projects/01-biological-language-organ/lexicons/i6_multilingual_top5.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from shared.corpus import line_split  # noqa: E402
from shared.fweet import load_by_page_line  # noqa: E402

JOYCE_IPA_DIR = REPO_ROOT / "data" / "ipa" / "joyce-derived"
DEFAULT_LANGUAGES = ("en-us", "de", "fr-fr", "la", "it")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", type=int, default=1)
    p.add_argument("--episode", type=int, default=6)
    p.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES),
                   help="Target language espeak codes (default: en-us de fr-fr la it).")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    target_set = set(args.languages)
    print(f"Target languages: {sorted(target_set)}")

    ipa_path = JOYCE_IPA_DIR / f"book{args.book:02d}_ep{args.episode:02d}.jsonl"
    if not ipa_path.exists():
        sys.exit(f"Missing {ipa_path}. Run generate_joyce_derived_ipa.py first.")

    print(f"Loading joyce-derived IPA from {ipa_path.relative_to(REPO_ROOT)}...")
    ipa_per_pl: dict[str, list[dict]] = defaultdict(list)
    with ipa_path.open() as f:
        for line in f:
            rec = json.loads(line)
            ipa_per_pl[rec["page_line"]].append(rec)
    print(f"  {len(ipa_per_pl):,} page-lines, "
          f"{sum(len(v) for v in ipa_per_pl.values()):,} tokens")

    print("Loading FWEET index...")
    fweet_by_pl = load_by_page_line()

    training_words: list[dict] = []
    probe_lines: list[dict] = []
    split_counts: Counter[str] = Counter()
    lang_counts_train: Counter[str] = Counter()
    lang_counts_gold: Counter[str] = Counter()
    n_probe_lines_with_foreign_gold = 0

    for pl in sorted(ipa_per_pl):
        split = line_split(pl)
        split_counts[split] += 1
        recs = ipa_per_pl[pl]

        if split == "train":
            for rec in recs:
                en_phonemes = rec["ipa"].get("en-us-baseline", [])
                if not en_phonemes:
                    continue
                fweet_langs = rec.get("fweet_languages", {})
                high_foreign = sorted(
                    L for L, info in fweet_langs.items()
                    if info.get("confidence") == "high"
                    and L != "en-us"
                    and L in target_set
                )
                if high_foreign:
                    for L in high_foreign:
                        training_words.append({
                            "text": rec["orth"],
                            "page_line": pl,
                            "language": L,
                            "phonemes": en_phonemes,
                        })
                        lang_counts_train[L] += 1
                elif "en-us" in target_set:
                    training_words.append({
                        "text": rec["orth"],
                        "page_line": pl,
                        "language": "en-us",
                        "phonemes": en_phonemes,
                    })
                    lang_counts_train["en-us"] += 1
            continue

        # split in {"val", "test"} — held-out probe line
        tokens = []
        for rec in recs:
            en_phonemes = rec["ipa"].get("en-us-baseline", [])
            if not en_phonemes:
                continue
            tokens.append({"text": rec["orth"], "phonemes": en_phonemes})
        if not tokens:
            continue

        gold: set[str] = set()
        for elu in fweet_by_pl.get(pl, ()):
            if elu.espeak_code and elu.espeak_code in target_set:
                gold.add(elu.espeak_code)
        if not gold and "en-us" in target_set:
            gold.add("en-us")
        if gold - {"en-us"}:
            n_probe_lines_with_foreign_gold += 1
        for L in gold:
            lang_counts_gold[L] += 1

        probe_lines.append({
            "page_line": pl,
            "split": split,
            "tokens": tokens,
            "gold_languages": sorted(gold),
        })

    print(f"\nLine splits: train={split_counts['train']}  "
          f"val={split_counts['val']}  test={split_counts['test']}")
    print(f"\nTraining instances: {len(training_words):,}")
    for L, c in lang_counts_train.most_common():
        print(f"  {L:8s} {c:>5,}")
    print(f"\nProbe lines (val+test): {len(probe_lines):,}")
    print(f"  with ≥1 non-English gold: {n_probe_lines_with_foreign_gold:,} "
          f"({100*n_probe_lines_with_foreign_gold/max(len(probe_lines),1):.1f}%)")
    print(f"Per-language gold occurrences across probe lines:")
    for L, c in lang_counts_gold.most_common():
        print(f"  {L:8s} {c:>5,}")

    lexicon = {
        "name": f"multilingual_book{args.book:02d}_ep{args.episode:02d}_top{len(target_set)}",
        "description": (
            "Tier-1 phonotactic attribution lexicon (idea 2). Phonemes are "
            "en-us-baseline; CONTEXT_L learns Joyce's English-realized "
            "phonotactics of foreign-origin words. Training = train split; "
            "probe = val+test."
        ),
        "source_artifact": str(ipa_path.relative_to(REPO_ROOT)),
        "languages": sorted(target_set),
        "split_counts": dict(split_counts),
        "training_instance_counts_by_language": dict(lang_counts_train),
        "probe_gold_counts_by_language": dict(lang_counts_gold),
        "num_training_words": len(training_words),
        "num_probe_lines": len(probe_lines),
        "num_probe_lines_with_foreign_gold": n_probe_lines_with_foreign_gold,
        "training_words": training_words,
        "probe_lines": probe_lines,
    }
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2))
    try:
        rel = out_path.relative_to(REPO_ROOT)
        print(f"\nWrote {rel}")
    except ValueError:
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
