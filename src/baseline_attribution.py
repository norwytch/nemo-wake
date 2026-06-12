#!/usr/bin/env python3
"""Classical phoneme-substrate baselines for idea 2 phonotactic attribution.

(1) Frequency baseline: predict argmax_L P(L) over the training prior. This is
    the trivial baseline; under no cap it collapses to "always predict en-us"
    (~98% of FW tokens) and under the class-balanced cap it is a 1-of-5
    uniform random equivalent.

(2) Phoneme-bigram LangID baseline (classical 1990s LangID): per language L,
    estimate a unigram distribution over phoneme bigrams from the training
    data with Laplace smoothing. At probe, score a line as
        sum_{b in line bigrams} log P(b | L)
    and predict argmax_L. This is the strong baseline NEMO has to beat — it
    uses the exact same substrate (en-us bigrams) and the same training
    instances. If NEMO loses to this, the biological-plausibility argument
    has no quantitative leg.

Reads the same lexicon and probes the same held-out lines as
replicate_multilingual_context.py so all numbers are apples-to-apples.

Run with two modes per invocation: uncapped (naive freq baseline) AND with
--max-per-lang matching the NEMO run.

Usage:
    python src/baseline_attribution.py --lexicon lexicons/i6_multilingual_top5.json \\
        --output results/baselines_i6_uncapped.json
    python src/baseline_attribution.py --lexicon lexicons/i6_multilingual_top5.json \\
        --max-per-lang 100 \\
        --output results/baselines_i6_cap100.json
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lexicon", type=Path, required=True)
    p.add_argument("--ngram", type=int, default=2, help="Phoneme n-gram size (default bigrams).")
    p.add_argument("--max-per-lang", type=int, default=None,
                   help="Optional class-balanced cap to match the NEMO run.")
    p.add_argument("--smoothing", type=float, default=1.0, help="Laplace alpha.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def word_units(phonemes, n):
    if n <= 1:
        return list(phonemes)
    if len(phonemes) < n:
        return ["|".join(phonemes)]
    return ["|".join(phonemes[i:i + n]) for i in range(len(phonemes) - n + 1)]


def per_lang_recall(results, languages):
    tp = Counter()
    nt = Counter()
    for r in results:
        for L in r["gold_languages"]:
            nt[L] += 1
            if r["top1"] == L:
                tp[L] += 1
    return {L: (tp[L], nt[L]) for L in languages}


def summarize(name, results, languages):
    n = len(results)
    in_gold = sum(1 for r in results if r["top1"] in r["gold_languages"])
    print(f"\n=== {name} ===")
    print(f"  Overall top-1 ∈ gold: {in_gold}/{n} = {100*in_gold/n:.1f}%")
    recall = per_lang_recall(results, languages)
    for L in languages:
        tp, nt = recall[L]
        if nt:
            print(f"    {L:8s} recall {tp}/{nt} = {100*tp/nt:.1f}%")
    foreign = [r for r in results if set(r["gold_languages"]) - {"en-us"}]
    if foreign:
        f_in_gold = sum(1 for r in foreign if r["top1"] in r["gold_languages"])
        print(f"    foreign-gold subset {f_in_gold}/{len(foreign)} = "
              f"{100*f_in_gold/len(foreign):.1f}%")


def main():
    args = parse_args()
    data = json.loads(args.lexicon.read_text())
    languages = list(data["languages"])
    train_all = data["training_words"]
    probe_lines = data["probe_lines"]

    if args.max_per_lang is not None:
        rng = random.Random(args.seed)
        by_lang: dict[str, list] = {L: [] for L in languages}
        for w in train_all:
            if w["language"] in by_lang:
                by_lang[w["language"]].append(w)
        train = []
        for L in languages:
            items = by_lang[L]
            rng.shuffle(items)
            train.extend(items[: args.max_per_lang])
    else:
        train = train_all

    lang_counts: Counter[str] = Counter()
    bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_bigrams: set[str] = set()
    for w in train:
        L = w["language"]
        lang_counts[L] += 1
        for b in word_units(w["phonemes"], args.ngram):
            bigram_counts[L][b] += 1
            all_bigrams.add(b)

    V = max(len(all_bigrams), 1)
    total_per_lang = {L: sum(bigram_counts[L].values()) for L in languages}

    print(f"Training mode: {'capped at ' + str(args.max_per_lang) if args.max_per_lang else 'UNCAPPED (naive)'}")
    print(f"Training instances per language:")
    for L in languages:
        print(f"  {L:8s} {lang_counts[L]:>5} instances, {total_per_lang[L]:>5} bigrams")
    print(f"Bigram vocab |V|={V}, Laplace α={args.smoothing}")

    prior_argmax = max(languages, key=lambda L: lang_counts[L])
    print(f"\nFrequency baseline predicts: {prior_argmax!r}")

    # Precompute log-likelihoods
    log_p: dict[str, dict[str, float]] = {L: {} for L in languages}
    log_default: dict[str, float] = {}
    for L in languages:
        Z = total_per_lang[L] + args.smoothing * V
        log_default[L] = math.log(args.smoothing / Z) if Z > 0 else float("-inf")
        for b in all_bigrams:
            c = bigram_counts[L].get(b, 0)
            log_p[L][b] = math.log((c + args.smoothing) / Z) if Z > 0 else float("-inf")

    # Predict for each probe line
    freq_results = []
    bigram_results = []
    for ln in probe_lines:
        bigrams_in_line = []
        for t in ln["tokens"]:
            bigrams_in_line.extend(word_units(t["phonemes"], args.ngram))
        gold = ln["gold_languages"]
        freq_results.append({
            "page_line": ln["page_line"], "split": ln["split"],
            "gold_languages": gold, "top1": prior_argmax,
            "profile": [{"language": L, "score": lang_counts[L]}
                        for L in sorted(languages, key=lambda L: -lang_counts[L])],
        })
        scores = {L: 0.0 for L in languages}
        for b in bigrams_in_line:
            for L in languages:
                scores[L] += log_p[L].get(b, log_default[L])
        ranked = sorted(scores, key=lambda L: -scores[L])
        bigram_results.append({
            "page_line": ln["page_line"], "split": ln["split"],
            "gold_languages": gold, "top1": ranked[0],
            "profile": [{"language": L, "score": round(scores[L], 4)} for L in ranked],
        })

    summarize("Frequency baseline", freq_results, languages)
    summarize("Bigram LangID baseline", bigram_results, languages)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "experiment": "baselines",
        "lexicon_path": str(args.lexicon),
        "ngram": args.ngram,
        "max_per_lang": args.max_per_lang,
        "smoothing": args.smoothing,
        "languages": languages,
        "training_instances_by_language": dict(lang_counts),
        "training_bigrams_by_language": total_per_lang,
        "bigram_vocab_size": V,
        "frequency_results": freq_results,
        "bigram_langid_results": bigram_results,
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
