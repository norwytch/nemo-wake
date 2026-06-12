#!/usr/bin/env python3
"""Multilingual CONTEXT area with one slot per target language (Mitropolsky-2025 MOOD style).

Implements idea 2 (phonotactic recast) Tier 1.

PHON: explicit area with one block per phoneme n-gram (bigrams by default — the
    +64% order-aware substrate from idea 8). Unified inventory across the target
    languages: shared symbols overlap; disjoint symbols (e.g. /ɑ̃/ French-only)
    don't share blocks, so the model learns "language L uses these patterns".
CONTEXT: explicit area with one fixed assembly slot per language. n = L * ctx_k;
    k-cap is ctx_k.

Training: per (token, language) instance, fire token's bigram union into PHON,
fix the corresponding language slot in CONTEXT, project + Hebbian recurrence so
PHON→CONTEXT synapses preferentially weight L's slot neurons. Tokens that have
no high-confidence foreign FWEET tag train into the en-us slot (Joyce wrote in
English by default; FWEET annotates departures).

Probe: per held-out page-line, fire the union of all tokens' bigrams into PHON,
project to CONTEXT. CONTEXT k-cap selects the dominant language slot. Per-slot
overlap = how many of the slot's ctx_k neurons made it into the top-ctx_k.
Top-1 prediction = slot with most overlap.

Class imbalance note: I.6 has ~98% en-us training instances. We subsample en-us
per --max-per-lang so the substrate has to learn phonotactic discrimination, not
just memorize frequency.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from _common import (
    add_nemo_to_path,
    ensure_hash_seed,
    make_run_dir,
    open_tee_log,
    save_json,
    setup_seeded,
)

PHON = "PHON"
CONTEXT = "CONTEXT"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lexicon", type=Path, required=True)
    p.add_argument("--ngram", type=int, default=2,
                   help="Phoneme n-gram size for PHON units. 2 = bigrams (recommended).")
    p.add_argument("--max-per-lang", type=int, default=200,
                   help="Class-balanced cap on training instances per language. "
                        "Default 200 keeps foreign classes intact (max foreign is "
                        "la=86 in I.6) while clamping en-us so it can't dominate.")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--ctx-k", type=int, default=100,
                   help="Neurons per language slot in CONTEXT.")
    p.add_argument("--phon-k", type=int, default=20)
    p.add_argument("--p", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.03)
    p.add_argument("--proj-rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def word_units(phonemes, n):
    if n <= 1:
        return list(phonemes)
    if len(phonemes) < n:
        return ["|".join(phonemes)]
    return ["|".join(phonemes[i:i + n]) for i in range(len(phonemes) - n + 1)]


def load_lexicon(path: Path, ngram: int, max_per_lang: int, seed: int):
    data = json.loads(path.read_text())
    languages = list(data["languages"])
    train_all = data["training_words"]
    probe = data["probe_lines"]

    # Class-balanced subsample
    by_lang: dict[str, list] = {L: [] for L in languages}
    for w in train_all:
        L = w["language"]
        if L in by_lang:
            by_lang[L].append(w)
    rng = random.Random(seed)
    train: list[dict] = []
    for L in languages:
        items = by_lang[L]
        rng.shuffle(items)
        kept = items[:max_per_lang]
        for w in kept:
            w["_units"] = word_units(w["phonemes"], ngram)
        train.extend(kept)

    units = set()
    for w in train:
        units.update(w["_units"])
    for ln in probe:
        line_units = []
        for t in ln["tokens"]:
            tu = word_units(t["phonemes"], ngram)
            t["_units"] = tu
            line_units.extend(tu)
        ln["_units"] = line_units
        units.update(line_units)
    unit_to_idx = {u: i for i, u in enumerate(sorted(units))}
    lang_to_slot = {L: i for i, L in enumerate(languages)}
    return train, probe, languages, unit_to_idx, lang_to_slot


def build_brain(unit_to_idx, n_languages, args):
    import brain
    b = brain.Brain(args.p)
    b.add_explicit_area(PHON, len(unit_to_idx) * args.phon_k, args.phon_k, args.beta)
    b.add_explicit_area(CONTEXT, n_languages * args.ctx_k, args.ctx_k, args.beta)
    return b


def activate_unit_union(b, units, unit_to_idx, phon_k):
    winners = []
    for u in set(units):
        idx = unit_to_idx[u]
        winners.extend(range(idx * phon_k, (idx + 1) * phon_k))
    b.area_by_name[PHON].winners = sorted(winners)
    b.area_by_name[PHON].fix_assembly()


def activate_slot(b, slot_idx, ctx_k):
    a = b.area_by_name[CONTEXT]
    start = slot_idx * ctx_k
    a.winners = list(range(start, start + ctx_k))
    a.fix_assembly()


def clear_all(b):
    for area in (PHON, CONTEXT):
        b.area_by_name[area].unfix_assembly()
        b.area_by_name[area].winners = []


def train_token(b, units, slot_idx, unit_to_idx, args):
    activate_unit_union(b, units, unit_to_idx, args.phon_k)
    activate_slot(b, slot_idx, args.ctx_k)
    b.project({}, {PHON: [CONTEXT]})
    for _ in range(args.proj_rounds):
        b.project({}, {PHON: [CONTEXT], CONTEXT: [PHON, CONTEXT]})
    clear_all(b)


def probe(b, units, unit_to_idx, lang_to_slot, args):
    clear_all(b)
    if not units:
        return [{"language": L, "overlap": 0, "overlap_frac": 0.0}
                for L in lang_to_slot]
    activate_unit_union(b, units, unit_to_idx, args.phon_k)
    b.project({}, {PHON: [CONTEXT]})
    ctx_winners = set(b.area_by_name[CONTEXT].winners)
    profile = []
    for L, slot_idx in lang_to_slot.items():
        slot = set(range(slot_idx * args.ctx_k, (slot_idx + 1) * args.ctx_k))
        ov = len(ctx_winners & slot)
        profile.append({"language": L, "overlap": ov,
                        "overlap_frac": round(ov / args.ctx_k, 3)})
    profile.sort(key=lambda x: -x["overlap"])
    return profile


def main():
    args = parse_args()
    ensure_hash_seed(args.seed)
    run_dir = make_run_dir(f"mlctx_ngram{args.ngram}")
    open_tee_log(run_dir)
    add_nemo_to_path()
    setup_seeded(args.seed)

    train, probe_lines, languages, unit_to_idx, lang_to_slot = load_lexicon(
        args.lexicon, args.ngram, args.max_per_lang, args.seed
    )

    train_counts = Counter(w["language"] for w in train)
    n_languages = len(languages)
    save_json(run_dir, "config.json", {
        "experiment": "multilingual_context",
        "lexicon_path": str(args.lexicon),
        "ngram": args.ngram,
        "languages": languages,
        "lang_to_slot": lang_to_slot,
        "num_units": len(unit_to_idx),
        "phon_n": len(unit_to_idx) * args.phon_k,
        "context_n": n_languages * args.ctx_k,
        "num_training_instances": len(train),
        "training_instances_by_language": dict(train_counts),
        "num_probe_lines": len(probe_lines),
        "max_per_lang": args.max_per_lang,
        "rounds": args.rounds, "ctx_k": args.ctx_k, "phon_k": args.phon_k,
        "p": args.p, "beta": args.beta,
        "proj_rounds": args.proj_rounds, "seed": args.seed,
    })

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] ngram={args.ngram}  units={len(unit_to_idx)}  "
          f"PHON_n={len(unit_to_idx)*args.phon_k}  CONTEXT_n={n_languages*args.ctx_k}",
          flush=True)
    print(f"  Training instances after class-balanced cap (max_per_lang={args.max_per_lang}):")
    for L in languages:
        print(f"    {L:8s} {train_counts[L]:>4}")
    b = build_brain(unit_to_idx, n_languages, args)

    print(f"[{time.time()-t0:6.1f}s] Training {args.rounds} rounds × {len(train)} instances",
          flush=True)
    for r in range(args.rounds):
        for w in train:
            train_token(b, w["_units"], lang_to_slot[w["language"]], unit_to_idx, args)
        print(f"  [{time.time()-t0:6.1f}s] round {r+1}/{args.rounds}", flush=True)

    b.disable_plasticity = True

    print(f"\n[{time.time()-t0:6.1f}s] === Probing {len(probe_lines)} held-out lines ===",
          flush=True)
    probe_results = []
    for ln in probe_lines:
        prof = probe(b, ln["_units"], unit_to_idx, lang_to_slot, args)
        top1 = prof[0]["language"]
        probe_results.append({
            "page_line": ln["page_line"],
            "split": ln["split"],
            "gold_languages": ln["gold_languages"],
            "top1": top1,
            "profile": prof,
        })

    elapsed = time.time() - t0
    save_json(run_dir, "results.json", {
        "experiment": "multilingual_context",
        "ngram": args.ngram,
        "elapsed_seconds": elapsed,
        "languages": languages,
        "lang_to_slot": lang_to_slot,
        "probe_results": probe_results,
    })

    n = len(probe_results)
    in_gold = sum(1 for r in probe_results if r["top1"] in r["gold_languages"])
    print(f"\n=== Held-out attribution (top-1 ∈ gold) ===")
    print(f"  Overall: {in_gold}/{n} = {100*in_gold/n:.1f}%")

    # Per-language recall: among lines where L ∈ gold, how often is L the top-1?
    per_lang_tp = Counter()
    per_lang_n = Counter()
    for r in probe_results:
        for L in r["gold_languages"]:
            per_lang_n[L] += 1
            if r["top1"] == L:
                per_lang_tp[L] += 1
    print(f"\n  Per-language recall (top1 == L | L ∈ gold):")
    for L in languages:
        if per_lang_n[L]:
            print(f"    {L:8s} {per_lang_tp[L]}/{per_lang_n[L]} = "
                  f"{100*per_lang_tp[L]/per_lang_n[L]:.1f}%")

    # Foreign-only subset: probe lines with ≥1 non-en-us gold
    foreign_idx = [r for r in probe_results
                   if set(r["gold_languages"]) - {"en-us"}]
    if foreign_idx:
        f_in_gold = sum(1 for r in foreign_idx if r["top1"] in r["gold_languages"])
        print(f"\n  Foreign-gold subset ({len(foreign_idx)} lines): "
              f"{f_in_gold}/{len(foreign_idx)} = {100*f_in_gold/len(foreign_idx):.1f}%")

    print(f"\n[{elapsed:.1f}s] Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
