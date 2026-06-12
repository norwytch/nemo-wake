#!/usr/bin/env python3
"""Score a NEMO partial-activation map against a reference decomposition.

Reference-agnostic. Given per-probe activation profiles (from
replicate_ngram_phon.py / replicate_phoneme_phon.py results.json) and, per
probe, a reference set of "true" constituent sub-words, measure how well the
activation map recovers the decomposition:

  - precision@k        (k = number of reference constituents; top-k of the
                        ranked profile that are true constituents)
  - reciprocal rank    (1 / rank of the first true constituent; aggregated → MRR)
  - separation         (mean overlap_frac of constituents − mean of distractors)

Each comes with a random baseline (expected value under a random ranking of the
trained words), so a single absolute number isn't mistaken for signal.

Reference source (in priority order):
  1. --reference / --reference-file JSON: {probe_text: [sub_word, ...]}
  2. a 'constituents' field on each probe entry in the results.json

Usage:
    python src/activation_alignment.py \\
        --results results/ngram2_phon_NNNN/results.json \\
        --reference '{"riverrun": ["river", "run"]}'
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path


def expected_reciprocal_rank(k: int, n: int) -> float:
    """E[1/rank of first relevant] for k relevant items among n, uniformly shuffled."""
    if k <= 0 or n <= 0 or k > n:
        return 0.0
    total = comb(n, k)
    e = 0.0
    for r in range(1, n - k + 2):
        e += (1.0 / r) * comb(n - r, k - 1) / total
    return e


def score_probe(profile: list[dict], constituents: set[str]) -> dict | None:
    """profile: [{trained_word, overlap, overlap_frac}, ...] sorted desc by overlap."""
    ranked = [p["trained_word"] for p in profile]
    n = len(ranked)
    k = len(constituents)
    if k == 0 or n == 0:
        return None
    ranks = [(ranked.index(c) + 1) if c in ranked else (n + 1) for c in constituents]
    first_rank = min(ranks)
    topk = set(ranked[:k])
    hits = len(topk & constituents)
    frac = {p["trained_word"]: p["overlap_frac"] for p in profile}
    con = [frac.get(c, 0.0) for c in constituents]
    dis = [frac[w] for w in ranked if w not in constituents]
    mean_con = sum(con) / len(con) if con else 0.0
    mean_dis = sum(dis) / len(dis) if dis else 0.0
    return {
        "k": k,
        "n_trained": n,
        "constituent_ranks": sorted(ranks),
        "reciprocal_rank": 1.0 / first_rank,
        "precision_at_k": hits / k,
        "separation": mean_con - mean_dis,
        "mean_constituent_frac": round(mean_con, 4),
        "mean_distractor_frac": round(mean_dis, 4),
        "random_precision_at_k": k / n,
        "random_reciprocal_rank": expected_reciprocal_rank(k, n),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True,
                   help="results.json from replicate_ngram_phon.py / replicate_phoneme_phon.py")
    p.add_argument("--reference", type=str, default=None,
                   help='Inline JSON: {"probe_text": ["sub_word", ...]}')
    p.add_argument("--reference-file", type=Path, default=None,
                   help="JSON file with the same shape as --reference")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with args.results.open() as f:
        results = json.load(f)
    probes = results.get("probe_q", [])

    reference = {}
    if args.reference_file:
        reference.update(json.loads(args.reference_file.read_text()))
    if args.reference:
        reference.update(json.loads(args.reference))

    rows = []
    agg = {"precision_at_k": [], "reciprocal_rank": [], "separation": [],
           "random_precision_at_k": [], "random_reciprocal_rank": []}
    for probe in probes:
        text = probe["text"]
        profile = probe.get("activation_profile")
        if not profile:
            continue
        constituents = set(reference.get(text) or probe.get("constituents") or [])
        if not constituents:
            continue
        s = score_probe(profile, constituents)
        if s is None:
            continue
        s["text"] = text
        s["constituents"] = sorted(constituents)
        rows.append(s)
        for m in agg:
            agg[m].append(s[m])

    if not rows:
        sys.exit("No probes with both an activation_profile and a reference. "
                 "Pass --reference or add 'constituents' to probe entries.")

    print(f"=== Activation-map alignment ({results.get('experiment')}, "
          f"ngram={results.get('ngram')}) ===")
    print(f"  scored {len(rows)} probe(s)\n")
    print(f"  {'probe':14s} {'k':>2} {'ranks':>10} {'P@k':>6} {'rand':>6} "
          f"{'RR':>6} {'rand':>6} {'sep':>7}")
    for r in rows:
        print(f"  {r['text']:14s} {r['k']:>2} {str(r['constituent_ranks']):>10} "
              f"{r['precision_at_k']:>6.2f} {r['random_precision_at_k']:>6.2f} "
              f"{r['reciprocal_rank']:>6.2f} {r['random_reciprocal_rank']:>6.2f} "
              f"{r['separation']:>7.3f}")

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n  === aggregate (n={len(rows)}) ===")
    print(f"  precision@k:    {mean(agg['precision_at_k']):.3f}  "
          f"(random {mean(agg['random_precision_at_k']):.3f})")
    print(f"  MRR:            {mean(agg['reciprocal_rank']):.3f}  "
          f"(random {mean(agg['random_reciprocal_rank']):.3f})")
    print(f"  mean separation:{mean(agg['separation']):.3f}  (random ~0.0)")

    out = args.output or args.results.parent / "activation_alignment.json"
    with out.open("w") as f:
        json.dump({
            "experiment": results.get("experiment"),
            "ngram": results.get("ngram"),
            "n_scored": len(rows),
            "aggregate": {m: mean(v) for m, v in agg.items()},
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
