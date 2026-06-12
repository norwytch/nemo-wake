#!/usr/bin/env python3
"""Cross-method attribution evaluator for idea 2 (Tier 1).

Loads one or more results files (NEMO multilingual_context, baseline_attribution
outputs) and produces apples-to-apples comparison metrics:

  - Top-1 accuracy: overall + foreign-gold subset (the meaningful one)
  - Per-language precision / recall / F1 (top-1 as predicted-positive)
  - Macro-F1 across languages
  - Confusion matrix (primary gold language vs predicted top-1)

A 'method' is one of the named result lists inside a file. Baseline files carry
two methods each ('frequency_results' and 'bigram_langid_results'); NEMO files
carry one ('probe_results').

Usage:
    python src/evaluate_attribution.py \\
        --inputs nemo:results/mlctx_ngram2_NNNN/results.json \\
                 baseline_capped:results/baselines/cap100.json \\
                 baseline_uncapped:results/baselines/uncapped.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more <label>:<path> entries (label decides output naming).")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def load_results(path: Path):
    data = json.loads(Path(path).read_text())
    methods: dict[str, list] = {}
    if "probe_results" in data:
        methods["nemo"] = data["probe_results"]
    if "frequency_results" in data:
        methods["frequency"] = data["frequency_results"]
    if "bigram_langid_results" in data:
        methods["bigram_langid"] = data["bigram_langid_results"]
    return methods, data.get("languages", []), data


def score_method(records, languages):
    n = len(records)
    top1_in_gold = sum(1 for r in records if r["top1"] in r["gold_languages"])
    foreign = [r for r in records if set(r["gold_languages"]) - {"en-us"}]
    f_in_gold = sum(1 for r in foreign if r["top1"] in r["gold_languages"])

    tp = Counter(); fp = Counter(); fn = Counter()
    for r in records:
        pred = r["top1"]
        gold = set(r["gold_languages"])
        for L in languages:
            in_pred = (pred == L)
            in_gold = (L in gold)
            if in_pred and in_gold: tp[L] += 1
            elif in_pred: fp[L] += 1
            elif in_gold: fn[L] += 1

    per_lang = {}
    for L in languages:
        prec = tp[L] / (tp[L] + fp[L]) if (tp[L] + fp[L]) else 0.0
        rec = tp[L] / (tp[L] + fn[L]) if (tp[L] + fn[L]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_lang[L] = {"tp": tp[L], "fp": fp[L], "fn": fn[L],
                       "precision": round(prec, 4), "recall": round(rec, 4),
                       "f1": round(f1, 4)}
    macro_f1 = sum(v["f1"] for v in per_lang.values()) / len(languages) if languages else 0

    # Confusion matrix keyed by sorted-gold primary
    conf: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        primary = sorted(r["gold_languages"])[0] if r["gold_languages"] else "?"
        conf[primary][r["top1"]] += 1

    return {
        "n": n, "n_foreign": len(foreign),
        "overall_top1_in_gold": round(top1_in_gold / n, 4) if n else 0,
        "foreign_subset_top1_in_gold": round(f_in_gold / len(foreign), 4) if foreign else 0,
        "per_language": per_lang,
        "macro_f1": round(macro_f1, 4),
        "confusion": {g: dict(c) for g, c in conf.items()},
    }


def main():
    args = parse_args()
    all_methods: dict[str, list] = {}
    languages: list[str] = []
    for spec in args.inputs:
        if ":" not in spec:
            sys.exit(f"Bad --inputs entry {spec!r}; expected label:path")
        label, path = spec.split(":", 1)
        ms, langs, _ = load_results(Path(path))
        if langs and not languages:
            languages = langs
        for method_name, records in ms.items():
            all_methods[f"{label}.{method_name}"] = records
    if not languages:
        sys.exit("No languages found in any input file.")

    summary = {m: score_method(r, languages) for m, r in all_methods.items()}
    n_probes = max((s["n"] for s in summary.values()), default=0)

    print(f"=== Attribution comparison ({len(all_methods)} methods, {n_probes} probes) ===")
    print(f"  languages: {languages}\n")
    print(f"  {'method':32s} {'overall':>9} {'foreign':>9} {'macro F1':>10}")
    for method, s in summary.items():
        print(f"  {method:32s} {100*s['overall_top1_in_gold']:>8.1f}% "
              f"{100*s['foreign_subset_top1_in_gold']:>8.1f}% {s['macro_f1']:>10.3f}")

    print(f"\n  Per-language F1:")
    print("    " + f"{'method':30s} " + " ".join(f"{L:>8s}" for L in languages))
    for method, s in summary.items():
        cells = " ".join(f"{s['per_language'][L]['f1']:>8.3f}" for L in languages)
        print(f"    {method:30s} " + cells)

    print(f"\n  Per-language recall:")
    print("    " + f"{'method':30s} " + " ".join(f"{L:>8s}" for L in languages))
    for method, s in summary.items():
        cells = " ".join(f"{s['per_language'][L]['recall']:>8.3f}" for L in languages)
        print(f"    {method:30s} " + cells)

    print(f"\n  Confusion matrices (rows = primary gold language, cols = predicted top1):")
    for method, s in summary.items():
        print(f"\n  {method}:")
        header = f"{'gold↓ / pred→':16s} " + " ".join(f"{L:>8s}" for L in languages)
        print(f"    {header}")
        for gl in languages:
            cnt_for_gl = s["confusion"].get(gl, {})
            cells = " ".join(f"{cnt_for_gl.get(L, 0):>8d}" for L in languages)
            print(f"    {gl:16s} {cells}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "languages": languages, "summary": summary,
        }, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
