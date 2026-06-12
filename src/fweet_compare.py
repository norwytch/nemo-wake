#!/usr/bin/env python3
"""Compare model's per-segmenter portmanteau readings to FWEET scholarly readings.

For each portmanteau probed in a morpheme_phon experiment, look up FWEET's
high-confidence source-form annotations (from joyce-pos-hypotheses). Determine
which segmenter's morpheme count best matches the scholarly source-form
structure, and report each segmenter's N/V reading alongside.

Scholarly count heuristic: number of whitespace-separated words in the FWEET
source_form. So "pas encore" -> 2; "Erinnerung" -> 1; "riverranno" -> 1.

Match: |segmenter_morph_count - scholarly_count| <= 1 (allow some slack for
strip-punct artifacts in segmenter morphs).

Outputs a per-portmanteau table and an aggregate concordance rate per
segmenter.

Usage:
    python src/fweet_compare.py \\
        --gallery-results results/morpheme_phon_NNNN/results.json \\
        --pos-artifact data/pos/joyce-pos-hypotheses/book01_ep01.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


SEGMENTERS = ["bpe-4000", "unigram-4000", "morfessor", "flatcat"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["count", "activation"], default="count",
                   help="count: morpheme-count vs FWEET (legacy). activation: "
                        "three-lens — NEMO activation map vs each segmenter's "
                        "decomposition, cross-tabbed with FWEET count concordance.")
    # count mode
    p.add_argument("--gallery-results", type=Path, default=None,
                   help="[count mode] morpheme_phon results.json")
    p.add_argument("--pos-artifact", type=Path, default=None,
                   help="[count mode] joyce-pos-hypotheses/book{NN}_ep{NN}.jsonl")
    # activation mode
    p.add_argument("--results", type=Path, default=None,
                   help="[activation mode] replicate_ngram_phon results.json (has activation_profile)")
    p.add_argument("--lexicon", type=Path, default=None,
                   help="[activation mode] phoneme gallery (has constituents_by_segmenter + fweet_source_forms)")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional output JSON path")
    return p.parse_args()


def count_scholarly_morphemes(source_form: str) -> int:
    """Heuristic: number of whitespace-separated words in the source form.
    Stripping parentheticals first."""
    # Strip anything in parens (motif markers etc.)
    cleaned = []
    depth = 0
    for c in source_form:
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            cleaned.append(c)
    return len("".join(cleaned).split())


def run_activation_mode(args):
    """Three-lens comparison: NEMO activation map vs each segmenter's decomposition,
    cross-tabbed against FWEET source-form count concordance."""
    import sys

    from activation_alignment import score_probe

    if not args.results or not args.lexicon:
        sys.exit("activation mode needs --results and --lexicon")
    results = json.loads(args.results.read_text())
    lexicon = json.loads(args.lexicon.read_text())
    probe_meta = {p["text"]: p for p in lexicon.get("probe_words", [])}

    rows = []
    seg_sep = defaultdict(list)         # segmenter -> [separation per probe]
    seg_pk = defaultdict(list)          # segmenter -> [precision@k per probe]
    fweet_concord = {s: {"matches": 0, "total": 0} for s in SEGMENTERS}
    agree = {"agree": 0, "total": 0}    # NEMO-favoured segmenter == FWEET-favoured?

    for probe in results.get("probe_q", []):
        text = probe["text"]
        profile = probe.get("activation_profile")
        meta = probe_meta.get(text)
        if not profile or not meta:
            continue
        cbs = meta.get("constituents_by_segmenter", {})
        fweet = meta.get("fweet_source_forms", [])
        scholarly = [count_scholarly_morphemes(f["source_form"]) for f in fweet if f.get("source_form")]
        modal = max(set(scholarly), key=scholarly.count) if scholarly else None

        per_seg = {}
        for seg, constituents in cbs.items():
            constituents = [c for c in constituents if c]
            if not constituents:
                continue
            s = score_probe(profile, set(constituents))
            if s is None:
                continue
            per_seg[seg] = {"separation": s["separation"], "precision_at_k": s["precision_at_k"],
                            "reciprocal_rank": s["reciprocal_rank"], "n_constituents": len(constituents)}
            seg_sep[seg].append(s["separation"])
            seg_pk[seg].append(s["precision_at_k"])
            if modal is not None:
                fweet_concord[seg]["total"] += 1
                if abs(len(constituents) - modal) <= 1:
                    fweet_concord[seg]["matches"] += 1

        nemo_best = max(per_seg, key=lambda s: per_seg[s]["separation"]) if per_seg else None
        fweet_best = [s for s in per_seg if modal is not None and abs(per_seg[s]["n_constituents"] - modal) <= 1]
        if nemo_best and fweet_best:
            agree["total"] += 1
            if nemo_best in fweet_best:
                agree["agree"] += 1
        rows.append({"portmanteau": text, "page_line": meta.get("page_line"),
                     "fweet_source_forms": [f["source_form"] for f in fweet],
                     "scholarly_modal_count": modal, "per_segmenter": per_seg,
                     "nemo_favoured": nemo_best, "fweet_favoured": fweet_best})

    # ---- print ----
    print(f"=== Three-lens activation alignment ({results.get('experiment')}, "
          f"ngram={results.get('ngram')}) ===")
    print(f"  scored {len(rows)} portmanteaus\n")
    print(f"  {'segmenter':14s} {'mean sep':>9} {'mean P@k':>9} {'FWEET concord':>14}")
    for seg in SEGMENTERS:
        if not seg_sep[seg]:
            continue
        ms = sum(seg_sep[seg]) / len(seg_sep[seg])
        mp = sum(seg_pk[seg]) / len(seg_pk[seg])
        c = fweet_concord[seg]
        cr = f"{c['matches']}/{c['total']}" if c["total"] else "n/a"
        print(f"  {seg:14s} {ms:>9.3f} {mp:>9.3f} {cr:>14}")
    if agree["total"]:
        print(f"\n  NEMO-favoured segmenter agrees with FWEET-favoured: "
              f"{agree['agree']}/{agree['total']} = {100*agree['agree']/agree['total']:.0f}%")

    out_path = args.output or args.results.parent / "three_lens_compare.json"
    out_path.write_text(json.dumps({
        "experiment": results.get("experiment"), "ngram": results.get("ngram"),
        "n_scored": len(rows),
        "mean_separation": {s: (sum(v) / len(v) if v else None) for s, v in seg_sep.items()},
        "mean_precision_at_k": {s: (sum(v) / len(v) if v else None) for s, v in seg_pk.items()},
        "fweet_concordance": fweet_concord,
        "nemo_fweet_agreement": agree,
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


def main():
    args = parse_args()
    if args.mode == "activation":
        return run_activation_mode(args)
    if not args.gallery_results or not args.pos_artifact:
        import sys
        sys.exit("count mode needs --gallery-results and --pos-artifact")
    with args.gallery_results.open() as f:
        results = json.load(f)
    probes = results.get("probe_q", [])
    # The morpheme_phon script's config strips down probe entries to (text, morphemes).
    # The portmanteau/segmenter metadata lives in the original lexicon, which the config
    # references via lexicon_path. Resolve that path and read the lexicon for full metadata.
    config_path = args.gallery_results.parent / "config.json"
    with config_path.open() as f:
        config = json.load(f)
    lexicon_path = Path(config.get("lexicon_path", ""))
    if not lexicon_path.is_absolute():
        # The path is recorded as it was passed on the CLI (likely relative to project dir).
        # Try resolving relative to the project root.
        project_root = args.gallery_results.parent.parent.parent
        lexicon_path = project_root / lexicon_path
    with lexicon_path.open() as f:
        lexicon = json.load(f)
    probe_meta = {p["text"]: p for p in lexicon.get("probe_words", [])}

    # Read FWEET source-form hypotheses per orth from the POS artifact
    fweet_by_orth = defaultdict(list)  # orth -> list of (source_form, language, pos_tags)
    with args.pos_artifact.open() as f:
        for line in f:
            rec = json.loads(line)
            orth = rec["orth"].strip(".,;:!?\"()")
            for h in rec["hypotheses"]:
                if h["method"] == "fweet_source_form":
                    fweet_by_orth[orth].append({
                        "source_form": h.get("input"),
                        "language": h.get("language"),
                        "pos": h.get("pos", []),
                    })

    # Group probe results by portmanteau
    by_portm = defaultdict(dict)  # portm_orth -> {segmenter: probe_result_with_meta}
    for probe in probes:
        meta = probe_meta.get(probe["text"])
        if meta is None:
            continue
        portm = meta.get("portmanteau")
        if portm is None:
            # Old single-portmanteau lexicons don't have this field. Skip.
            continue
        seg = meta["segmenter"]
        by_portm[portm][seg] = {
            **probe,
            "morphemes": meta["morphemes"],
            "morpheme_count": len(meta["morphemes"]),
            "page_line": meta.get("page_line"),
        }

    # Build comparison output
    portm_rows = []
    concordance = {seg: {"matches": 0, "total": 0} for seg in
                    ["bpe-4000", "unigram-4000", "morfessor", "flatcat"]}
    for portm in sorted(by_portm.keys()):
        entries = by_portm[portm]
        fweet_entries = fweet_by_orth.get(portm, [])
        # Scholarly counts per source-form
        scholarly_counts = [count_scholarly_morphemes(fe["source_form"])
                            for fe in fweet_entries]
        row = {
            "portmanteau": portm,
            "page_line": next(iter(entries.values()))["page_line"] if entries else None,
            "fweet_sources": fweet_entries,
            "scholarly_morpheme_counts": scholarly_counts,
            "scholarly_count_modal": (
                max(set(scholarly_counts), key=scholarly_counts.count) if scholarly_counts else None
            ),
            "by_segmenter": {},
        }
        for seg in ["bpe-4000", "unigram-4000", "morfessor", "flatcat"]:
            if seg not in entries:
                continue
            e = entries[seg]
            seg_count = e["morpheme_count"]
            matches_scholarly = (
                row["scholarly_count_modal"] is not None
                and abs(seg_count - row["scholarly_count_modal"]) <= 1
            )
            row["by_segmenter"][seg] = {
                "morphemes": e["morphemes"],
                "morpheme_count": seg_count,
                "noun_input": e["noun_input"],
                "verb_input": e["verb_input"],
                "ratio_noun_to_verb": e["ratio_noun_to_verb"],
                "matches_scholarly_count": matches_scholarly,
            }
            if row["scholarly_count_modal"] is not None:
                concordance[seg]["total"] += 1
                if matches_scholarly:
                    concordance[seg]["matches"] += 1
        portm_rows.append(row)

    # Print readable summary
    print(f"=== Portmanteau / FWEET comparison ===")
    print(f"  Total portmanteaus in gallery: {len(portm_rows)}")
    n_with_fweet = sum(1 for r in portm_rows if r["fweet_sources"])
    print(f"  With FWEET annotation: {n_with_fweet}")
    print()
    print(f"=== Per-portmanteau readings ===\n")
    for r in portm_rows:
        if not r["fweet_sources"]:
            continue  # skip those without scholarly comparison
        print(f"{r['portmanteau']} ({r['page_line']})")
        for fe in r["fweet_sources"]:
            print(f"  FWEET {fe['language']:>6}: '{fe['source_form']}' "
                  f"({count_scholarly_morphemes(fe['source_form'])} words)")
        print(f"  scholarly modal count: {r['scholarly_count_modal']}")
        for seg in ["bpe-4000", "unigram-4000", "morfessor", "flatcat"]:
            seg_info = r["by_segmenter"].get(seg)
            if not seg_info:
                continue
            match_str = "✓" if seg_info["matches_scholarly_count"] else "✗"
            print(f"  {seg:>14s} [{match_str}]: {seg_info['morpheme_count']} morphs "
                  f"{seg_info['morphemes']!s:40s}  N/V={seg_info['ratio_noun_to_verb']:>7.2f}")
        print()

    print(f"=== Concordance rate (segmenter morph count matches scholarly within ±1) ===")
    for seg, c in concordance.items():
        if c["total"] == 0:
            continue
        rate = c["matches"] / c["total"] * 100
        print(f"  {seg:>14s}: {c['matches']:>3}/{c['total']:<3} = {rate:>5.1f}%")

    out_path = args.output or args.gallery_results.parent / "fweet_compare.json"
    with out_path.open("w") as f:
        json.dump({
            "total_portmanteaus": len(portm_rows),
            "with_fweet": n_with_fweet,
            "concordance": concordance,
            "rows": portm_rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
