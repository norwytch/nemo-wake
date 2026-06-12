#!/usr/bin/env python3
"""Build a morpheme_phon-compatible gallery lexicon from joyce-pos-hypotheses.

For each token in the chosen chapter/page range that meets the portmanteau
filter (multi-morph in ≥1 segmenter AND segmenter disagreement on decomposition
AND at least one morpheme tagged N-like and one tagged V-like), emit:
  - one training_word entry per unique morpheme across all segmenters (with
    its tagger-assigned POS mapped to noun/verb bucket)
  - four probe_word entries per portmanteau (one per segmenter), each with
    that segmenter's morpheme decomposition

Untrained morphemes (those tagged INTJ/PUNCT/DET/ADJ/etc — no N/V mapping)
are listed in inventory metadata but get no training_word entry; they still
get a PHON block via probe inclusion.

Outputs a JSON ready to feed into src/replicate_morpheme_phon.py.

Default filters:
  - Token must appear with multi-morph decomposition in ≥1 segmenter
  - Total distinct decompositions across 4 segmenters ≥ 2 (segmenter disagreement)
  - Among all morphemes (across segmenters), at least 1 N-tagged AND 1 V-tagged
    (otherwise the portmanteau has no bilateral potential)

Usage:
    python scripts/preprocessing/text/build_portmanteau_gallery_lexicon.py \\
        --book 1 --episode 1 \\
        --output projects/01-biological-language-organ/lexicons/ep1_portmanteau_gallery.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
POS_DIR = REPO_ROOT / "data" / "pos" / "joyce-pos-hypotheses"

NOUN_LIKE = {"NOUN", "PROPN"}
VERB_LIKE = {"VERB", "AUX"}
UNTRAINED_TAGS = {"INTJ", "PUNCT", "DET", "ADJ", "ADV", "ADP", "SCONJ", "CCONJ", "NUM", "PART", "PRON", "X", "SYM", "SPACE"}
PUNCT_MORPHS = {",", ".", ";", ":", "!", "?", "'", "\""}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", type=int, default=1)
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--page-min", type=int, default=None)
    p.add_argument("--page-max", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--include-controls", action="store_true", default=True,
                   help="Include river/bay/run/goes as control words (default on).")
    p.add_argument("--require-segmenter-disagreement", action="store_true", default=True,
                   help="Require ≥2 distinct decompositions across 4 segmenters.")
    p.add_argument("--require-bilateral-potential", action="store_true", default=True,
                   help="Require ≥1 N-tagged AND ≥1 V-tagged morpheme across all segmenters.")
    p.add_argument("--require-fweet", action="store_true", default=False,
                   help="Restrict to tokens with ≥1 high-confidence FWEET source-form hypothesis. "
                        "Crucial for scholarly-ground-truth comparison: only words with FWEET attribution "
                        "get a defensible scholarly decomposition to compare against.")
    p.add_argument("--max-portmanteaus", type=int, default=None,
                   help="Cap output (default: all matching tokens).")
    return p.parse_args()


def tag_to_bucket(tag: str) -> str | None:
    """Return 'noun', 'verb', or None for untrained tags."""
    if tag in NOUN_LIKE:
        return "noun"
    if tag in VERB_LIKE:
        return "verb"
    return None


def first_tag(pos_list):
    """First tag in a possibly-multi-element pos list."""
    return pos_list[0] if pos_list else None


def clean_morpheme(m: str) -> str | None:
    """Strip trailing punctuation. Return None for empty/punct-only morphs."""
    if not m:
        return None
    stripped = m.rstrip(".,;:!?")
    if not stripped or stripped in PUNCT_MORPHS:
        return None
    return stripped


def main():
    args = parse_args()
    src = POS_DIR / f"book{args.book:02d}_ep{args.episode:02d}.jsonl"
    if not src.exists():
        raise SystemExit(f"POS file not found: {src}")

    # First pass: identify portmanteau candidates and collect their decompositions
    portmanteaus = []  # list of dicts {orth, page_line, segmenter_decomps, surface_pos}
    seen_orths = set()
    with src.open() as f:
        for line in f:
            rec = json.loads(line)
            page_line = rec["page_line"]
            page = int(page_line.split(".")[0])
            if args.page_min and page < args.page_min:
                continue
            if args.page_max and page > args.page_max:
                continue
            orth = rec["orth"].strip(".,;:!?\"()")
            if not orth or orth in seen_orths:
                continue
            # Collect morpheme hypotheses per segmenter
            by_seg = defaultdict(list)  # segmenter -> [(morph, pos_tag), ...]
            surface_pos = None
            fweet_source_forms = []  # list of {source_form, language, pos}
            for h in rec["hypotheses"]:
                if h["method"] == "surface":
                    surface_pos = first_tag(h.get("pos", []))
                if h["method"] == "morpheme":
                    morph = clean_morpheme(h.get("input", ""))
                    if morph is None:
                        continue
                    by_seg[h["segmenter"]].append((morph, first_tag(h.get("pos", []))))
                if h["method"] == "fweet_source_form":
                    fweet_source_forms.append({
                        "source_form": h.get("input"),
                        "language": h.get("language"),
                        "pos": h.get("pos", []),
                    })
            if not by_seg:
                continue
            if args.require_fweet and not fweet_source_forms:
                continue
            # Apply filters
            max_morphs = max(len(m) for m in by_seg.values())
            if max_morphs < 2:
                continue
            distinct_decomps = len(set(tuple(m for m, _ in lst) for lst in by_seg.values()))
            if args.require_segmenter_disagreement and distinct_decomps < 2:
                continue
            # Check bilateral potential
            all_tags = set()
            for lst in by_seg.values():
                for _, pos in lst:
                    if pos:
                        all_tags.add(pos)
            has_n = bool(all_tags & NOUN_LIKE)
            has_v = bool(all_tags & VERB_LIKE)
            if args.require_bilateral_potential and not (has_n and has_v):
                continue
            portmanteaus.append({
                "orth": orth,
                "page_line": page_line,
                "surface_pos": surface_pos,
                "segmenter_decomps": dict(by_seg),
                "distinct_decomps": distinct_decomps,
                "fweet_source_forms": fweet_source_forms,
            })
            seen_orths.add(orth)
            if args.max_portmanteaus and len(portmanteaus) >= args.max_portmanteaus:
                break

    print(f"Found {len(portmanteaus)} portmanteau candidates (book {args.book} ep {args.episode}, "
          f"page filter {args.page_min}-{args.page_max}).")

    if not portmanteaus:
        raise SystemExit("No portmanteaus passed filters.")

    # Second pass: collect all unique morphemes and their tagger POS (across all portmanteaus)
    # If a morpheme appears with different POS in different contexts, use the first tag we see.
    morpheme_to_pos = {}  # morph -> first_seen_tag
    morpheme_provenance = defaultdict(list)  # morph -> list of (orth, segmenter)
    for pm in portmanteaus:
        for seg, morphs in pm["segmenter_decomps"].items():
            for morph, pos in morphs:
                if morph not in morpheme_to_pos:
                    morpheme_to_pos[morph] = pos
                morpheme_provenance[morph].append((pm["orth"], seg))

    # Build training_words and untrained_morphemes lists
    training_words = []
    untrained_morphemes = []
    if args.include_controls:
        training_words.extend([
            {"text": "river",   "bucket": "noun", "morphemes": ["river"],   "note": "control"},
            {"text": "bay",     "bucket": "noun", "morphemes": ["bay"],     "note": "control"},
            {"text": "run",     "bucket": "verb", "morphemes": ["run"],     "note": "control"},
            {"text": "goes",    "bucket": "verb", "morphemes": ["goes"],    "note": "control"},
        ])
    for morph in sorted(morpheme_to_pos.keys()):
        pos = morpheme_to_pos[morph]
        # Skip morphemes that match control words (already trained)
        if morph in {"river", "bay", "run", "goes"} and args.include_controls:
            continue
        bucket = tag_to_bucket(pos)
        if bucket is None:
            untrained_morphemes.append({
                "morpheme": morph,
                "tagger_pos": pos,
                "appears_in": morpheme_provenance[morph][:3],
            })
        else:
            training_words.append({
                "text": morph,
                "bucket": bucket,
                "morphemes": [morph],
                "tagger_pos": pos,
                "appears_in": morpheme_provenance[morph][:3],
            })

    # Build probe_words: one per (portmanteau, segmenter)
    probe_words = []
    for pm in portmanteaus:
        for seg in ["bpe-4000", "unigram-4000", "morfessor", "flatcat"]:
            if seg not in pm["segmenter_decomps"]:
                continue
            morphs = [m for m, _ in pm["segmenter_decomps"][seg]]
            if not morphs:
                continue
            probe_words.append({
                "text": f"{pm['orth']}_{seg.replace('-', '')}",
                "portmanteau": pm["orth"],
                "page_line": pm["page_line"],
                "surface_pos": pm["surface_pos"],
                "segmenter": seg,
                "morphemes": morphs,
            })

    output = {
        "name": args.output.stem,
        "description": (
            f"Auto-generated portmanteau gallery from book {args.book} episode {args.episode}"
            + (f" pages {args.page_min}-{args.page_max}" if (args.page_min or args.page_max) else "")
            + f". {len(portmanteaus)} portmanteaus × ≤4 segmenters = {len(probe_words)} probes. "
              f"{len(training_words)} trained morphemes (including controls), "
              f"{len(untrained_morphemes)} untrained morphemes in inventory."
        ),
        "source_artifact": f"data/pos/joyce-pos-hypotheses/book{args.book:02d}_ep{args.episode:02d}.jsonl",
        "filters": {
            "page_min": args.page_min,
            "page_max": args.page_max,
            "require_segmenter_disagreement": args.require_segmenter_disagreement,
            "require_bilateral_potential": args.require_bilateral_potential,
        },
        "num_portmanteaus": len(portmanteaus),
        "num_probes": len(probe_words),
        "num_trained_morphemes": len(training_words),
        "num_untrained_morphemes": len(untrained_morphemes),
        "untrained_morphemes_in_inventory": untrained_morphemes,
        "training_words": training_words,
        "probe_words": probe_words,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}")
    print(f"  portmanteaus: {len(portmanteaus)}, probes: {len(probe_words)}")
    print(f"  trained morphemes (incl controls): {len(training_words)}")
    print(f"  untrained morphemes in inventory: {len(untrained_morphemes)}")


if __name__ == "__main__":
    main()
