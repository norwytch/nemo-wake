#!/usr/bin/env python3
"""Build a phoneme-schema portmanteau gallery for replicate_ngram_phon.py (Design B).

Trained sub-units are the unique segmenter morphemes (phonemized in English IPA
via espeak en-us, bucketed by their POS hypothesis). Probes are the portmanteaus
themselves (phonemized whole). Each probe carries, per segmenter, the ordered
morpheme decomposition (its buckable constituents) plus FWEET source-forms — so
downstream scoring can ask: which segmenter's decomposition does NEMO's
partial-activation map best recover, and does that agree with FWEET?

Source: data/pos/joyce-pos-hypotheses/book{NN}_ep{NN}.jsonl (multi-method POS).

Portmanteau filters (same spirit as build_portmanteau_gallery_lexicon.py):
  - multi-morph in ≥1 segmenter
  - ≥2 distinct decompositions across the 4 segmenters (disagreement)
  - ≥1 N-like and ≥1 V-like morpheme across all segmenters (bilateral potential)

Phonemization note: morphemes are phonemized in isolation, so concatenated
morpheme phonemes won't exactly equal the whole-word phonemes (espeak applies
word-level rules). That's the realistic in-isolation-vs-in-compound scenario;
n-gram overlap is robust to the minor edge effects.

Usage:
    python scripts/preprocessing/text/build_phoneme_gallery.py \\
        --book 1 --episode 1 \\
        --output projects/01-biological-language-organ/lexicons/ep1_phoneme_gallery.json
"""
import argparse
import json
import string
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
POS_DIR = REPO_ROOT / "data" / "pos" / "joyce-pos-hypotheses"

NOUN_LIKE = {"NOUN", "PROPN"}
VERB_LIKE = {"VERB", "AUX"}
SEGMENTERS = ("bpe-4000", "unigram-4000", "morfessor", "flatcat")
PUNCT = string.punctuation + "—–“”‘’«»…"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", type=int, default=1)
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--page-min", type=int, default=None)
    p.add_argument("--page-max", type=int, default=None)
    p.add_argument("--require-fweet", action="store_true", default=True,
                   help="Keep only portmanteaus with ≥1 FWEET source-form (the "
                        "ones the three-lens comparison can actually score). On by default.")
    p.add_argument("--all-portmanteaus", dest="require_fweet", action="store_false",
                   help="Disable the FWEET requirement (keeps every filter-passing token).")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def bucket_for(pos_list):
    for p in pos_list:
        if p in NOUN_LIKE:
            return "noun"
    for p in pos_list:
        if p in VERB_LIKE:
            return "verb"
    return None


def page_of(page_line: str) -> int:
    try:
        return int(page_line.split(".")[0])
    except (ValueError, IndexError):
        return -1


def phonemize_batch(strings, language="en-us"):
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    sep = Separator(phone=" ", syllable="", word="|")
    backend = EspeakBackend(language, preserve_punctuation=False, with_stress=False,
                            language_switch="remove-flags")
    clean = [s.strip(PUNCT) or s for s in strings]
    out = backend.phonemize(clean, separator=sep, strip=True)
    return {s: ipa.split() for s, ipa in zip(strings, out)}


def main():
    args = parse_args()
    pos_path = POS_DIR / f"book{args.book:02d}_ep{args.episode:02d}.jsonl"
    if not pos_path.exists():
        sys.exit(f"Missing {pos_path}. Run pos_tag.py first.")

    # ---- collect candidate portmanteaus ----
    portmanteaus = []  # {orth, page_line, decomp{seg:[morphs]}, fweet:[{lang,form}]}
    with pos_path.open() as f:
        for line in f:
            rec = json.loads(line)
            pg = page_of(rec["page_line"])
            if args.page_min is not None and pg < args.page_min:
                continue
            if args.page_max is not None and pg > args.page_max:
                continue
            decomp = defaultdict(list)
            morph_pos = {}
            fweet = []
            for h in rec["hypotheses"]:
                if h["method"] == "morpheme":
                    decomp[h["segmenter"]].append(h["input"])
                    morph_pos[h["input"]] = h.get("pos", [])
                elif h["method"] == "fweet_source_form":
                    fweet.append({"language": h.get("language"), "source_form": h.get("input")})
            if not decomp:
                continue
            # filters
            multi = any(len(v) >= 2 for v in decomp.values())
            distinct = len({tuple(v) for v in decomp.values()})
            all_morphs = {m for v in decomp.values() for m in v}
            has_noun = any(bucket_for(morph_pos.get(m, [])) == "noun" for m in all_morphs)
            has_verb = any(bucket_for(morph_pos.get(m, [])) == "verb" for m in all_morphs)
            if not (multi and distinct >= 2 and has_noun and has_verb):
                continue
            if args.require_fweet and not fweet:
                continue
            portmanteaus.append({
                "orth": rec["orth"].strip(PUNCT) or rec["orth"],
                "page_line": rec["page_line"],
                "decomp": {s: decomp[s] for s in decomp},
                "morph_pos": morph_pos,
                "fweet": fweet,
            })

    if not portmanteaus:
        sys.exit("No portmanteaus passed the filters in this range.")

    # ---- collect unique trainable morphemes (N/V-bucketed) ----
    morph_bucket = {}
    for pm in portmanteaus:
        for m, pos in pm["morph_pos"].items():
            b = bucket_for(pos)
            if b and m.strip(PUNCT):
                morph_bucket.setdefault(m, b)

    # ---- phonemize all morphemes + portmanteaus in one espeak pass each ----
    morph_list = sorted(morph_bucket)
    portm_list = sorted({pm["orth"] for pm in portmanteaus})
    print(f"Phonemizing {len(morph_list)} morphemes + {len(portm_list)} portmanteaus (espeak en-us)...")
    morph_ipa = phonemize_batch(morph_list)
    portm_ipa = phonemize_batch(portm_list)

    # ---- assemble lexicon ----
    training_words = []
    for m in morph_list:
        ph = morph_ipa.get(m, [])
        if not ph:
            continue
        training_words.append({"text": m, "bucket": morph_bucket[m], "phonemes": ph})

    trained_set = {w["text"] for w in training_words}
    probe_words = []
    seen = set()
    for pm in portmanteaus:
        if pm["orth"] in seen:
            continue
        seen.add(pm["orth"])
        ph = portm_ipa.get(pm["orth"], [])
        if not ph:
            continue
        # per-segmenter constituents, restricted to morphemes that became training words
        cbs = {}
        for seg in SEGMENTERS:
            morphs = pm["decomp"].get(seg)
            if not morphs:
                continue
            cbs[seg] = [m.strip(PUNCT) for m in morphs if m.strip(PUNCT) in trained_set]
        probe_words.append({
            "text": pm["orth"],
            "page_line": pm["page_line"],
            "phonemes": ph,
            "constituents_by_segmenter": cbs,
            "fweet_source_forms": pm["fweet"],
        })

    n_noun = sum(1 for w in training_words if w["bucket"] == "noun")
    n_verb = sum(1 for w in training_words if w["bucket"] == "verb")
    lexicon = {
        "name": f"phoneme_gallery_book{args.book:02d}_ep{args.episode:02d}",
        "description": "Design-B phoneme gallery: segmenter morphemes (en-us IPA, "
                       "POS-bucketed) as trained sub-units; portmanteaus as probes.",
        "source_artifact": str(pos_path.relative_to(REPO_ROOT)),
        "ipa_variant": "en-us-baseline (espeak, isolated)",
        "balance_audit": {"training_nouns": n_noun, "training_verbs": n_verb,
                          "num_portmanteaus": len(probe_words)},
        "training_words": training_words,
        "probe_words": probe_words,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2))
    print(f"  training_words: {len(training_words)} ({n_noun} noun / {n_verb} verb)")
    print(f"  probe portmanteaus: {len(probe_words)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
