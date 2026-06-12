#!/usr/bin/env python3
"""Morpheme-bag PHON. Word PHON activation = union of morpheme blocks.

PHON is restructured: one explicit block per unique morpheme in the lexicon.
A word's PHON activation is the *union* of the blocks for its morphemes (no
position information; multisets collapsed to sets).

This is the morpheme-level analog of replicate_phoneme_phon.py. Morpheme is
the more linguistically meaningful unit for the Wake question: portmanteaus
like `riverrun` decompose into morphemes that carry POS associations
(`river` is a noun-meaningful morpheme, `run` a verb-meaningful one),
whereas phonemes don't carry meaning individually.

Lexicon schema:
  training_words: [{"text": str, "bucket": "noun"|"verb", "morphemes": [str, ...]}]
  probe_words:    [{"text": str, "morphemes": [str, ...]}]

Morpheme tokens come from an unsupervised segmenter trained on the Wake
(BPE-4k, Unigram-4k, Morfessor, or FlatCat — see scripts/preprocessing/README.md).
This script doesn't run the segmenter; it consumes the morpheme lists
written into the lexicon JSON, with the segmenter named in the JSON metadata.
"""
import argparse
import json
import sys
import time
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
VISUAL = "VISUAL"
MOTOR = "MOTOR"
NOUN = "NOUN"
VERB = "VERB"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lexicon", type=Path, required=True,
                   help="Lexicon JSON with 'training_words' and 'probe_words' arrays. "
                        "Each entry needs 'text', 'morphemes' (list), and "
                        "'bucket' (training_words only).")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--n", type=int, default=10000,
                   help="LEX_n for NOUN/VERB sparse hubs.")
    p.add_argument("--lex-k", type=int, default=100)
    p.add_argument("--phon-k", type=int, default=100,
                   help="Neurons per morpheme block. Default 100; morphemes per "
                        "word are typically 1–3 so total active PHON ≈ ctx_k.")
    p.add_argument("--ctx-k", type=int, default=100)
    p.add_argument("--p", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.06)
    p.add_argument("--proj-rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_lexicon(path: Path):
    with path.open() as f:
        data = json.load(f)
    train = data.get("training_words", [])
    probe = data.get("probe_words", [])
    if not train:
        sys.exit(f"Lexicon {path} must contain 'training_words'.")
    morphemes = set()
    for w in train + probe:
        if "morphemes" not in w:
            sys.exit(f"Word {w.get('text', '?')} missing 'morphemes' list.")
        morphemes.update(w["morphemes"])
    morpheme_to_idx = {m: i for i, m in enumerate(sorted(morphemes))}
    visual_counter = 0
    motor_counter = 0
    for w in train:
        if w["bucket"] == "noun":
            w["visual_idx"] = visual_counter; visual_counter += 1
            w["motor_idx"] = None
        elif w["bucket"] == "verb":
            w["motor_idx"] = motor_counter; motor_counter += 1
            w["visual_idx"] = None
        else:
            sys.exit(f"Word {w['text']}: unknown bucket {w['bucket']!r}")
    for w in probe:
        w["visual_idx"] = None
        w["motor_idx"] = None
    meta = {k: v for k, v in data.items() if k not in ("training_words", "probe_words")}
    return train, probe, morpheme_to_idx, visual_counter, motor_counter, meta


def build_brain(morpheme_to_idx, num_visual, num_motor, args):
    import brain
    b = brain.Brain(args.p)
    b.add_explicit_area(PHON, len(morpheme_to_idx) * args.phon_k, args.phon_k, args.beta)
    b.add_explicit_area(VISUAL, max(num_visual, 1) * args.ctx_k, args.ctx_k, args.beta)
    b.add_explicit_area(MOTOR, max(num_motor, 1) * args.ctx_k, args.ctx_k, args.beta)
    b.add_area(NOUN, args.n, args.lex_k, args.beta)
    b.add_area(VERB, args.n, args.lex_k, args.beta)
    return b


def activate_morpheme_union(b, morphemes, morpheme_to_idx, phon_k):
    winners = []
    for m in set(morphemes):
        idx = morpheme_to_idx[m]
        winners.extend(range(idx * phon_k, (idx + 1) * phon_k))
    b.area_by_name[PHON].winners = sorted(winners)
    b.area_by_name[PHON].fix_assembly()


def activate_block(b, area_name, block_idx):
    area = b.area_by_name[area_name]
    k = area.k
    start = block_idx * k
    area.winners = list(range(start, start + k))
    area.fix_assembly()


def clear_all(b):
    for area in (PHON, VISUAL, MOTOR):
        b.area_by_name[area].unfix_assembly()
        b.area_by_name[area].winners = []
    b.area_by_name[NOUN].winners = []
    b.area_by_name[VERB].winners = []


def train_word(b, word, morpheme_to_idx, args):
    activate_morpheme_union(b, word["morphemes"], morpheme_to_idx, args.phon_k)
    is_noun = word["bucket"] == "noun"
    if is_noun:
        activate_block(b, VISUAL, word["visual_idx"])
        first_proj = {PHON: [NOUN, VERB], VISUAL: [NOUN]}
        recurrent = {
            PHON: [NOUN], VISUAL: [NOUN],
            NOUN: [PHON, NOUN, VISUAL],
        }
    else:
        activate_block(b, MOTOR, word["motor_idx"])
        first_proj = {PHON: [NOUN, VERB], MOTOR: [VERB]}
        recurrent = {
            PHON: [NOUN, VERB], MOTOR: [VERB],
            VERB: [PHON, VERB, MOTOR],
        }
        # NOTE: above keeps PHON projecting to both, which mirrors mutual-inhibition's
        # "drop the losing hub" pattern. For verb training, the losing hub is NOUN.
        recurrent = {
            PHON: [VERB], MOTOR: [VERB],
            VERB: [PHON, VERB, MOTOR],
        }
    b.project({}, first_proj)
    for _ in range(args.proj_rounds):
        b.project({}, recurrent)
    clear_all(b)


def phon_inputs(b, morphemes, morpheme_to_idx, phon_k):
    clear_all(b)
    activate_morpheme_union(b, morphemes, morpheme_to_idx, phon_k)
    b.project({}, {PHON: [NOUN, VERB]})
    pw = b.area_by_name[PHON].winners
    nw = b.area_by_name[NOUN].winners
    vw = b.area_by_name[VERB].winners
    noun_in = float(b.connectomes[PHON][NOUN][pw][:, nw].sum()) if pw and nw else 0.0
    verb_in = float(b.connectomes[PHON][VERB][pw][:, vw].sum()) if pw and vw else 0.0
    return noun_in, verb_in


def main():
    args = parse_args()
    ensure_hash_seed(args.seed)
    run_dir = make_run_dir("morpheme_phon")
    open_tee_log(run_dir)
    add_nemo_to_path()
    setup_seeded(args.seed)

    train, probe, morpheme_to_idx, num_visual, num_motor, lex_meta = load_lexicon(args.lexicon)

    save_json(run_dir, "config.json", {
        "experiment": "morpheme_phon",
        "lexicon_path": str(args.lexicon),
        "lexicon_meta": lex_meta,
        "morpheme_to_idx": morpheme_to_idx,
        "num_morphemes": len(morpheme_to_idx),
        "num_training_words": len(train),
        "num_probe_words": len(probe),
        "training_words": [{"text": w["text"], "bucket": w["bucket"],
                            "morphemes": w["morphemes"]} for w in train],
        "probe_words": [{"text": w["text"], "morphemes": w["morphemes"]} for w in probe],
        "rounds": args.rounds, "n": args.n, "lex_k": args.lex_k,
        "phon_k": args.phon_k, "ctx_k": args.ctx_k, "p": args.p,
        "beta": args.beta, "proj_rounds": args.proj_rounds, "seed": args.seed,
    })

    t0 = time.time()
    print(
        f"[{time.time()-t0:6.1f}s] Building brain "
        f"(num_morphemes={len(morpheme_to_idx)}, PHON_n={len(morpheme_to_idx) * args.phon_k}, "
        f"VISUAL_slots={num_visual}, MOTOR_slots={num_motor})",
        flush=True,
    )
    print(f"  Morpheme inventory: {sorted(morpheme_to_idx.keys())}", flush=True)
    b = build_brain(morpheme_to_idx, num_visual, num_motor, args)

    print(
        f"[{time.time()-t0:6.1f}s] Training {args.rounds} rounds × "
        f"{len(train)} words = {args.rounds * len(train)} tutoring steps",
        flush=True,
    )
    for r in range(args.rounds):
        for w in train:
            train_word(b, w, morpheme_to_idx, args)
        if (r + 1) % 10 == 0 or r == args.rounds - 1:
            print(f"  [{time.time()-t0:6.1f}s] round {r+1}/{args.rounds}", flush=True)

    b.disable_plasticity = True

    print(f"\n[{time.time()-t0:6.1f}s] === Trained words: Q ratios ===", flush=True)
    train_results = []
    for w in train:
        n_in, v_in = phon_inputs(b, w["morphemes"], morpheme_to_idx, args.phon_k)
        ratio = n_in / v_in if v_in > 0 else float("inf")
        train_results.append({
            "text": w["text"], "bucket": w["bucket"], "morphemes": w["morphemes"],
            "noun_input": n_in, "verb_input": v_in,
            "ratio_noun_to_verb": ratio,
        })
        morph_str = " + ".join(w["morphemes"])
        print(
            f"  {w['text']:10s} ({w['bucket']:4s})  [{morph_str:20s}]  "
            f"NOUN_in={n_in:>9.1f}  VERB_in={v_in:>9.1f}  N/V={ratio:>7.2f}",
            flush=True,
        )

    print(f"\n[{time.time()-t0:6.1f}s] === Held-out probe words: Q ratios ===", flush=True)
    probe_results = []
    for w in probe:
        n_in, v_in = phon_inputs(b, w["morphemes"], morpheme_to_idx, args.phon_k)
        ratio = n_in / v_in if v_in > 0 else float("inf")
        probe_results.append({
            "text": w["text"], "morphemes": w["morphemes"],
            "expected_composition": w.get("expected_composition"),
            "noun_input": n_in, "verb_input": v_in,
            "ratio_noun_to_verb": ratio,
        })
        morph_str = " + ".join(w["morphemes"])
        print(
            f"  {w['text']:10s} (HELDOUT) [{morph_str:20s}]  "
            f"NOUN_in={n_in:>9.1f}  VERB_in={v_in:>9.1f}  N/V={ratio:>7.2f}",
            flush=True,
        )

    print(f"\n[{time.time()-t0:6.1f}s] === Per-morpheme attribution ===", flush=True)
    morpheme_results = []
    for m in sorted(morpheme_to_idx.keys()):
        n_in, v_in = phon_inputs(b, [m], morpheme_to_idx, args.phon_k)
        ratio = n_in / v_in if v_in > 0 else float("inf")
        appears_in_n = [w["text"] for w in train if w["bucket"] == "noun" and m in w["morphemes"]]
        appears_in_v = [w["text"] for w in train if w["bucket"] == "verb" and m in w["morphemes"]]
        morpheme_results.append({
            "morpheme": m,
            "noun_input": n_in, "verb_input": v_in,
            "ratio_noun_to_verb": ratio,
            "trained_in_nouns": appears_in_n, "trained_in_verbs": appears_in_v,
        })
        provenance = f"N:{appears_in_n} / V:{appears_in_v}" if (appears_in_n or appears_in_v) else "untrained"
        print(
            f"  {m:12s}  NOUN_in={n_in:>9.1f}  VERB_in={v_in:>9.1f}  N/V={ratio:>7.2f}  "
            f"{provenance}",
            flush=True,
        )
    b.disable_plasticity = False

    elapsed = time.time() - t0
    save_json(run_dir, "results.json", {
        "experiment": "morpheme_phon",
        "elapsed_seconds": elapsed,
        "trained_q": train_results,
        "probe_q": probe_results,
        "per_morpheme_q": morpheme_results,
    })
    print(f"\n[{elapsed:.1f}s] Results written to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
