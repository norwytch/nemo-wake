#!/usr/bin/env python3
"""Phoneme-bag PHON. Word PHON activation = union of phoneme blocks.

PHON is restructured: one explicit block per unique phoneme in the lexicon.
A word's PHON activation is the *union* of the blocks for its phonemes (no
position information; multisets collapsed to sets).

Training: each training word fires PHON-as-phoneme-union plus its assigned
context (VISUAL or MOTOR), mutual-inhibited toward the target hub.

Probe: held-out words (in the 'probe_words' bucket of the lexicon) are
NEVER trained. After training is done, fire their phoneme-union into PHON
and project to {NOUN, VERB}. Measure NOUN_in and VERB_in.

Phase 3 hypothesis: a probe word like `riverrun` whose phoneme bag is the
union of a trained noun (`river`) and a trained verb (`run`) develops
bilateral routing automatically — without being declared ambiguous in the
lexicon, and without any sub-word-level labeling. The shared phonemic
substrate is what propagates the noun/verb signal.

Why this matters: the previous experiments (binary dual co-firing, graded
stochastic) required us to explicitly declare which words are ambiguous.
Phoneme-bag PHON lets ambiguity *emerge* from the phonemic composition.
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
                        "Each entry needs 'text', 'phonemes' (IPA list), and "
                        "'bucket' (for training_words only).")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--n", type=int, default=10000,
                   help="LEX_n for NOUN/VERB sparse hubs.")
    p.add_argument("--lex-k", type=int, default=100)
    p.add_argument("--phon-k", type=int, default=20,
                   help="Neurons per phoneme block. Smaller default than other "
                        "scripts so that words with N phonemes activate ~N*phon-k "
                        "PHON neurons (comparable to ctx-k for small N).")
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
    # Collect all unique phonemes across training + probe to size PHON.
    phonemes = set()
    for w in train + probe:
        if "phonemes" not in w:
            sys.exit(f"Word {w.get('text', '?')} missing 'phonemes' list.")
        phonemes.update(w["phonemes"])
    phoneme_to_idx = {ph: i for i, ph in enumerate(sorted(phonemes))}
    # Assign per-word context slots. Only training words get a context slot
    # (probe words are never trained, so they never need to fire VISUAL/MOTOR).
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
    return train, probe, phoneme_to_idx, visual_counter, motor_counter, meta


def build_brain(phoneme_to_idx, num_visual, num_motor, args):
    import brain
    b = brain.Brain(args.p)
    b.add_explicit_area(PHON, len(phoneme_to_idx) * args.phon_k, args.phon_k, args.beta)
    # Each context area has at least 1 slot to avoid zero-size allocation.
    b.add_explicit_area(VISUAL, max(num_visual, 1) * args.ctx_k, args.ctx_k, args.beta)
    b.add_explicit_area(MOTOR, max(num_motor, 1) * args.ctx_k, args.ctx_k, args.beta)
    b.add_area(NOUN, args.n, args.lex_k, args.beta)
    b.add_area(VERB, args.n, args.lex_k, args.beta)
    return b


def activate_phoneme_union(b, phonemes, phoneme_to_idx, phon_k):
    """Set PHON.winners to the union of phoneme blocks for `phonemes`.
    PHON is explicit, so we can freely set winners to any subset of n."""
    winners = []
    for ph in set(phonemes):  # dedupe; bag-of-phonemes
        idx = phoneme_to_idx[ph]
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


def train_word(b, word, phoneme_to_idx, args):
    activate_phoneme_union(b, word["phonemes"], phoneme_to_idx, args.phon_k)
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
            PHON: [VERB], MOTOR: [VERB],
            VERB: [PHON, VERB, MOTOR],
        }
    b.project({}, first_proj)
    for _ in range(args.proj_rounds):
        b.project({}, recurrent)
    clear_all(b)


def phon_inputs(b, phonemes, phoneme_to_idx, phon_k):
    """Fire phoneme union, project to both hubs, return (noun_in, verb_in)."""
    clear_all(b)
    activate_phoneme_union(b, phonemes, phoneme_to_idx, phon_k)
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
    run_dir = make_run_dir("phoneme_phon")
    open_tee_log(run_dir)
    add_nemo_to_path()
    setup_seeded(args.seed)

    train, probe, phoneme_to_idx, num_visual, num_motor, lex_meta = load_lexicon(args.lexicon)

    save_json(run_dir, "config.json", {
        "experiment": "phoneme_phon",
        "lexicon_path": str(args.lexicon),
        "lexicon_meta": lex_meta,
        "phoneme_to_idx": phoneme_to_idx,
        "num_phonemes": len(phoneme_to_idx),
        "num_training_words": len(train),
        "num_probe_words": len(probe),
        "training_words": [{"text": w["text"], "bucket": w["bucket"],
                            "phonemes": w["phonemes"]} for w in train],
        "probe_words": [{"text": w["text"], "phonemes": w["phonemes"]} for w in probe],
        "rounds": args.rounds, "n": args.n, "lex_k": args.lex_k,
        "phon_k": args.phon_k, "ctx_k": args.ctx_k, "p": args.p,
        "beta": args.beta, "proj_rounds": args.proj_rounds, "seed": args.seed,
    })

    t0 = time.time()
    print(
        f"[{time.time()-t0:6.1f}s] Building brain "
        f"(num_phonemes={len(phoneme_to_idx)}, PHON_n={len(phoneme_to_idx) * args.phon_k}, "
        f"VISUAL_slots={num_visual}, MOTOR_slots={num_motor})",
        flush=True,
    )
    print(
        f"  Phoneme inventory: {sorted(phoneme_to_idx.keys())}",
        flush=True,
    )
    b = build_brain(phoneme_to_idx, num_visual, num_motor, args)

    print(
        f"[{time.time()-t0:6.1f}s] Training {args.rounds} rounds × "
        f"{len(train)} words = {args.rounds * len(train)} tutoring steps",
        flush=True,
    )
    for r in range(args.rounds):
        for w in train:
            train_word(b, w, phoneme_to_idx, args)
        if (r + 1) % 10 == 0 or r == args.rounds - 1:
            print(f"  [{time.time()-t0:6.1f}s] round {r+1}/{args.rounds}", flush=True)

    b.disable_plasticity = True

    # ===== Probe trained words =====
    print(f"\n[{time.time()-t0:6.1f}s] === Trained words: Q ratios ===", flush=True)
    train_results = []
    for w in train:
        n_in, v_in = phon_inputs(b, w["phonemes"], phoneme_to_idx, args.phon_k)
        ratio = n_in / v_in if v_in > 0 else float("inf")
        train_results.append({
            "text": w["text"], "bucket": w["bucket"],
            "phonemes": w["phonemes"],
            "noun_input": n_in, "verb_input": v_in,
            "ratio_noun_to_verb": ratio,
        })
        phon_str = " ".join(w["phonemes"])
        print(
            f"  {w['text']:10s} ({w['bucket']:4s})  /{phon_str:18s}/  "
            f"NOUN_in={n_in:>9.1f}  VERB_in={v_in:>9.1f}  N/V={ratio:>7.2f}",
            flush=True,
        )

    # ===== Probe held-out words =====
    print(f"\n[{time.time()-t0:6.1f}s] === Held-out probe words: Q ratios ===", flush=True)
    probe_results = []
    for w in probe:
        n_in, v_in = phon_inputs(b, w["phonemes"], phoneme_to_idx, args.phon_k)
        ratio = n_in / v_in if v_in > 0 else float("inf")
        probe_results.append({
            "text": w["text"], "phonemes": w["phonemes"],
            "expected_composition": w.get("expected_composition"),
            "noun_input": n_in, "verb_input": v_in,
            "ratio_noun_to_verb": ratio,
        })
        phon_str = " ".join(w["phonemes"])
        print(
            f"  {w['text']:10s} (HELDOUT) /{phon_str:18s}/  "
            f"NOUN_in={n_in:>9.1f}  VERB_in={v_in:>9.1f}  N/V={ratio:>7.2f}",
            flush=True,
        )
        if w.get("expected_composition"):
            print(f"    composition: {w['expected_composition']}", flush=True)

    b.disable_plasticity = False

    # ===== Per-phoneme attribution: fire just one phoneme block at a time =====
    print(f"\n[{time.time()-t0:6.1f}s] === Per-phoneme NOUN/VERB attribution ===", flush=True)
    b.disable_plasticity = True
    phoneme_results = []
    for ph in sorted(phoneme_to_idx.keys()):
        n_in, v_in = phon_inputs(b, [ph], phoneme_to_idx, args.phon_k)
        ratio = n_in / v_in if v_in > 0 else float("inf")
        # Find which training words contain this phoneme
        appears_in_n = [w["text"] for w in train if w["bucket"] == "noun" and ph in w["phonemes"]]
        appears_in_v = [w["text"] for w in train if w["bucket"] == "verb" and ph in w["phonemes"]]
        phoneme_results.append({
            "phoneme": ph,
            "noun_input": n_in, "verb_input": v_in,
            "ratio_noun_to_verb": ratio,
            "trained_in_nouns": appears_in_n,
            "trained_in_verbs": appears_in_v,
        })
        provenance = f"N:{appears_in_n} / V:{appears_in_v}" if (appears_in_n or appears_in_v) else "untrained"
        print(
            f"  /{ph:3s}/  NOUN_in={n_in:>9.1f}  VERB_in={v_in:>9.1f}  N/V={ratio:>7.2f}  "
            f"{provenance}",
            flush=True,
        )
    b.disable_plasticity = False

    elapsed = time.time() - t0
    save_json(run_dir, "results.json", {
        "experiment": "phoneme_phon",
        "elapsed_seconds": elapsed,
        "trained_q": train_results,
        "probe_q": probe_results,
        "per_phoneme_q": phoneme_results,
    })
    print(f"\n[{elapsed:.1f}s] Results written to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
