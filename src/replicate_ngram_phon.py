#!/usr/bin/env python3
"""N-gram PHON. Word PHON activation = union of contiguous phoneme-n-gram blocks.

Order-aware variant of replicate_phoneme_phon.py. Where that script collapses a
word to a *set* of phoneme blocks (bag-of-phonemes, order-free), this one uses
contiguous phoneme n-grams as the unit:

    riverrun  /ɹ ɪ v ʌ n/   --ngram 2-->  {ɹ|ɪ, ɪ|v, v|ʌ, ʌ|n}
    river     /ɹ ɪ v ɚ/      --ngram 2-->  {ɹ|ɪ, ɪ|v, v|ɚ}
    run       /ɹ ʌ n/         --ngram 2-->  {ɹ|ʌ, ʌ|n}

So `riverrun` shares {ɹ|ɪ, ɪ|v} with `river` and {ʌ|n} with `run` — the overlap
is the shared *ordered* substring, position-invariant. With --ngram 1 the unit
is the bare phoneme and the script is identical to replicate_phoneme_phon.py
(the bag baseline). This is the Tier-1 order ablation for idea 8: does an
order-aware phonemic substrate recover portmanteau decompositions better than
the order-free bag?

Two outputs per probe word:
  - noun_in / verb_in : the scalar N/V routing strength (as in phoneme_phon).
  - activation_profile : the PARTIAL-ACTIVATION MAP idea 8 names. After training,
    each trained sub-word's hub assembly is recorded (its k-cap in NOUN or VERB).
    At probe, the portmanteau's induced k-cap is overlapped against every trained
    word's anchor assembly. High overlap with `river` (noun) and `run` (verb) =
    the model has decomposed `riverrun` into those constituents, emergently,
    without sub-word labels.

Lexicon schema (same as replicate_phoneme_phon.py):
  training_words: [{"text", "bucket": "noun"|"verb", "phonemes": [str, ...]}]
  probe_words:    [{"text", "phonemes": [str, ...], "expected_composition"?}]
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
                   help="Lexicon JSON with 'training_words' and 'probe_words'. "
                        "Each entry needs 'text', 'phonemes' (IPA list), and "
                        "'bucket' (training_words only).")
    p.add_argument("--ngram", type=int, default=2,
                   help="N-gram size over phonemes. 1 = bag-of-phonemes baseline "
                        "(identical to replicate_phoneme_phon.py).")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--n", type=int, default=10000,
                   help="LEX_n for NOUN/VERB sparse hubs.")
    p.add_argument("--lex-k", type=int, default=100)
    p.add_argument("--phon-k", type=int, default=20,
                   help="Neurons per n-gram block.")
    p.add_argument("--ctx-k", type=int, default=100)
    p.add_argument("--p", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.06)
    p.add_argument("--proj-rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def word_units(phonemes, n):
    """Contiguous phoneme n-grams as '|'-joined strings.

    n<=1 -> bare phonemes (bag baseline). If the word is shorter than n, emit a
    single unit spanning the whole phoneme sequence so short words still get a
    PHON block.
    """
    if n <= 1:
        return list(phonemes)
    if len(phonemes) < n:
        return ["|".join(phonemes)]
    return ["|".join(phonemes[i:i + n]) for i in range(len(phonemes) - n + 1)]


def load_lexicon(path: Path, ngram: int):
    with path.open() as f:
        data = json.load(f)
    train = data.get("training_words", [])
    probe = data.get("probe_words", [])
    if not train:
        sys.exit(f"Lexicon {path} must contain 'training_words'.")
    units = set()
    for w in train + probe:
        if "phonemes" not in w:
            sys.exit(f"Word {w.get('text', '?')} missing 'phonemes' list.")
        w["_units"] = word_units(w["phonemes"], ngram)
        units.update(w["_units"])
    unit_to_idx = {u: i for i, u in enumerate(sorted(units))}
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
    return train, probe, unit_to_idx, visual_counter, motor_counter, meta


def build_brain(unit_to_idx, num_visual, num_motor, args):
    import brain
    b = brain.Brain(args.p)
    b.add_explicit_area(PHON, len(unit_to_idx) * args.phon_k, args.phon_k, args.beta)
    b.add_explicit_area(VISUAL, max(num_visual, 1) * args.ctx_k, args.ctx_k, args.beta)
    b.add_explicit_area(MOTOR, max(num_motor, 1) * args.ctx_k, args.ctx_k, args.beta)
    b.add_area(NOUN, args.n, args.lex_k, args.beta)
    b.add_area(VERB, args.n, args.lex_k, args.beta)
    return b


def activate_unit_union(b, units, unit_to_idx, phon_k):
    """Set PHON.winners to the union of n-gram blocks for `units`."""
    winners = []
    for u in set(units):
        idx = unit_to_idx[u]
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


def train_word(b, word, unit_to_idx, args):
    activate_unit_union(b, word["_units"], unit_to_idx, args.phon_k)
    if word["bucket"] == "noun":
        activate_block(b, VISUAL, word["visual_idx"])
        first_proj = {PHON: [NOUN, VERB], VISUAL: [NOUN]}
        recurrent = {PHON: [NOUN], VISUAL: [NOUN], NOUN: [PHON, NOUN, VISUAL]}
    else:
        activate_block(b, MOTOR, word["motor_idx"])
        first_proj = {PHON: [NOUN, VERB], MOTOR: [VERB]}
        recurrent = {PHON: [VERB], MOTOR: [VERB], VERB: [PHON, VERB, MOTOR]}
    b.project({}, first_proj)
    for _ in range(args.proj_rounds):
        b.project({}, recurrent)
    clear_all(b)


def fire_to_hubs(b, units, unit_to_idx, phon_k):
    """Fire a unit union, project PHON->{NOUN,VERB}, return (noun_in, verb_in, noun_w, verb_w)."""
    clear_all(b)
    activate_unit_union(b, units, unit_to_idx, phon_k)
    b.project({}, {PHON: [NOUN, VERB]})
    pw = b.area_by_name[PHON].winners
    nw = b.area_by_name[NOUN].winners
    vw = b.area_by_name[VERB].winners
    noun_in = float(b.connectomes[PHON][NOUN][pw][:, nw].sum()) if pw and nw else 0.0
    verb_in = float(b.connectomes[PHON][VERB][pw][:, vw].sum()) if pw and vw else 0.0
    return noun_in, verb_in, list(nw), list(vw)


def main():
    args = parse_args()
    ensure_hash_seed(args.seed)
    run_dir = make_run_dir(f"ngram{args.ngram}_phon")
    open_tee_log(run_dir)
    add_nemo_to_path()
    setup_seeded(args.seed)

    train, probe, unit_to_idx, num_visual, num_motor, lex_meta = load_lexicon(
        args.lexicon, args.ngram
    )

    save_json(run_dir, "config.json", {
        "experiment": "ngram_phon",
        "ngram": args.ngram,
        "lexicon_path": str(args.lexicon),
        "lexicon_meta": lex_meta,
        "num_units": len(unit_to_idx),
        "num_training_words": len(train),
        "num_probe_words": len(probe),
        "training_words": [{"text": w["text"], "bucket": w["bucket"],
                            "phonemes": w["phonemes"], "units": w["_units"]} for w in train],
        "probe_words": [{"text": w["text"], "phonemes": w["phonemes"],
                         "units": w["_units"]} for w in probe],
        "rounds": args.rounds, "n": args.n, "lex_k": args.lex_k,
        "phon_k": args.phon_k, "ctx_k": args.ctx_k, "p": args.p,
        "beta": args.beta, "proj_rounds": args.proj_rounds, "seed": args.seed,
    })

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] ngram={args.ngram}  num_units={len(unit_to_idx)}  "
          f"PHON_n={len(unit_to_idx) * args.phon_k}", flush=True)
    b = build_brain(unit_to_idx, num_visual, num_motor, args)

    print(f"[{time.time()-t0:6.1f}s] Training {args.rounds} rounds × {len(train)} words", flush=True)
    for r in range(args.rounds):
        for w in train:
            train_word(b, w, unit_to_idx, args)
        if (r + 1) % 10 == 0 or r == args.rounds - 1:
            print(f"  [{time.time()-t0:6.1f}s] round {r+1}/{args.rounds}", flush=True)

    b.disable_plasticity = True

    # ===== Record trained-word anchor assemblies (the partial-activation-map basis) =====
    print(f"\n[{time.time()-t0:6.1f}s] === Recording trained-word anchor assemblies ===", flush=True)
    anchors = []
    for w in train:
        _, _, nw, vw = fire_to_hubs(b, w["_units"], unit_to_idx, args.phon_k)
        winners = nw if w["bucket"] == "noun" else vw
        anchors.append({"text": w["text"], "bucket": w["bucket"], "winners": winners})

    def activation_profile(units):
        _, _, nw, vw = fire_to_hubs(b, units, unit_to_idx, args.phon_k)
        nw_set, vw_set = set(nw), set(vw)
        prof = []
        for a in anchors:
            hub = nw_set if a["bucket"] == "noun" else vw_set
            ov = len(hub & set(a["winners"]))
            prof.append({"trained_word": a["text"], "bucket": a["bucket"],
                         "overlap": ov, "overlap_frac": round(ov / args.lex_k, 3)})
        prof.sort(key=lambda x: -x["overlap"])
        return prof

    # ===== Probe trained words =====
    print(f"\n[{time.time()-t0:6.1f}s] === Trained words ===", flush=True)
    train_results = []
    for w in train:
        n_in, v_in, _, _ = fire_to_hubs(b, w["_units"], unit_to_idx, args.phon_k)
        ratio = n_in / v_in if v_in > 0 else float("inf")
        train_results.append({"text": w["text"], "bucket": w["bucket"],
                              "phonemes": w["phonemes"], "noun_input": n_in,
                              "verb_input": v_in, "ratio_noun_to_verb": ratio})
        print(f"  {w['text']:10s} ({w['bucket']:4s})  NOUN_in={n_in:>9.1f}  "
              f"VERB_in={v_in:>9.1f}  N/V={ratio:>7.2f}", flush=True)

    # ===== Probe held-out words + partial-activation map =====
    print(f"\n[{time.time()-t0:6.1f}s] === Held-out probe words + activation profile ===", flush=True)
    probe_results = []
    for w in probe:
        n_in, v_in, _, _ = fire_to_hubs(b, w["_units"], unit_to_idx, args.phon_k)
        ratio = n_in / v_in if v_in > 0 else float("inf")
        prof = activation_profile(w["_units"])
        probe_results.append({"text": w["text"], "phonemes": w["phonemes"],
                              "expected_composition": w.get("expected_composition"),
                              "noun_input": n_in, "verb_input": v_in,
                              "ratio_noun_to_verb": ratio,
                              "activation_profile": prof})
        print(f"  {w['text']:10s} (HELDOUT)  NOUN_in={n_in:>9.1f}  VERB_in={v_in:>9.1f}  "
              f"N/V={ratio:>7.2f}", flush=True)
        top = [f"{p['trained_word']}({p['bucket'][0]}):{p['overlap_frac']}" for p in prof[:5] if p["overlap"] > 0]
        print(f"    activates: {', '.join(top) if top else '(none)'}", flush=True)

    elapsed = time.time() - t0
    save_json(run_dir, "results.json", {
        "experiment": "ngram_phon",
        "ngram": args.ngram,
        "elapsed_seconds": elapsed,
        "trained_q": train_results,
        "probe_q": probe_results,
    })
    print(f"\n[{elapsed:.1f}s] Results written to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
