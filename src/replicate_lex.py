#!/usr/bin/env python3
"""Replicate §3 lexicon acquisition (Papadimitriou et al.).

Trains a LearnBrain on a toy language with N nouns and N verbs (default 2 each),
2-word intransitive sentences. Verifies after training:
  - Property P: given a context (VISUAL/MOTOR), recover correct word's PHON.
  - Property Q: given a PHON, route to correct lexical hub (NOUN or VERB).
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

# Matches PHON_INDICES in nemo-core/learner.py: DOG=0, CAT=1, JUMP=2, RUN=3.
NAMED_CORPUS_WORDS = ["DOG", "CAT", "JUMP", "RUN"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-nouns", type=int, default=2)
    p.add_argument("--num-verbs", type=int, default=2)
    p.add_argument("--rounds", type=int, default=30,
                   help="passes over all num_nouns x num_verbs sentence combos")
    p.add_argument("--n", type=int, default=10000,
                   help="LEX_n for NOUN and VERB sparse areas")
    p.add_argument("--k", type=int, default=100,
                   help="k (firing count) for explicit and sparse areas")
    p.add_argument("--p", type=float, default=0.05, help="connection probability")
    p.add_argument("--beta", type=float, default=0.06, help="Hebbian plasticity rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use-named-corpus", action="store_true",
                   help="Use train_simple with named DOG/CAT/JUMP/RUN words. "
                        "Requires --num-nouns 2 --num-verbs 2.")
    p.add_argument("--lexicon", type=Path, default=None,
                   help="Path to a lexicon JSON file with 'nouns' and 'verbs' arrays "
                        "(each item: {text, pos, page_line}). Overrides --num-nouns/--num-verbs "
                        "and mutually exclusive with --use-named-corpus.")
    return p.parse_args()


def load_lexicon(path: Path):
    """Load and validate a lexicon JSON file. Returns (noun_names, verb_names, metadata)."""
    with path.open() as f:
        data = json.load(f)
    if "nouns" not in data or "verbs" not in data:
        sys.exit(f"Lexicon {path} must contain 'nouns' and 'verbs' keys.")
    noun_names, verb_names = [], []
    for entry in data["nouns"]:
        if entry.get("pos") not in (None, "NOUN", "PROPN"):
            sys.exit(f"Noun entry in {path} has pos={entry.get('pos')!r}: {entry}")
        noun_names.append(entry["text"])
    for entry in data["verbs"]:
        if entry.get("pos") not in (None, "VERB", "AUX"):
            sys.exit(f"Verb entry in {path} has pos={entry.get('pos')!r}: {entry}")
        verb_names.append(entry["text"])
    metadata = {k: v for k, v in data.items() if k not in ("nouns", "verbs")}
    return noun_names, verb_names, metadata


def main():
    args = parse_args()
    if args.use_named_corpus and (args.num_nouns != 2 or args.num_verbs != 2):
        sys.exit("--use-named-corpus requires --num-nouns 2 --num-verbs 2 "
                 "(nemo-core's train_simple is locked to 4 named sentences).")
    if args.lexicon and args.use_named_corpus:
        sys.exit("--lexicon and --use-named-corpus are mutually exclusive.")

    lexicon_nouns, lexicon_verbs, lexicon_meta = (None, None, None)
    if args.lexicon:
        lexicon_nouns, lexicon_verbs, lexicon_meta = load_lexicon(args.lexicon)
        args.num_nouns = len(lexicon_nouns)
        args.num_verbs = len(lexicon_verbs)

    ensure_hash_seed(args.seed)

    run_dir = make_run_dir("lex")
    open_tee_log(run_dir)

    add_nemo_to_path()
    setup_seeded(args.seed)
    import learner

    config = {"experiment": "lex", **{k: str(v) if isinstance(v, Path) else v
                                       for k, v in vars(args).items()}}
    if lexicon_meta is not None:
        config["lexicon_meta"] = lexicon_meta
        config["lexicon_nouns"] = lexicon_nouns
        config["lexicon_verbs"] = lexicon_verbs
    save_json(run_dir, "config.json", config)

    t0 = time.time()
    print(
        f"[{time.time()-t0:6.1f}s] Building LearnBrain "
        f"(n={args.n}, k={args.k}, num_nouns={args.num_nouns}, "
        f"num_verbs={args.num_verbs}, p={args.p}, beta={args.beta}, "
        f"seed={args.seed})",
        flush=True,
    )
    b = learner.LearnBrain(
        args.p,
        LEX_k=args.k,
        LEX_n=args.n,
        PHON_k=args.k,
        CONTEXTUAL_k=args.k,
        num_nouns=args.num_nouns,
        num_verbs=args.num_verbs,
        beta=args.beta,
    )

    if args.use_named_corpus:
        print(
            f"[{time.time()-t0:6.1f}s] Training {args.rounds} rounds of the named "
            f"4-sentence corpus [CAT JUMP, CAT RUN, DOG JUMP, DOG RUN]...",
            flush=True,
        )
        b.train_simple(args.rounds)
    else:
        total_sentences = args.num_nouns * args.num_verbs
        lex_desc = (
            f"lexicon={lexicon_nouns} + {lexicon_verbs}"
            if lexicon_nouns is not None
            else "anonymous indices"
        )
        print(
            f"[{time.time()-t0:6.1f}s] Training {args.rounds} rounds over "
            f"{total_sentences} sentence combos ({args.rounds * total_sentences} total). "
            f"{lex_desc}",
            flush=True,
        )
        b.train(args.rounds)

    all_lexicon_names = (
        (lexicon_nouns + lexicon_verbs) if lexicon_nouns is not None else None
    )

    def label(i: int) -> str:
        if all_lexicon_names is not None:
            return all_lexicon_names[i]
        if args.use_named_corpus:
            return NAMED_CORPUS_WORDS[i]
        return f"word {i}"

    # Property P
    print(f"\n[{time.time()-t0:6.1f}s] === Property P (context -> word) ===", flush=True)
    p_results = {}
    for w_idx in range(args.num_nouns + args.num_verbs):
        recovered = b.testIndexedWord(w_idx, no_print=True)
        ok = recovered == w_idx
        recovered_label = label(recovered) if isinstance(recovered, int) else None
        p_results[label(w_idx)] = {
            "recovered_index": recovered,
            "recovered": recovered_label,
            "ok": ok,
        }
        print(
            f"  {label(w_idx)}: recovered={recovered_label} ({'ok' if ok else 'FAIL'})",
            flush=True,
        )
    p_pass = all(r["ok"] for r in p_results.values())

    # Property Q
    print(f"\n[{time.time()-t0:6.1f}s] === Property Q (PHON -> POS routing) ===", flush=True)
    q_results = {}
    for w_idx in range(args.num_nouns + args.num_verbs):
        b.disable_plasticity = True
        b.activate(learner.PHON, w_idx)
        b.project({}, {learner.PHON: [learner.NOUN, learner.VERB]})
        noun_in = float(b.get_input_from(learner.PHON, learner.NOUN))
        verb_in = float(b.get_input_from(learner.PHON, learner.VERB))
        expected = "NOUN" if w_idx < args.num_nouns else "VERB"
        winner = "NOUN" if noun_in > verb_in else "VERB"
        ok = winner == expected
        q_results[label(w_idx)] = {
            "noun_input": noun_in,
            "verb_input": verb_in,
            "winner": winner,
            "expected": expected,
            "ok": ok,
        }
        print(
            f"  {label(w_idx)} ({expected}): "
            f"NOUN_in={noun_in:.1f}, VERB_in={verb_in:.1f} "
            f"-> {winner} ({'ok' if ok else 'FAIL'})",
            flush=True,
        )
        b.disable_plasticity = False
    q_pass = all(r["ok"] for r in q_results.values())

    elapsed = time.time() - t0
    print(
        f"\n[{elapsed:6.1f}s] DONE. "
        f"Property P: {'PASS' if p_pass else 'FAIL'}. "
        f"Property Q: {'PASS' if q_pass else 'FAIL'}.",
        flush=True,
    )

    save_json(run_dir, "results.json", {
        "experiment": "lex",
        "elapsed_seconds": elapsed,
        "property_P": {"results": p_results, "pass": p_pass},
        "property_Q": {"results": q_results, "pass": q_pass},
    })
    print(f"Results written to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
