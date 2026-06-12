#!/usr/bin/env python3
"""Replicate §4 word order with transitive verbs, VSO order (Papadimitriou et al.).

Trains a wo.LearnBrain on transitive sentences ordered Verb-Subject-Object.
After training, generates `trials` sentences and reports how many emit VSO.
"""
import argparse
import time

from _common import (
    add_nemo_to_path,
    ensure_hash_seed,
    make_run_dir,
    open_tee_log,
    save_json,
    setup_seeded,
)

ORDER = ["V", "S", "O"]
ORDER_NAME = "VSO"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-nouns", type=int, default=2)
    p.add_argument("--num-verbs", type=int, default=2)
    p.add_argument("--rounds", type=int, default=10,
                   help="number of random training sentences")
    p.add_argument("--n", type=int, default=10000,
                   help="NON_EXPLICIT_n for TPJ / SYNTAX areas")
    p.add_argument("--k", type=int, default=100,
                   help="k (firing count) for explicit and sparse areas")
    p.add_argument("--p", type=float, default=0.1, help="connection probability")
    p.add_argument("--beta", type=float, default=0.06, help="Hebbian plasticity rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=5,
                   help="number of generation trials after training")
    p.add_argument("--num-tpj-firings", type=int, default=3,
                   help="TPJ firings during generation (paper default: 3)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_hash_seed(args.seed)

    run_dir = make_run_dir(f"word_order_{ORDER_NAME.lower()}")
    open_tee_log(run_dir)

    add_nemo_to_path()
    setup_seeded(args.seed)
    import word_order_int as wo

    save_json(run_dir, "config.json", {
        "experiment": f"word_order_{ORDER_NAME.lower()}",
        "order": ORDER,
        **vars(args),
    })

    t0 = time.time()
    print(
        f"[{time.time()-t0:6.1f}s] Building word-order LearnBrain "
        f"(n={args.n}, k={args.k}, num_nouns={args.num_nouns}, "
        f"num_verbs={args.num_verbs}, p={args.p}, beta={args.beta}, "
        f"seed={args.seed}, mood 0 = {ORDER_NAME})",
        flush=True,
    )
    b = wo.LearnBrain(
        args.p,
        EXPLICIT_k=args.k,
        NON_EXPLICIT_k=args.k,
        NON_EXPLICIT_n=args.n,
        beta=args.beta,
        num_nouns=args.num_nouns,
        num_verbs=args.num_verbs,
        num_moods=1,
        mood_to_trans_word_order={0: ORDER},
    )

    print(
        f"[{time.time()-t0:6.1f}s] Training {args.rounds} transitive "
        f"{ORDER_NAME} sentences...",
        flush=True,
    )
    sentence_times = []
    for i in range(args.rounds):
        t1 = time.time()
        b.input_random_trans_sentence()
        dt = time.time() - t1
        sentence_times.append(dt)
        print(f"[{time.time()-t0:6.1f}s]   sentence {i}: {dt:.1f}s", flush=True)

    print(
        f"\n[{time.time()-t0:6.1f}s] === Generating {args.trials} sentences ===",
        flush=True,
    )
    trial_results = []
    for i in range(args.trials):
        print(f"-- trial {i} --", flush=True)
        order = b.generate_random_sentence(
            mood_index=0, num_tpj_firings=args.num_tpj_firings,
        )
        trial_results.append(list(order))
        print(f"  emitted: {order}", flush=True)

    correct = sum(1 for o in trial_results if o == ORDER)
    elapsed = time.time() - t0
    print(
        f"\n[{elapsed:6.1f}s] DONE. {correct}/{args.trials} trials emitted {ORDER_NAME}.",
        flush=True,
    )

    save_json(run_dir, "results.json", {
        "experiment": f"word_order_{ORDER_NAME.lower()}",
        "target_order": ORDER,
        "elapsed_seconds": elapsed,
        "sentence_times": sentence_times,
        "trials": trial_results,
        "correct": correct,
        "total_trials": args.trials,
    })
    print(f"Results written to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
