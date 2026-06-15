"""Superposition with ground truth, in the Assembly Calculus.

Mechanistic-interpretability toy model on a biologically-motivated substrate. The features
are known by construction: each concept is a stimulus whose assembly (its k-cap winner set
in one area) we record. We pack more and more concepts into the area and watch what happens
to the known features as load rises — the ground truth interpretability methods usually lack.

Concepts are trained sequentially. Right after a concept is trained we snapshot its assembly;
after every later concept has trained, we re-fire it and see how much it drifted. For each
total count M (area of n neurons, cap k, so n/k nominal "slots") we measure:

  - overlap        mean pairwise Jaccard between final assemblies (chance ~ k/n)
  - polysemanticity for each used neuron, how many assemblies it belongs to
  - retention      mean Jaccard(snapshot, final) over concepts — 1.0 = no forgetting;
                   falling retention = catastrophic interference from later learning
  - first_retention retention of concept 0 specifically (it sees the most later training)

There is also a hard ceiling: a non-explicit area grows its support as it recruits winners,
and past some load it can no longer sample k fresh neurons for a new assembly. We catch that
and report the largest M the substrate could actually hold.

Usage:
  python3 src/superposition_testbed.py --n 4000 --k 50 --concepts 4 8 16 32 64 --seed 0
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

from _common import add_nemo_to_path, ensure_hash_seed, make_run_dir, save_json, setup_seeded

AREA = "A"


def _fire(b, stim):
    """Fire a stimulus into the area with plasticity off; return induced winner set."""
    b.area_by_name[AREA].winners = []
    b.project({stim: [AREA]}, {})
    return set(b.area_by_name[AREA].winners)


def jaccard(a, c):
    u = len(a | c)
    return len(a & c) / u if u else 0.0


def run_one(n, k, beta, m, rounds):
    """Sequentially train m concepts; snapshot each as it forms, then re-fire at the end.

    Returns (snapshots, finals, reached) where reached is the count actually trained before
    the area ran out of room to allocate a new assembly (the substrate's hard ceiling).
    """
    import brain
    b = brain.Brain(0.05)
    b.add_area(AREA, n, k, beta)

    snapshots, stims = {}, {}
    reached = 0
    for c in range(m):
        s = f"c{c}"
        b.add_stimulus(s, k)
        stims[c] = s
        try:
            b.disable_plasticity = False
            b.project({s: [AREA]}, {})                 # form
            for _ in range(rounds - 1):
                b.project({s: [AREA]}, {AREA: [AREA]})  # stabilise
        except RuntimeError:
            break                                       # area is full — capacity ceiling
        b.disable_plasticity = True
        snapshots[c] = _fire(b, s)
        reached = c + 1

    b.disable_plasticity = True
    finals = {c: _fire(b, stims[c]) for c in range(reached)}
    return snapshots, finals, reached


def metrics(snapshots, finals, reached, n, k):
    finals_list = [finals[c] for c in range(reached)]
    # pairwise Jaccard overlap (chance ~ k/n)
    jac = [jaccard(finals[i], finals[j]) for i, j in combinations(range(reached), 2)]
    mean_overlap = sum(jac) / len(jac) if jac else 0.0

    # neuron polysemanticity
    membership = {}
    for a in finals_list:
        for nrn in a:
            membership[nrn] = membership.get(nrn, 0) + 1
    used = len(membership)
    frac_poly = sum(1 for v in membership.values() if v > 1) / used if used else 0.0
    mean_member = sum(membership.values()) / used if used else 0.0
    max_member = max(membership.values()) if membership else 0

    # retention: how much each concept's assembly survived later training
    ret = [jaccard(snapshots[c], finals[c]) for c in range(reached)]
    mean_ret = sum(ret) / len(ret) if ret else 0.0
    first_ret = ret[0] if ret else 0.0

    return {
        "concepts_requested": None,  # filled by caller
        "concepts_held": reached,
        "load_Mk_over_n": round(reached * k / n, 3),
        "mean_pairwise_overlap": round(mean_overlap, 4),
        "frac_used_polysemantic": round(frac_poly, 4),
        "mean_assemblies_per_used_neuron": round(mean_member, 3),
        "max_assemblies_per_neuron": max_member,
        "mean_retention": round(mean_ret, 4),
        "first_concept_retention": round(first_ret, 4),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4000, help="Area size.")
    p.add_argument("--k", type=int, default=50, help="Cap (assembly size). n/k = #slots.")
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--concepts", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                   help="Values of M (number of concepts) to sweep.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_hash_seed(args.seed)
    run_dir = make_run_dir("superposition")
    add_nemo_to_path()
    setup_seeded(args.seed)

    slots = args.n // args.k
    print(f"area n={args.n} k={args.k}  ->  {slots} nominal slots (n/k)\n")
    header = (f"{'M':>4} {'held':>5} {'load':>5} {'overlap':>8} {'%poly':>7} "
              f"{'mean/nrn':>9} {'max':>4} {'retain':>7} {'c0ret':>7}")
    print(header)
    rows = []
    for m in args.concepts:
        snaps, finals, reached = run_one(args.n, args.k, args.beta, m, args.rounds)
        row = metrics(snaps, finals, reached, args.n, args.k)
        row["concepts_requested"] = m
        rows.append(row)
        print(f"{m:>4} {row['concepts_held']:>5} {row['load_Mk_over_n']:>5} "
              f"{row['mean_pairwise_overlap']:>8} {row['frac_used_polysemantic']:>7} "
              f"{row['mean_assemblies_per_used_neuron']:>9} {row['max_assemblies_per_neuron']:>4} "
              f"{row['mean_retention']:>7} {row['first_concept_retention']:>7}", flush=True)
        if row["concepts_held"] < m:
            print(f"     (area capacity ceiling: held {row['concepts_held']} of {m} requested)",
                  flush=True)

    save_json(run_dir, "results.json", {
        "experiment": "superposition_testbed",
        "n": args.n, "k": args.k, "beta": args.beta, "rounds": args.rounds,
        "slots": slots, "seed": args.seed, "sweep": rows,
    })
    print(f"\nWrote {run_dir}/results.json")


if __name__ == "__main__":
    main()
