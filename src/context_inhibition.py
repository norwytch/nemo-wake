"""Post-hoc gain control (lateral-inhibition surrogate) for CONTEXT slots.

Hebbian collapse under class imbalance leaves CONTEXT slots with wildly different
baseline activation: the dominant class (en-us) fires high into every probe, the
minority slots sit near zero. The raw argmax-overlap read-out therefore almost always
picks the dominant slot, driving minority recall to ~0.

This module standardizes each slot's overlap against per-slot statistics estimated on
the held-out *val* split, then re-argmaxes on *test* only. It is a leakage-free
read-out change — no retraining, no nemo-core edit — and the evaluator-side surrogate
of the lateral inhibition Mitropolsky & Papadimitriou (2025) use between MOOD slots.

CLI:
  # single run: raw vs inhibited test metrics + paired bootstrap CI on the delta
  python3 src/context_inhibition.py --inputs results/mlctx_.../results.json

  # multi-seed: mean +/- std of raw vs inhibited across seeds
  python3 src/context_inhibition.py --inputs results/mlctx_a/results.json \
      results/mlctx_b/results.json ... --aggregate
"""
import argparse
import json
import math
import random
from pathlib import Path

from evaluate_attribution import score_method


def _profile_map(rec):
    return {p["language"]: p["overlap"] for p in rec["profile"]}


def fit_slot_stats(val_records, languages):
    """Per-slot (mean, std) of overlap across the val split."""
    stats = {}
    for L in languages:
        xs = [_profile_map(r).get(L, 0) for r in val_records]
        mu = sum(xs) / len(xs) if xs else 0.0
        var = sum((x - mu) ** 2 for x in xs) / len(xs) if xs else 0.0
        stats[L] = (mu, math.sqrt(var))
    return stats


def inhibited_top1(rec, languages, stats, eps=1e-6):
    pm = _profile_map(rec)
    best_z, best_L = None, None
    for L in languages:
        mu, sd = stats[L]
        z = (pm.get(L, 0) - mu) / (sd + eps)
        if best_z is None or z > best_z:
            best_z, best_L = z, L
    return best_L


def load(path):
    d = json.loads(Path(path).read_text())
    recs = d.get("probe_results") or d.get("probe_q")
    langs = d["languages"]
    val = [r for r in recs if r.get("split") == "val"]
    test = [r for r in recs if r.get("split") == "test"]
    return langs, val, test


def rescore(path):
    """Return raw-vs-inhibited test-only metrics for one run."""
    langs, val, test = load(path)
    stats = fit_slot_stats(val, langs)
    raw = test
    inhib = [{**r, "top1": inhibited_top1(r, langs, stats)} for r in test]
    return {
        "path": str(path),
        "n_test": len(test),
        "n_test_foreign": sum(1 for r in test if set(r["gold_languages"]) - {"en-us"}),
        "raw": score_method(raw, langs),
        "inhibited": score_method(inhib, langs),
        "_raw_recs": raw, "_inhib_recs": inhib, "_langs": langs,
    }


def _foreign_correct(recs):
    f = [r for r in recs if set(r["gold_languages"]) - {"en-us"}]
    return sum(1 for r in f if r["top1"] in r["gold_languages"]), len(f)


def paired_bootstrap(res, n_boot=2000, seed=0):
    """Paired bootstrap over test lines for a CI on the inhibited-minus-raw delta.

    Resamples test-line indices once per iteration and recomputes both read-outs on the
    same resample, so the two methods see identical lines (paired). Reports the foreign
    -gold accuracy of each and the delta.
    """
    raw, inhib, langs = res["_raw_recs"], res["_inhib_recs"], res["_langs"]
    rng = random.Random(seed)
    n = len(raw)
    idx = list(range(n))
    raw_f, inhib_f, deltas = [], [], []
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        rr = [raw[i] for i in sample]
        ii = [inhib[i] for i in sample]
        rc, rn = _foreign_correct(rr)
        ic, _n = _foreign_correct(ii)
        rf = rc / rn if rn else 0.0
        iff = ic / rn if rn else 0.0
        raw_f.append(rf)
        inhib_f.append(iff)
        deltas.append(iff - rf)

    def ci(xs):
        xs = sorted(xs)
        lo = xs[int(0.025 * len(xs))]
        hi = xs[int(0.975 * len(xs))]
        return round(lo, 4), round(hi, 4)

    return {
        "n_boot": n_boot,
        "raw_foreign_ci": ci(raw_f),
        "inhibited_foreign_ci": ci(inhib_f),
        "delta_foreign_ci": ci(deltas),
        "delta_foreign_mean": round(sum(deltas) / len(deltas), 4),
        "delta_excludes_zero": (ci(deltas)[0] > 0) or (ci(deltas)[1] < 0),
    }


def aggregate(paths):
    """Mean +/- std of raw vs inhibited foreign-gold + macro-F1 across seeds."""
    rows = [rescore(p) for p in paths]

    def stat(getter):
        xs = [getter(r) for r in rows]
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
        return round(mu, 4), round(sd, 4), [round(x, 4) for x in xs]

    return {
        "n_seeds": len(rows),
        "raw_foreign": stat(lambda r: r["raw"]["foreign_subset_top1_in_gold"]),
        "inhibited_foreign": stat(lambda r: r["inhibited"]["foreign_subset_top1_in_gold"]),
        "raw_macro_f1": stat(lambda r: r["raw"]["macro_f1"]),
        "inhibited_macro_f1": stat(lambda r: r["inhibited"]["macro_f1"]),
        "per_seed": [{"path": r["path"],
                      "raw_foreign": r["raw"]["foreign_subset_top1_in_gold"],
                      "inhibited_foreign": r["inhibited"]["foreign_subset_top1_in_gold"],
                      "raw_macro_f1": r["raw"]["macro_f1"],
                      "inhibited_macro_f1": r["inhibited"]["macro_f1"]} for r in rows],
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", type=Path, required=True,
                   help="One or more mlctx results.json paths.")
    p.add_argument("--aggregate", action="store_true",
                   help="Aggregate mean+/-std across the inputs (multi-seed).")
    p.add_argument("--boot", type=int, default=2000, help="Bootstrap iterations (single-run mode).")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.aggregate:
        out = aggregate(args.inputs)
        print(json.dumps(out, indent=2))
    else:
        out = {}
        for p in args.inputs:
            res = rescore(p)
            boot = paired_bootstrap(res, n_boot=args.boot)
            summary = {
                "path": res["path"], "n_test": res["n_test"],
                "n_test_foreign": res["n_test_foreign"],
                "raw_foreign": res["raw"]["foreign_subset_top1_in_gold"],
                "inhibited_foreign": res["inhibited"]["foreign_subset_top1_in_gold"],
                "raw_macro_f1": res["raw"]["macro_f1"],
                "inhibited_macro_f1": res["inhibited"]["macro_f1"],
                "bootstrap": boot,
            }
            out[res["path"]] = summary
            print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
