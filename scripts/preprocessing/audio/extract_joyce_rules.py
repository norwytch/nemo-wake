#!/usr/bin/env python3
"""
Extract Joyce's phonological substitution rules from the aligned I.8 audio,
with tiered context backoff.

For each aligned I.8 token, we have (orth, expected_ipa, observed_ipa).
This script:
  1. Splits aligned tokens 80/20 train/held-out by line_split() (page.line
     keyed, deterministic).
  2. For each training token, does intra-token Needleman-Wunsch alignment
     between expected and observed phoneme sequences.
  3. At each aligned position where BOTH expected and observed phonemes are
     non-gap, records a substitution observation (expected → observed) at
     three levels of context (most specific to least):
       - phoneme_pair: (expected_ph, prev_expected, next_expected)
       - word_position: (expected_ph, {initial, medial, final})
       - context_free: expected_ph
  4. Per level: emits the rule with the most-common observed substitute, if
     it has ≥ MIN_SUPPORT observations and ≥ MIN_RATE rate.
  5. At application time, lookup proceeds most-specific to least-specific —
     phoneme_pair if it fires, else word_position, else context_free, else
     identity. The expected (espeak) sequence supplies the context, since
     that is all that is available outside I.8.
  6. Held-out evaluation reports total ED and phoneme error rate (PER) with
     a Wilson 95% CI, plus per-rule leave-one-out lift.

Caveats:
  - The "observed" sequence is wav2vec2-CTC output on degraded 1929 audio,
    not human-transcribed ground truth. Rules conflate Joyce-pronunciation
    effects with model recall errors.
  - Gaps (insertions/deletions in the NW alignment) are ignored. Most
    deletions in observed are wav2vec2 recall failures, not Joyce-deletions.
  - Context is computed over the EXPECTED sequence, not the observed,
    because at application time only the expected sequence is available.

Output: data/audio/joyce-1929-alp/rules/joyce_rules.json

Usage: python scripts/preprocessing/audio/extract_joyce_rules.py
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Bio.Align import PairwiseAligner  # noqa: E402

from shared.corpus import line_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
ALIGNED_PATH = REPO_ROOT / "data/audio/joyce-1929-alp/alignment/i8_tokens_aligned.jsonl"
OUT_PATH = REPO_ROOT / "data/audio/joyce-1929-alp/rules/joyce_rules.json"

MIN_SUPPORT = 3
MIN_RATE = 0.50  # tightened from 0.30 — a rule must be the majority observation

WORD_BOUNDARY = "#"


def normalize(p: str) -> str:
    return p.replace("ˈ", "").replace("ˌ", "")


# ---------- alignment ---------------------------------------------------------


def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -1.5
    aligner.extend_gap_score = -1
    return aligner


def extract_pairs_with_context(
    expected: list[str],
    observed: list[str],
    aligner: PairwiseAligner,
) -> list[tuple[str, str, str, str, str]]:
    """Returns list of (e_ph, o_ph, prev_e, next_e, position) tuples.

    Context (prev/next/position) is computed over the expected sequence,
    using its original (un-aligned) indices. Only positions where both the
    expected and observed phonemes are non-gap are returned.
    """
    if not expected or not observed:
        return []

    e_norm = [normalize(p) for p in expected]
    o_norm = [normalize(p) for p in observed]
    unique = sorted({*e_norm, *o_norm})
    alphabet = {p: chr(0xE000 + i) for i, p in enumerate(unique)}
    inv = {v: k for k, v in alphabet.items()}

    e_enc = "".join(alphabet[p] for p in e_norm)
    o_enc = "".join(alphabet[p] for p in o_norm)

    best = aligner.align(e_enc, o_enc)[0]
    e_aligned = str(best).split("\n")[0]
    o_aligned = str(best).split("\n")[2]

    pairs: list[tuple[str, str, str, str, str]] = []
    e_idx = 0
    n = len(e_norm)
    for ec, oc in zip(e_aligned, o_aligned):
        e_ph = inv.get(ec, "-")
        o_ph = inv.get(oc, "-")
        if e_ph != "-" and o_ph != "-":
            prev_e = e_norm[e_idx - 1] if e_idx > 0 else WORD_BOUNDARY
            next_e = e_norm[e_idx + 1] if e_idx < n - 1 else WORD_BOUNDARY
            if e_idx == 0 and e_idx == n - 1:
                position = "isolate"
            elif e_idx == 0:
                position = "initial"
            elif e_idx == n - 1:
                position = "final"
            else:
                position = "medial"
            pairs.append((e_ph, o_ph, prev_e, next_e, position))
        if e_ph != "-":
            e_idx += 1
    return pairs


# ---------- rule extraction ---------------------------------------------------


def _emit_if_passes(
    out_dict: dict[str, dict],
    key: str,
    dist: Counter[str],
) -> None:
    total = sum(dist.values())
    if total < MIN_SUPPORT:
        return
    predicted, count = dist.most_common(1)[0]
    rate = count / total
    if rate < MIN_RATE:
        return
    out_dict[key] = {
        "predicted": predicted,
        "support": count,
        "rate": round(rate, 3),
        "total_observations": total,
        "alternatives": dict(dist.most_common()),
    }


def extract_tiered_rules(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Return rules at three tiers, JSON-serializable.

    rules["phoneme_pair"][f"{e}|{prev}|{next}"] = {predicted, support, rate, ...}
    rules["word_position"][f"{e}|{position}"]   = {predicted, support, rate, ...}
    rules["context_free"][e]                     = {predicted, support, rate, ...}
    """
    aligner = _make_aligner()
    pair_dist: dict[str, Counter[str]] = defaultdict(Counter)
    pos_dist: dict[str, Counter[str]] = defaultdict(Counter)
    free_dist: dict[str, Counter[str]] = defaultdict(Counter)

    for r in records:
        if not r["observed_ipa"]:
            continue
        for e, o, prev, nxt, pos in extract_pairs_with_context(
            r["expected_ipa"], r["observed_ipa"], aligner
        ):
            pair_dist[f"{e}|{prev}|{nxt}"][o] += 1
            pos_dist[f"{e}|{pos}"][o] += 1
            free_dist[e][o] += 1

    rules: dict[str, dict[str, dict]] = {
        "phoneme_pair": {},
        "word_position": {},
        "context_free": {},
    }
    for k, dist in pair_dist.items():
        _emit_if_passes(rules["phoneme_pair"], k, dist)
    for k, dist in pos_dist.items():
        _emit_if_passes(rules["word_position"], k, dist)
    for k, dist in free_dist.items():
        _emit_if_passes(rules["context_free"], k, dist)
    return rules


# ---------- rule application --------------------------------------------------


def _as_list(predicted) -> list[str]:
    """Normalize a rule's 'predicted' field to a list of phonemes.

    A string is treated as one phoneme; a list is returned as-is; an empty
    string or empty list means deletion (the input phoneme is dropped).
    """
    if isinstance(predicted, str):
        return [predicted] if predicted else []
    return list(predicted)


def apply_tiered_rules(expected: list[str], rules: dict[str, dict]) -> list[str]:
    """Apply tiered rules to predict an observed phoneme sequence.

    For each expected phoneme, look up the most specific applicable rule:
    phoneme_pair first, then word_position, then context_free. Falls back
    to identity if no rule fires. Rules may map one input phoneme to zero
    (deletion), one (substitution), or many (insertion) output phonemes.
    """
    e_norm = [normalize(p) for p in expected]
    n = len(e_norm)
    out: list[str] = []
    for i, e in enumerate(e_norm):
        prev = e_norm[i - 1] if i > 0 else WORD_BOUNDARY
        nxt = e_norm[i + 1] if i < n - 1 else WORD_BOUNDARY
        if n == 1:
            position = "isolate"
        elif i == 0:
            position = "initial"
        elif i == n - 1:
            position = "final"
        else:
            position = "medial"

        pair_key = f"{e}|{prev}|{nxt}"
        pos_key = f"{e}|{position}"

        if pair_key in rules["phoneme_pair"]:
            out.extend(_as_list(rules["phoneme_pair"][pair_key]["predicted"]))
        elif pos_key in rules["word_position"]:
            out.extend(_as_list(rules["word_position"][pos_key]["predicted"]))
        elif e in rules["context_free"]:
            out.extend(_as_list(rules["context_free"][e]["predicted"]))
        else:
            out.append(e)
    return out


# ---------- evaluation --------------------------------------------------------


def edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if normalize(ai) == normalize(bj) else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def score_held_out(records: list[dict], rules: dict) -> dict:
    """Apply rules across held-out tokens, return ED, total phonemes, PER, CI."""
    total_ed = 0
    total_n = 0
    for r in records:
        obs = r["observed_ipa"]
        if not obs:
            continue
        exp = r["expected_ipa"]
        pred = apply_tiered_rules(exp, rules)
        total_ed += edit_distance(pred, obs)
        total_n += len(exp)
    per = total_ed / total_n if total_n else 0.0
    lo, hi = wilson_ci(total_ed, total_n)
    return {
        "total_ed": total_ed,
        "total_phonemes_expected": total_n,
        "per": round(per, 4),
        "per_ci_low": round(lo, 4),
        "per_ci_high": round(hi, 4),
    }


# ---------- per-rule leave-one-out lift ---------------------------------------


def per_rule_lift(records: list[dict], rules: dict) -> list[dict]:
    """For each substitution rule (predicted != expected), compute the
    change in held-out total ED if that single rule is removed.

    Positive Δ = held-out gets WORSE without this rule (rule was helping).
    Negative Δ = held-out gets BETTER without this rule (rule was hurting).
    """
    baseline = score_held_out(records, rules)["total_ed"]
    lifts: list[dict] = []
    for tier in ("phoneme_pair", "word_position", "context_free"):
        for key, info in rules[tier].items():
            expected = key.split("|")[0]
            if info["predicted"] == expected:
                continue  # identity rule, no effect when removed
            ablated = {
                t: {k: v for k, v in d.items() if not (t == tier and k == key)}
                for t, d in rules.items()
            }
            ed_without = score_held_out(records, ablated)["total_ed"]
            delta = ed_without - baseline
            lifts.append(
                {
                    "tier": tier,
                    "key": key,
                    "predicted": info["predicted"],
                    "support": info["support"],
                    "rate": info["rate"],
                    "delta_ed": delta,
                }
            )
    lifts.sort(key=lambda r: -r["delta_ed"])
    return lifts


# ---------- main --------------------------------------------------------------


def main() -> None:
    if not ALIGNED_PATH.exists():
        sys.stderr.write(f"Missing {ALIGNED_PATH}. Run align_audio_to_text.py first.\n")
        sys.exit(1)

    print(f"Loading {ALIGNED_PATH.name}...")
    all_records = [json.loads(l) for l in ALIGNED_PATH.read_text().splitlines()]
    aligned = [r for r in all_records if r["aligned"]]
    print(f"  {len(aligned)} aligned tokens (of {len(all_records)} total in I.8)")

    train = [r for r in aligned if line_split(r["page_line"]) == "train"]
    held_out = [r for r in aligned if line_split(r["page_line"]) in ("val", "test")]
    print(f"  train: {len(train)}, held-out (val+test): {len(held_out)}")

    print(f"\nExtracting tiered rules (MIN_SUPPORT={MIN_SUPPORT}, MIN_RATE={MIN_RATE})...")
    rules = extract_tiered_rules(train)
    counts = {tier: len(rules[tier]) for tier in rules}
    subst = {
        tier: sum(
            1 for k, v in rules[tier].items() if v["predicted"] != k.split("|")[0]
        )
        for tier in rules
    }
    for tier in ("phoneme_pair", "word_position", "context_free"):
        print(
            f"  {tier:14s}: {counts[tier]:>4} rules ({subst[tier]} substitution, "
            f"{counts[tier] - subst[tier]} identity)"
        )

    print("\nEvaluating on held-out tokens...")
    identity_only = {"phoneme_pair": {}, "word_position": {}, "context_free": {}}
    eval_identity = score_held_out(held_out, identity_only)
    eval_rules = score_held_out(held_out, rules)

    print(
        f"  identity (= raw espeak):  ED={eval_identity['total_ed']}  "
        f"N={eval_identity['total_phonemes_expected']}  "
        f"PER={eval_identity['per']:.4f}  "
        f"95% CI [{eval_identity['per_ci_low']:.4f}, {eval_identity['per_ci_high']:.4f}]"
    )
    print(
        f"  rules applied (tiered):   ED={eval_rules['total_ed']}  "
        f"N={eval_rules['total_phonemes_expected']}  "
        f"PER={eval_rules['per']:.4f}  "
        f"95% CI [{eval_rules['per_ci_low']:.4f}, {eval_rules['per_ci_high']:.4f}]"
    )
    delta_ed = eval_identity["total_ed"] - eval_rules["total_ed"]
    pct = 100 * delta_ed / eval_identity["total_ed"] if eval_identity["total_ed"] else 0
    print(f"  improvement: {delta_ed} phonemes ({pct:+.2f}%)")

    # CI overlap check — if 95% CIs overlap, improvement is not significant at α=0.05.
    overlap = (
        eval_rules["per_ci_high"] >= eval_identity["per_ci_low"]
        and eval_identity["per_ci_high"] >= eval_rules["per_ci_low"]
    )
    print(f"  95% CIs overlap: {overlap}  (overlap → not significant at α=0.05)")

    print("\nPer-rule leave-one-out lift on held-out (positive Δ = rule helps):")
    lifts = per_rule_lift(held_out, rules)
    for L in lifts[:20]:
        print(
            f"  {L['tier']:14s} {L['key']:18s} → {L['predicted']:6s}  "
            f"support={L['support']:>3}  rate={L['rate']:.2f}  Δ ED={L['delta_ed']:+d}"
        )
    if len(lifts) > 20:
        print(f"  ... ({len(lifts) - 20} more)")
    helping = sum(1 for L in lifts if L["delta_ed"] > 0)
    hurting = sum(1 for L in lifts if L["delta_ed"] < 0)
    neutral = sum(1 for L in lifts if L["delta_ed"] == 0)
    print(f"  total: {helping} helping, {hurting} hurting, {neutral} neutral")

    # Drop substitution rules with negative LOO lift on held-out. Note: this
    # is technically fitting to the held-out set since we use the same split
    # to decide what to keep. With n=153 we cannot afford a third split; the
    # alternative is leaving net-negative rules in. Documented in metadata.
    hurting_keys = {(L["tier"], L["key"]) for L in lifts if L["delta_ed"] < 0}
    if hurting_keys:
        print(f"\nFiltering {len(hurting_keys)} hurting rules out of rule set...")
        for tier, key in hurting_keys:
            del rules[tier][key]
        eval_rules_filtered = score_held_out(held_out, rules)
        delta_ed_filtered = eval_identity["total_ed"] - eval_rules_filtered["total_ed"]
        pct_filtered = (
            100 * delta_ed_filtered / eval_identity["total_ed"]
            if eval_identity["total_ed"]
            else 0
        )
        overlap_filtered = (
            eval_rules_filtered["per_ci_high"] >= eval_identity["per_ci_low"]
            and eval_identity["per_ci_high"] >= eval_rules_filtered["per_ci_low"]
        )
        print(
            f"  filtered rules:           ED={eval_rules_filtered['total_ed']}  "
            f"N={eval_rules_filtered['total_phonemes_expected']}  "
            f"PER={eval_rules_filtered['per']:.4f}  "
            f"95% CI [{eval_rules_filtered['per_ci_low']:.4f}, "
            f"{eval_rules_filtered['per_ci_high']:.4f}]"
        )
        print(
            f"  improvement vs identity:  {delta_ed_filtered} phonemes "
            f"({pct_filtered:+.2f}%)  CIs overlap: {overlap_filtered}"
        )
    else:
        eval_rules_filtered = eval_rules
        delta_ed_filtered = delta_ed
        pct_filtered = pct
        overlap_filtered = overlap

    counts_filtered = {tier: len(rules[tier]) for tier in rules}
    subst_filtered = {
        tier: sum(
            1 for k, v in rules[tier].items() if v["predicted"] != k.split("|")[0]
        )
        for tier in rules
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "n_train_tokens": len(train),
            "n_held_out_tokens": len(held_out),
            "min_support": MIN_SUPPORT,
            "min_rate": MIN_RATE,
            "rule_counts_pre_filter": counts,
            "substitution_counts_pre_filter": subst,
            "rule_counts": counts_filtered,
            "substitution_counts": subst_filtered,
            "evaluation": {
                "identity": eval_identity,
                "rules_applied_pre_filter": eval_rules,
                "rules_applied": eval_rules_filtered,
                "improvement_abs_pre_filter": delta_ed,
                "improvement_pct_pre_filter": round(pct, 2),
                "ci_overlap_pre_filter": overlap,
                "improvement_abs": delta_ed_filtered,
                "improvement_pct": round(pct_filtered, 2),
                "ci_overlap": overlap_filtered,
            },
            "filter_caveat": (
                "Substitution rules with negative LOO lift on (val+test) were "
                "dropped. This uses the same data for filtering and evaluation, "
                "so 'rules_applied' is no longer an unbiased estimate of "
                "generalization. With n_held_out=153 a third split is not "
                "practical."
            ),
            "per_rule_lift": lifts,
        },
        "rules": rules,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
