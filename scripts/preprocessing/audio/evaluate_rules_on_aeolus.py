#!/usr/bin/env python3
"""
Out-of-distribution evaluation of the I.8-derived Joyce rules on the
1924 Aeolus recording.

The rules in data/audio/joyce-1929-alp/rules/joyce_rules.json were extracted
from Joyce's 1929 reading of Anna Livia Plurabelle (FW I.8). This script
applies them to Joyce's 1924 reading of the John F. Taylor speech from
Ulysses and asks: do the rules generalize across recordings?

Reports identity (raw espeak) vs rules-applied edit distance + phoneme
error rate with Wilson 95% CIs, plus a per-rule leave-one-out lift on
the Aeolus alignment.

Important caveats:
  - Different text genres (Wake's nonce vocabulary vs Ulysses's mostly
    standard English). Espeak baseline behaves differently on each;
    interpreting the comparison requires care.
  - Different recording years (1924 vs 1929) and equipment (HMV electrical
    vs Great 78 acoustic). Acoustic differences may affect wav2vec2 output.
  - 1924 audio quality is generally worse; alignment coverage is partial.

Input rules:    data/audio/joyce-1929-alp/rules/joyce_rules.json
Input alignment: data/audio/joyce-1924-aeolus/alignment/aeolus_tokens_aligned.jsonl
Output:         stdout summary + data/audio/joyce-1924-aeolus/rules/aeolus_eval.json

Usage: python scripts/preprocessing/audio/evaluate_rules_on_aeolus.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the tiered-application + scoring code from extract_joyce_rules.
from extract_joyce_rules import (  # noqa: E402
    per_rule_lift,
    score_held_out,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RULES_PATH = REPO_ROOT / "data/audio/joyce-1929-alp/rules/joyce_rules.json"
ALIGNED_PATH = (
    REPO_ROOT / "data/audio/joyce-1924-aeolus/alignment/aeolus_tokens_aligned.jsonl"
)
OUT_PATH = REPO_ROOT / "data/audio/joyce-1924-aeolus/rules/aeolus_eval.json"


def main() -> None:
    if not RULES_PATH.exists():
        sys.stderr.write(f"Missing {RULES_PATH}\n")
        sys.exit(1)
    if not ALIGNED_PATH.exists():
        sys.stderr.write(f"Missing {ALIGNED_PATH}\n")
        sys.exit(1)

    print(f"Loading rules from {RULES_PATH.name}...")
    rules_data = json.loads(RULES_PATH.read_text())
    rules = rules_data["rules"]
    for tier in ("phoneme_pair", "word_position", "context_free"):
        n_subst = sum(
            1 for k, v in rules[tier].items() if v["predicted"] != k.split("|")[0]
        )
        print(f"  {tier:14s}: {len(rules[tier])} rules ({n_subst} substitution)")

    print(f"\nLoading Aeolus alignment from {ALIGNED_PATH.name}...")
    all_records = [json.loads(l) for l in ALIGNED_PATH.read_text().splitlines()]
    aligned = [r for r in all_records if r["aligned"] and r["observed_ipa"]]
    print(f"  {len(aligned)} aligned tokens (of {len(all_records)} total)")

    print("\nEvaluating on Aeolus alignment...")
    identity_only = {"phoneme_pair": {}, "word_position": {}, "context_free": {}}
    eval_identity = score_held_out(aligned, identity_only)
    eval_rules = score_held_out(aligned, rules)

    print(
        f"  identity (raw espeak):    ED={eval_identity['total_ed']}  "
        f"N={eval_identity['total_phonemes_expected']}  "
        f"PER={eval_identity['per']:.4f}  "
        f"95% CI [{eval_identity['per_ci_low']:.4f}, {eval_identity['per_ci_high']:.4f}]"
    )
    print(
        f"  rules applied (I.8-trained tiered): ED={eval_rules['total_ed']}  "
        f"N={eval_rules['total_phonemes_expected']}  "
        f"PER={eval_rules['per']:.4f}  "
        f"95% CI [{eval_rules['per_ci_low']:.4f}, {eval_rules['per_ci_high']:.4f}]"
    )
    delta_ed = eval_identity["total_ed"] - eval_rules["total_ed"]
    pct = 100 * delta_ed / eval_identity["total_ed"] if eval_identity["total_ed"] else 0
    overlap = (
        eval_rules["per_ci_high"] >= eval_identity["per_ci_low"]
        and eval_identity["per_ci_high"] >= eval_rules["per_ci_low"]
    )
    print(f"  improvement: {delta_ed} phonemes ({pct:+.2f}%)")
    print(f"  95% CIs overlap: {overlap}  (overlap → not significant at α=0.05)")

    print("\nPer-rule leave-one-out lift on Aeolus (positive Δ = rule helps):")
    lifts = per_rule_lift(aligned, rules)
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

    # Cross-recording comparison: pull I.8 held-out numbers from the rules file
    # metadata for direct side-by-side.
    i8_meta = rules_data["metadata"]["evaluation"]
    print("\nCross-recording comparison:")
    print(f"  {'Set':<22} {'PER':>8} {'95% CI':>22} {'Improvement':>14}")
    print(
        f"  {'I.8 held-out identity':<22} {i8_meta['identity']['per']:>8.4f}  "
        f"[{i8_meta['identity']['per_ci_low']:.4f}, "
        f"{i8_meta['identity']['per_ci_high']:.4f}]"
    )
    print(
        f"  {'I.8 held-out rules':<22} {i8_meta['rules_applied']['per']:>8.4f}  "
        f"[{i8_meta['rules_applied']['per_ci_low']:.4f}, "
        f"{i8_meta['rules_applied']['per_ci_high']:.4f}]"
        f"   {i8_meta['improvement_pct']:+.2f}%"
    )
    print(
        f"  {'Aeolus identity':<22} {eval_identity['per']:>8.4f}  "
        f"[{eval_identity['per_ci_low']:.4f}, {eval_identity['per_ci_high']:.4f}]"
    )
    print(
        f"  {'Aeolus rules':<22} {eval_rules['per']:>8.4f}  "
        f"[{eval_rules['per_ci_low']:.4f}, {eval_rules['per_ci_high']:.4f}]"
        f"   {pct:+.2f}%"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "rules_source": str(RULES_PATH.relative_to(REPO_ROOT)),
        "alignment_source": str(ALIGNED_PATH.relative_to(REPO_ROOT)),
        "n_aligned_tokens": len(aligned),
        "identity": eval_identity,
        "rules_applied": eval_rules,
        "improvement_abs": delta_ed,
        "improvement_pct": round(pct, 2),
        "ci_overlap": overlap,
        "per_rule_lift": lifts,
        "i8_held_out_for_reference": i8_meta,
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
