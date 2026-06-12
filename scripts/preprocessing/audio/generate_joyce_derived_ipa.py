#!/usr/bin/env python3
"""
Generate the unified Joyce-derived IPA artifact for the full Wake corpus.

For each token in every chapter, produces:
  - en-us-baseline:               naive espeak English G2P
  - en-us-joyce-rule-applied:     English IPA with Joyce phonological rules
                                  (data-driven from I.8 audio) applied
                                  (tiered: phoneme_pair → word_position →
                                  context_free → identity). Present for every
                                  token except where audio observation is
                                  available.
  - en-us-joyce-audio-observed:   wav2vec2-CTC observation from Joyce's 1929
                                  recording. Present only for the ~1032
                                  aligned I.8 tokens.
  - en-us-hiberno-english:        English IPA with literature-grounded Dublin
                                  Hiberno-English rules (Wells 1982, Hickey
                                  2007) applied. Same tiered logic, independent
                                  rule source. Present for every token. Serves
                                  as an ablation against the data-driven
                                  Joyce rules.
  - {lang}-baseline:              per-language espeak G2P for each non-English
                                  language FWEET tags as active on the same
                                  page-line

Per-token FWEET metadata is also attached:
  - fweet_languages: {espeak_code: {confidence, source_form, edit_distance, gloss}}
    where confidence ∈ {high, medium, low, line-only} based on orthographic
    similarity between this token and the elucidation's source_form.
  - fweet_motifs:   list of motif labels active on the page-line
  - fweet_glosses:  list of "lang_label source_form: gloss" strings

Output: data/ipa/joyce-derived/book{NN}_ep{NN}.jsonl

Pipeline:
  1. Load Joyce rules + FWEET index + aligned I.8 audio observations.
  2. Pass 1: walk the corpus, collect every (token, language) pair to phonemize.
  3. Pass 2: batch-phonemize per language (one espeak backend per language).
  4. Pass 3: walk the corpus again, assemble records, write per-chapter JSONL.

Requires: espeak-ng on PATH, phonemizer in .venv.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.corpus import iter_lines, load_all  # noqa: E402
from shared.fweet import load_by_page_line  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RULES_PATH = REPO_ROOT / "data/audio/joyce-1929-alp/rules/joyce_rules.json"
HIBERNO_RULES_PATH = REPO_ROOT / "data/annotations/hiberno_english_rules.json"
ALIGNED_PATH = REPO_ROOT / "data/audio/joyce-1929-alp/alignment/i8_tokens_aligned.jsonl"
OUT_DIR = REPO_ROOT / "data/ipa/joyce-derived"


# Confidence thresholds for token ↔ source-form orthographic similarity.
HIGH_CONF_MAX = 0.25  # normalized edit distance below this → high confidence
MED_CONF_MAX = 0.50

WORD_BOUNDARY = "#"


def normalize(p: str) -> str:
    return p.replace("ˈ", "").replace("ˌ", "")


def _as_list(predicted) -> list[str]:
    """Normalize a rule's 'predicted' field to a list of phonemes.

    A string is treated as one phoneme; a list is returned as-is; an empty
    string or empty list means deletion (the input phoneme is dropped).
    """
    if isinstance(predicted, str):
        return [predicted] if predicted else []
    return list(predicted)


def apply_tiered_rules(expected: list[str], rules: dict) -> list[str]:
    """Apply tiered rules: phoneme_pair → word_position → context_free.

    Context is computed from the expected (espeak) sequence. Falls back to
    identity if no rule fires at any tier. Rules may map one input phoneme
    to zero (deletion), one (substitution), or many (insertion) output
    phonemes. Mirrors extract_joyce_rules.py.
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

        if pair_key in rules.get("phoneme_pair", {}):
            out.extend(_as_list(rules["phoneme_pair"][pair_key]["predicted"]))
        elif pos_key in rules.get("word_position", {}):
            out.extend(_as_list(rules["word_position"][pos_key]["predicted"]))
        elif e in rules.get("context_free", {}):
            out.extend(_as_list(rules["context_free"][e]["predicted"]))
        else:
            out.append(e)
    return out


def edit_distance_str(s1: str, s2: str) -> int:
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i] + [0] * len(s2)
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def confidence_for(token: str, source_form: str | None) -> tuple[str, int]:
    """Compute (confidence_label, edit_distance) for orthographic similarity.

    Strips non-alpha characters from both, then compares. Returns "line-only"
    if no source_form is available (we still know the language is active on
    the line, but not specifically for this token).
    """
    if not source_form:
        return "line-only", -1
    t = "".join(c for c in token.lower() if c.isalpha())
    sf = "".join(c for c in source_form.lower() if c.isalpha())
    if not t or not sf:
        return "line-only", -1
    ed = edit_distance_str(t, sf)
    norm = ed / max(len(t), len(sf))
    if norm < HIGH_CONF_MAX:
        return "high", ed
    if norm < MED_CONF_MAX:
        return "medium", ed
    return "low", ed


def _confidence_rank(c: str) -> int:
    return {"line-only": 0, "low": 1, "medium": 2, "high": 3}.get(c, -1)


def collect_chapter_tokens(book: int, episode: int) -> list[tuple[str, str]]:
    """Walk a chapter's lines, yield (page_line, orth) with line-break hyphens rejoined.

    Same convention as shared.tokenizers.iter_tokens / align_audio_to_text.
    """
    tokens: list[tuple[str, str]] = []
    carry_tok = ""
    carry_pl = ""
    for line in iter_lines(books=[book], episodes=[(book, episode)]):
        if not line.text:
            continue
        toks = line.text.split()
        if not toks:
            continue
        if carry_tok:
            # Rejoined token keeps the page.line where the word *starts*
            # (matches FWEET's indexing convention for hyphenated words).
            tokens.append((carry_pl, carry_tok + toks[0]))
            toks = toks[1:]
            carry_tok = ""
            carry_pl = ""
        if toks and toks[-1].endswith("-") and not toks[-1].endswith("--"):
            # Drop the typographic line-break hyphen.
            carry_tok = toks[-1][:-1]
            carry_pl = line.page_line
            toks = toks[:-1]
        for t in toks:
            tokens.append((line.page_line, t))
    if carry_tok:
        tokens.append((carry_pl, carry_tok))
    return tokens


def phonemize_tokens(tokens: list[str], language: str) -> dict[str, list[str]]:
    """Batch-phonemize a list of tokens in the given espeak language.

    Returns dict {token: [phonemes]}. Strips surrounding punctuation before
    phonemization; empty tokens map to [].
    """
    import string

    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    PUNCT = string.punctuation + "—–“”‘’«»…"
    sep = Separator(phone=" ", syllable="", word="|")

    try:
        backend = EspeakBackend(
            language,
            preserve_punctuation=False,
            with_stress=False,
            language_switch="remove-flags",
        )
    except RuntimeError as e:
        sys.stderr.write(f"WARN: espeak language {language!r} unavailable: {e}\n")
        return {t: [] for t in tokens}

    clean_tokens = [t.strip(PUNCT) or t for t in tokens]
    results = backend.phonemize(clean_tokens, separator=sep, strip=True)
    return {orig: ipa.split() for orig, ipa in zip(tokens, results)}


def load_aligned_i8_observations() -> dict[int, dict]:
    """Return dict mapping I.8 token-index (0-based) to its alignment record.

    The aligned file was generated with the same hyphen-rejoining iteration
    as collect_chapter_tokens(1, 8), so we can zip them by position. Includes
    only records where aligned=true (others have empty observed_ipa).
    """
    if not ALIGNED_PATH.exists():
        return {}
    out: dict[int, dict] = {}
    with ALIGNED_PATH.open() as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r["aligned"]:
                out[i] = r
    return out


def main() -> None:
    print("Loading Joyce rules (data-driven from I.8 audio)...")
    rules_data = json.loads(RULES_PATH.read_text())
    rules = rules_data["rules"]
    for tier in ("phoneme_pair", "word_position", "context_free"):
        tier_rules = rules.get(tier, {})
        n_subst = sum(
            1 for k, v in tier_rules.items() if v["predicted"] != k.split("|")[0]
        )
        print(f"  {tier:14s}: {len(tier_rules)} rules ({n_subst} substitution)")

    print("Loading Hiberno-English rules (literature-grounded)...")
    he_rules_data = json.loads(HIBERNO_RULES_PATH.read_text())
    he_rules = he_rules_data["rules"]
    for tier in ("phoneme_pair", "word_position", "context_free"):
        tier_rules = he_rules.get(tier, {})
        n_subst = sum(
            1 for k, v in tier_rules.items() if v["predicted"] != k.split("|")[0]
        )
        print(f"  {tier:14s}: {len(tier_rules)} rules ({n_subst} substitution)")

    print("Loading FWEET index...")
    fweet_by_pl = load_by_page_line()
    print(f"  {len(fweet_by_pl):,} page-lines with elucidations")

    print("Loading aligned I.8 audio observations...")
    aligned_i8 = load_aligned_i8_observations()
    print(f"  {len(aligned_i8)} tokens with audio-observed IPA")

    chapters = list(load_all())
    print(f"\nCorpus: {len(chapters)} chapters")

    # ---------- Pass 1: collect (token, language) pairs to phonemize ----------
    print("\nPass 1: collecting (token, language) phonemize pairs...")
    pairs_by_lang: dict[str, set[str]] = defaultdict(set)
    chapter_tokens: dict[tuple[int, int], list[tuple[str, str]]] = {}

    for chapter in chapters:
        toks = collect_chapter_tokens(chapter.book, chapter.episode)
        chapter_tokens[(chapter.book, chapter.episode)] = toks
        for page_line, orth in toks:
            pairs_by_lang["en-us"].add(orth)
            for elu in fweet_by_pl.get(page_line, ()):
                if elu.espeak_code:
                    pairs_by_lang[elu.espeak_code].add(orth)

    total_tokens = sum(len(v) for v in chapter_tokens.values())
    pair_counts = {lang: len(toks) for lang, toks in pairs_by_lang.items()}
    print(f"  total corpus tokens: {total_tokens:,}")
    print(f"  unique (token, language) pairs to phonemize:")
    for lang, n in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {lang:8s}: {n:,}")

    # ---------- Pass 2: batch-phonemize per language ----------
    print("\nPass 2: batch-phonemizing per language...")
    ipa_cache: dict[tuple[str, str], list[str]] = {}
    for lang in sorted(pairs_by_lang, key=lambda L: -len(pairs_by_lang[L])):
        tokens_list = sorted(pairs_by_lang[lang])
        print(f"  {lang:8s}: phonemizing {len(tokens_list):,} unique tokens...", flush=True)
        results = phonemize_tokens(tokens_list, lang)
        for tok, ipa in results.items():
            ipa_cache[(tok, lang)] = ipa

    # ---------- Pass 3: assemble + write ----------
    print("\nPass 3: assembling records and writing per-chapter JSONL...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_with_fweet_lang = 0

    for chapter in chapters:
        toks = chapter_tokens[(chapter.book, chapter.episode)]
        out_path = OUT_DIR / f"book{chapter.book:02d}_ep{chapter.episode:02d}.jsonl"

        is_i8 = chapter.book == 1 and chapter.episode == 8

        with out_path.open("w", encoding="utf-8") as f:
            for token_idx, (page_line, orth) in enumerate(toks):
                en_baseline = ipa_cache.get((orth, "en-us"), [])

                ipa_dict: dict[str, list[str]] = {
                    "en-us-baseline": en_baseline,
                    "en-us-hiberno-english": apply_tiered_rules(en_baseline, he_rules),
                }
                if is_i8 and token_idx in aligned_i8:
                    ipa_dict["en-us-joyce-audio-observed"] = aligned_i8[token_idx][
                        "observed_ipa"
                    ]
                    joyce_source = "audio-observed"
                else:
                    ipa_dict["en-us-joyce-rule-applied"] = apply_tiered_rules(
                        en_baseline, rules
                    )
                    joyce_source = "rule-applied"

                fweet_languages: dict[str, dict] = {}
                motifs: list[str] = []
                glosses: list[str] = []

                for elu in fweet_by_pl.get(page_line, ()):
                    # Motifs (any kind=motif tag)
                    for tag in elu.tags:
                        if tag.kind == "motif":
                            motifs.append(tag.label)
                    # Glosses
                    if elu.language_code and elu.source_form and elu.gloss:
                        glosses.append(
                            f"{elu.tags[0].label} {elu.source_form}: {elu.gloss}"
                        )
                    # Per-language confidence (keep best confidence per language)
                    if elu.espeak_code:
                        conf, ed = confidence_for(orth, elu.source_form)
                        existing = fweet_languages.get(elu.espeak_code)
                        if existing is None or _confidence_rank(conf) > _confidence_rank(
                            existing["confidence"]
                        ):
                            fweet_languages[elu.espeak_code] = {
                                "confidence": conf,
                                "source_form": elu.source_form,
                                "edit_distance": ed,
                                "gloss": elu.gloss,
                            }
                        # Add the per-language IPA to the ipa dict
                        key = f"{elu.espeak_code}-baseline"
                        if key not in ipa_dict:
                            ipa_dict[key] = ipa_cache.get((orth, elu.espeak_code), [])

                record = {
                    "orth": orth,
                    "page_line": page_line,
                    "ipa": ipa_dict,
                    "joyce_ipa_source": joyce_source,
                    "fweet_languages": fweet_languages,
                    "fweet_motifs": sorted(set(motifs)),
                    "fweet_glosses": glosses,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
                if fweet_languages:
                    total_with_fweet_lang += 1

        print(f"  → {out_path.relative_to(REPO_ROOT)} ({len(toks):,} tokens)")

    print(f"\nDone.")
    print(f"  total tokens written:           {total_written:,}")
    print(f"  tokens with ≥1 FWEET language:  {total_with_fweet_lang:,} "
          f"({100 * total_with_fweet_lang / total_written:.1f}%)")


if __name__ == "__main__":
    main()
