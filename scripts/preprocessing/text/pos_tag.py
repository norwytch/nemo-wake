#!/usr/bin/env python3
"""
POS-tag the Wake via multiple parallel methods.

The artifact records DISAGREEMENT across methods rather than a single tag,
in recognition that Wake tokens are often multi-functional (a portmanteau
may function as a noun in one reading and a verb in another). No method is
treated as authoritative; downstream consumers see all hypotheses with
provenance.

Methods (per token):
  1. surface              — Stanza English POS on the FW line (line-as-sentence,
                            whole line passed pre-tokenized so corpus
                            tokenization is preserved).
  2. fweet_source_form    — Stanza per-language POS on each FWEET source-form
                            in its OWN language.
  3. fweet_gloss          — Stanza English POS on each FWEET English gloss.
  4. morpheme             — Stanza English POS on each morpheme, one record
                            per (segmenter, morph). Segmenters: BPE 4k,
                            Unigram 4k, Morfessor Baseline, FlatCat.

Tagset: Universal Dependencies (Stanza's default).

Caveats:
  - Stanza's English POS tagger is trained on standard English. Surface tags
    on nonce Wake forms will be low-quality, predictable failure modes:
    word-shape heuristics and contextual fallbacks. We emit them anyway —
    that's the methodological point of this artifact.
  - Morpheme POS is the most experimental method. Off-the-shelf taggers
    aren't designed for isolated morphemes; results are largely word-shape
    guesses. Treat as low-confidence by construction.
  - Stanza models for some FWEET languages may not be available. Missing
    models log a warning and skip that language's source-form tagging
    (without breaking the rest of the pipeline).

Output: data/pos/joyce-pos-hypotheses/book{NN}_ep{NN}.jsonl
  One record per token with a list of POS hypotheses. See README.

Usage:
    python scripts/preprocessing/text/pos_tag.py                          # full corpus
    python scripts/preprocessing/text/pos_tag.py --book 1 --episode 1     # one chapter (smoke)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.corpus import iter_lines, load_all  # noqa: E402
from shared.fweet import load_by_page_line  # noqa: E402
from shared.tokenizers import (  # noqa: E402
    load_bpe,
    load_flatcat,
    load_morfessor,
    load_unigram,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "data/pos/joyce-pos-hypotheses"

# Map FWEET espeak codes to Stanza language codes. Stanza generally uses
# ISO-639 short codes; FWEET espeak codes mostly already match but a few
# need rewriting.
ESPEAK_TO_STANZA: dict[str, str] = {
    "en-us": "en",
    "fr-fr": "fr",
    "cmn": "zh-hans",  # Mandarin → Simplified Chinese pipeline
    "nb": "no",  # Norwegian Bokmål → Stanza's combined Norwegian
}


def stanza_code(espeak: str) -> str:
    return ESPEAK_TO_STANZA.get(espeak, espeak)


# Orthographic-similarity threshold for attaching a FWEET source-form / gloss
# to a specific token (mirrors generate_joyce_derived_ipa.py). Below this
# normalized edit distance, the source-form is treated as a high-confidence
# match for this token; otherwise the FWEET elucidation is line-level only
# and should not generate a per-token POS hypothesis.
HIGH_CONF_MAX = 0.25


def _edit_distance(s1: str, s2: str) -> int:
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


def is_high_confidence_match(token: str, source_form: str) -> bool:
    """Strip non-alpha, compare lowercased. True if normalized ED < HIGH_CONF_MAX."""
    t = "".join(c for c in token.lower() if c.isalpha())
    sf = "".join(c for c in source_form.lower() if c.isalpha())
    if not t or not sf:
        return False
    ed = _edit_distance(t, sf)
    return ed / max(len(t), len(sf)) < HIGH_CONF_MAX


def collect_chapter_tokens(book: int, episode: int) -> list[tuple[str, str]]:
    """Walk a chapter's lines, yield (page_line, orth) per token.

    Hyphen-rejoined across line breaks (matches shared.tokenizers.iter_tokens
    and align_audio_to_text.collect_fw_tokens conventions).
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
            tokens.append((carry_pl, carry_tok + toks[0]))
            toks = toks[1:]
            carry_tok = ""
            carry_pl = ""
        if toks and toks[-1].endswith("-") and not toks[-1].endswith("--"):
            carry_tok = toks[-1][:-1]
            carry_pl = line.page_line
            toks = toks[:-1]
        for t in toks:
            tokens.append((line.page_line, t))
    if carry_tok:
        tokens.append((carry_pl, carry_tok))
    return tokens


def segment_morpheme(orth: str, segmenters: dict) -> dict[str, list[str]]:
    """Run all available segmenters on a token. Returns {segmenter_name: [morphs]}."""
    out: dict[str, list[str]] = {}
    if "bpe-4000" in segmenters:
        out["bpe-4000"] = segmenters["bpe-4000"].encode(orth).tokens
    if "unigram-4000" in segmenters:
        out["unigram-4000"] = segmenters["unigram-4000"].encode(orth).tokens
    if "morfessor" in segmenters:
        morphs, _ = segmenters["morfessor"].viterbi_segment(orth)
        out["morfessor"] = list(morphs)
    if "flatcat" in segmenters:
        morphs, _ = segmenters["flatcat"].viterbi_segment(orth)
        out["flatcat"] = [
            m if isinstance(m, str) else getattr(m, "morph", str(m)) for m in morphs
        ]
    return out


def load_stanza_pipeline(lang: str, processors: str = "tokenize,pos") -> object | None:
    """Lazily load a Stanza pipeline; download models if absent. Returns None on failure."""
    import stanza

    try:
        return stanza.Pipeline(
            lang=lang,
            processors=processors,
            verbose=False,
            tokenize_pretokenized=True,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        )
    except Exception as e:
        sys.stderr.write(f"  WARN: failed to load Stanza pipeline for {lang!r}: {e}\n")
        sys.stderr.write("  attempting download...\n")
        try:
            stanza.download(lang, verbose=False)
            return stanza.Pipeline(
                lang=lang,
                processors=processors,
                verbose=False,
                tokenize_pretokenized=True,
            )
        except Exception as e2:
            sys.stderr.write(f"  WARN: download failed for {lang!r}: {e2}\n")
            return None


def tag_pretokenized(pipeline, tokens: list[str]) -> list[str]:
    """POS-tag a pre-tokenized sequence. Returns one UPOS tag per token (or empty)."""
    if not tokens or pipeline is None:
        return []
    doc = pipeline([tokens])
    tags: list[str] = []
    for sentence in doc.sentences:
        for w in sentence.words:
            tags.append(w.upos or "")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-method POS-hypothesis tagging of the Wake"
    )
    parser.add_argument("--book", type=int, default=None, help="Restrict to one book (smoke test)")
    parser.add_argument(
        "--episode", type=int, default=None, help="Restrict to one episode (with --book)"
    )
    args = parser.parse_args()

    print("Loading segmenters...")
    segmenters: dict = {}
    for name, loader in (
        ("bpe-4000", lambda: load_bpe(4000)),
        ("unigram-4000", lambda: load_unigram(4000)),
        ("morfessor", load_morfessor),
        ("flatcat", load_flatcat),
    ):
        try:
            segmenters[name] = loader()
            print(f"  {name} loaded")
        except Exception as e:
            sys.stderr.write(f"  WARN: {name} unavailable: {e}\n")

    print("\nLoading FWEET index...")
    fweet_by_pl = load_by_page_line()
    print(f"  {len(fweet_by_pl):,} page-lines with elucidations")

    # Pick chapters
    if args.book is not None:
        if args.episode is None:
            sys.stderr.write("--episode is required with --book\n")
            sys.exit(1)
        chapters = [c for c in load_all() if c.book == args.book and c.episode == args.episode]
    else:
        chapters = list(load_all())
    print(f"\nProcessing {len(chapters)} chapter(s)")

    # ---------- Pass 1: collect inputs to tag per language ----------
    print("\nPass 1: collecting per-language tagging inputs...")
    chapter_tokens: dict[tuple[int, int], list[tuple[str, str]]] = {}
    surface_lines: list[list[str]] = []  # one entry per line, list of token strs
    surface_token_to_line: list[tuple[int, int]] = []  # (line_idx, intra_line_idx) per global token

    source_form_inputs: dict[str, set[str]] = defaultdict(set)
    token_fweet_meta: dict[int, list[dict]] = defaultdict(list)
    gloss_inputs: set[str] = set()
    token_glosses: dict[int, list[str]] = defaultdict(list)
    morpheme_inputs: set[str] = set()
    token_morphs: dict[int, dict[str, list[str]]] = {}

    global_idx = 0

    for chapter in chapters:
        toks = collect_chapter_tokens(chapter.book, chapter.episode)
        chapter_tokens[(chapter.book, chapter.episode)] = toks
        cur_line_key: str | None = None
        cur_line_tokens: list[str] = []
        for page_line, orth in toks:
            if page_line != cur_line_key:
                if cur_line_key is not None:
                    surface_lines.append(cur_line_tokens)
                cur_line_key = page_line
                cur_line_tokens = []
            line_idx = len(surface_lines)  # index this token will end up at
            intra = len(cur_line_tokens)
            cur_line_tokens.append(orth)
            surface_token_to_line.append((line_idx, intra))
            # FWEET — only attach an elucidation to THIS token if its source_form
            # is a high-confidence orthographic match to the token. Without this
            # filter every token on a line inherits every elucidation on that
            # line (line-level → per-token broadcast), which produces noise.
            for elu in fweet_by_pl.get(page_line, ()):
                if not (elu.espeak_code and elu.source_form):
                    continue
                if not is_high_confidence_match(orth, elu.source_form):
                    continue
                source_form_inputs[elu.espeak_code].add(elu.source_form)
                token_fweet_meta[global_idx].append(
                    {"espeak_code": elu.espeak_code, "source_form": elu.source_form}
                )
                if elu.gloss:
                    gloss_inputs.add(elu.gloss)
                    token_glosses[global_idx].append(elu.gloss)
            # Morphemes
            segs = segment_morpheme(orth, segmenters)
            token_morphs[global_idx] = segs
            for morphs in segs.values():
                for m in morphs:
                    if m:
                        morpheme_inputs.add(m)
            global_idx += 1
        if cur_line_key is not None:
            surface_lines.append(cur_line_tokens)

    n_tokens = global_idx
    print(f"  total tokens: {n_tokens:,}")
    print(f"  surface lines: {len(surface_lines):,}")
    print(f"  unique glosses: {len(gloss_inputs):,}")
    print(f"  unique morphemes: {len(morpheme_inputs):,}")
    print("  unique source-forms by language:")
    for lang, sfs in sorted(source_form_inputs.items(), key=lambda kv: -len(kv[1])):
        print(f"    {lang:8s}: {len(sfs):,}")

    # ---------- Pass 2: load pipelines and tag ----------
    print("\nPass 2: tagging...")
    print("  Loading English (en) Stanza pipeline (first run downloads ~300 MB)...")
    en_pipeline = load_stanza_pipeline("en")

    print(f"  Surface tagging {len(surface_lines):,} lines via en...")
    surface_tags: dict[int, list[str]] = {}
    if en_pipeline is not None:
        for li, line_toks in enumerate(surface_lines):
            surface_tags[li] = tag_pretokenized(en_pipeline, line_toks)
    else:
        sys.stderr.write("  WARN: surface tagging skipped (no en pipeline)\n")

    print(f"  Gloss tagging {len(gloss_inputs):,} unique glosses via en...")
    gloss_tags: dict[str, list[tuple[str, str]]] = {}
    if en_pipeline is not None:
        for g in sorted(gloss_inputs):
            words = g.split()
            if not words:
                continue
            tags = tag_pretokenized(en_pipeline, words)
            gloss_tags[g] = list(zip(words, tags))

    print(f"  Morpheme tagging {len(morpheme_inputs):,} unique morphemes via en...")
    morph_tags: dict[str, str] = {}
    if en_pipeline is not None:
        for m in sorted(morpheme_inputs):
            tags = tag_pretokenized(en_pipeline, [m])
            morph_tags[m] = tags[0] if tags else ""

    source_form_tags: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for espeak in sorted(source_form_inputs, key=lambda L: -len(source_form_inputs[L])):
        s_lang = stanza_code(espeak)
        print(
            f"  Source-form tagging {len(source_form_inputs[espeak]):,} forms via "
            f"{espeak} → stanza:{s_lang}..."
        )
        pipe = en_pipeline if s_lang == "en" and en_pipeline is not None else load_stanza_pipeline(s_lang)
        if pipe is None:
            sys.stderr.write(f"  WARN: skipping {espeak} source-form tagging\n")
            continue
        for sf in sorted(source_form_inputs[espeak]):
            words = sf.split()
            if not words:
                continue
            tags = tag_pretokenized(pipe, words)
            source_form_tags[(espeak, sf)] = list(zip(words, tags))

    # ---------- Pass 3: assemble per-token records ----------
    print("\nPass 3: assembling per-token records...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    total_written = 0
    for chapter in chapters:
        toks = chapter_tokens[(chapter.book, chapter.episode)]
        out_path = OUT_DIR / f"book{chapter.book:02d}_ep{chapter.episode:02d}.jsonl"

        with out_path.open("w", encoding="utf-8") as f:
            for page_line, orth in toks:
                hypotheses: list[dict] = []

                # Surface
                if global_idx < len(surface_token_to_line):
                    li, intra = surface_token_to_line[global_idx]
                    tags = surface_tags.get(li, [])
                    if intra < len(tags) and tags[intra]:
                        hypotheses.append(
                            {
                                "method": "surface",
                                "language": "en",
                                "input": orth,
                                "pos": [tags[intra]],
                            }
                        )

                # FWEET source-forms (dedup per token)
                seen_sf: set[tuple[str, str]] = set()
                for meta in token_fweet_meta.get(global_idx, ()):
                    key = (meta["espeak_code"], meta["source_form"])
                    if key in seen_sf:
                        continue
                    seen_sf.add(key)
                    sf_pos = source_form_tags.get(key, [])
                    hypotheses.append(
                        {
                            "method": "fweet_source_form",
                            "language": meta["espeak_code"],
                            "input": meta["source_form"],
                            "pos": [pos for _w, pos in sf_pos if pos],
                        }
                    )

                # FWEET glosses (dedup per token)
                seen_g: set[str] = set()
                for g in token_glosses.get(global_idx, ()):
                    if g in seen_g:
                        continue
                    seen_g.add(g)
                    g_pos = gloss_tags.get(g, [])
                    hypotheses.append(
                        {
                            "method": "fweet_gloss",
                            "language": "en",
                            "input": g,
                            "pos": [pos for _w, pos in g_pos if pos],
                        }
                    )

                # Morphemes (one hypothesis per (segmenter, morph))
                for seg_name, morphs in token_morphs.get(global_idx, {}).items():
                    for m in morphs:
                        if not m:
                            continue
                        m_pos = morph_tags.get(m, "")
                        hypotheses.append(
                            {
                                "method": "morpheme",
                                "segmenter": seg_name,
                                "language": "en",
                                "input": m,
                                "pos": [m_pos] if m_pos else [],
                            }
                        )

                record = {
                    "orth": orth,
                    "page_line": page_line,
                    "hypotheses": hypotheses,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_written += 1
                global_idx += 1
        print(f"  → {out_path.relative_to(REPO_ROOT)} ({len(toks):,} tokens)")

    print(f"\nDone. {total_written:,} tokens written.")


if __name__ == "__main__":
    main()
