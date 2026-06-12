#!/usr/bin/env python3
"""
Align an audio-derived phoneme stream to expected text via Smith-Waterman.

Pipeline:
  1. Load audio phonemes (output of transcribe_audio_phonemes.py).
  2. Load orthographic tokens. Two sources are supported:
       a. FW corpus chapter (default): iter_lines via shared.corpus, with
          hyphen-rejoining across line breaks. page_line is real FW citation.
       b. External text file (--text-file): plain text, whitespace-tokenized,
          paragraph-aware. page_line is synthesized as "paragraph.position"
          for traceability.
  3. Run espeak G2P per token to build the expected phoneme stream, with a
     parallel array recording which token each expected phoneme belongs to.
  4. Smith-Waterman local alignment of observed (audio) phonemes against
     expected (text→G2P) phonemes, via BioPython's C-backed PairwiseAligner.
  5. For each ortho token, derive: observed IPA, start_s, end_s, edit distance
     between expected and observed.

The audio typically covers a subset of the expected text. Tokens outside the
alignment have aligned=false and null timestamps. The output is one record per
token, so the artifact is complete; downstream consumers filter to aligned=true
for the audible subset.

Disagreement between expected and observed IPA on aligned tokens is the
phonological signal of interest: where Joyce diverges from naive English G2P.

Output record schema:
  {
    "page_line": "213.11",
    "orth": "Well,",
    "expected_ipa": ["w", "ɛ", "l"],
    "observed_ipa": ["w", "ɛ", "l"],
    "start_s": 3.82,
    "end_s": 3.98,
    "aligned": true,
    "edit_distance": 0,
    "coverage": 1.0
  }

Usage:
    # I.8 (default — uses FW corpus)
    python scripts/preprocessing/audio/align_audio_to_text.py

    # Aeolus (external text file)
    python scripts/preprocessing/audio/align_audio_to_text.py \\
        --input  data/audio/joyce-1924-aeolus/alignment/aeolus_phonemes.jsonl \\
        --output data/audio/joyce-1924-aeolus/alignment/aeolus_tokens_aligned.jsonl \\
        --text-file data/raw/ulysses_aeolus_taylor.txt
"""

import argparse
import json
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Bio.Align import PairwiseAligner  # noqa: E402

from shared.corpus import iter_lines  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PHONEMES = REPO_ROOT / "data/audio/joyce-1929-alp/alignment/i8_phonemes.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/audio/joyce-1929-alp/alignment/i8_tokens_aligned.jsonl"

PUNCT = string.punctuation + "—–“”‘’«»…"


def collect_fw_tokens(book: int, episode: int) -> list[tuple[str, str]]:
    """Return [(page_line, orth_token), ...] for a FW chapter with line-break hyphens rejoined.

    Same token convention as shared.tokenizers.iter_tokens — a token ending in
    a single '-' (not '--') is carried to the next line's first token.
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


def collect_file_tokens(path: Path) -> list[tuple[str, str]]:
    """Return [(synth_page_line, orth_token), ...] from a plain text file.

    Lines starting with '---' or '#' or blank are treated as structural and
    skipped (so the file can carry a caveat header). Paragraphs are separated
    by blank lines. page_line is synthesized as f"{para_idx}.{tok_idx:02d}"
    so downstream tools that key on page_line have a stable handle.
    """
    raw = path.read_text(encoding="utf-8")
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith(("---", "#")):
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.extend(s.split())
    if current:
        paragraphs.append(current)

    tokens: list[tuple[str, str]] = []
    for p_idx, para in enumerate(paragraphs, start=1):
        for t_idx, tok in enumerate(para, start=1):
            tokens.append((f"{p_idx}.{t_idx:02d}", tok))
    return tokens


def g2p_tokens(tokens: list[str], language: str = "en-us") -> list[list[str]]:
    """Phonemize a list of orthographic tokens, returning per-token phoneme lists."""
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator

    backend = EspeakBackend(
        language,
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags",
    )
    sep = Separator(phone=" ", syllable="", word="|")
    # Strip surrounding punctuation; espeak handles internal punct OK
    clean = [t.strip(PUNCT) or t for t in tokens]
    ipa_strings = backend.phonemize(clean, separator=sep, strip=True)
    return [ipa.split() for ipa in ipa_strings]


def normalize_phoneme(p: str) -> str:
    """Strip stress markers so alignment compares segmental content only."""
    return p.replace("ˈ", "").replace("ˌ", "")


def build_alphabet(*sequences: list[str]) -> dict[str, str]:
    """Map each unique phoneme to a Private Use Area Unicode char for string-based alignment."""
    unique = sorted({normalize_phoneme(p) for seq in sequences for p in seq})
    if len(unique) > 0x1900:  # PUA has 6400 codepoints; we'd be in trouble
        raise RuntimeError(f"Too many unique phonemes ({len(unique)}) for PUA encoding")
    return {p: chr(0xE000 + i) for i, p in enumerate(unique)}


def encode(seq: list[str], alphabet: dict[str, str]) -> str:
    return "".join(alphabet[normalize_phoneme(p)] for p in seq)


def expected_to_observed_map(
    aligned_obs_blocks, aligned_exp_blocks, n_expected: int
) -> list[int | None]:
    """Build map[expected_idx] -> observed_idx (or None) from BioPython aligned blocks."""
    m: list[int | None] = [None] * n_expected
    for (o_s, o_e), (e_s, e_e) in zip(aligned_obs_blocks, aligned_exp_blocks):
        # Each block has equal length in both sequences (gaps appear between blocks).
        assert o_e - o_s == e_e - e_s
        for k in range(e_e - e_s):
            m[e_s + k] = o_s + k
    return m


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein on phoneme lists."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cost = 0 if normalize_phoneme(ai) == normalize_phoneme(bj) else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Align audio phonemes to expected text")
    parser.add_argument("--input", type=Path, default=DEFAULT_PHONEMES,
                        help="Audio-derived phoneme JSONL (from transcribe_audio_phonemes.py)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Token-aligned output JSONL")
    parser.add_argument("--text-file", type=Path, default=None,
                        help="External text file as expected source (overrides FW corpus).")
    parser.add_argument("--book", type=int, default=1, help="FW book (when not using --text-file)")
    parser.add_argument("--episode", type=int, default=8,
                        help="FW episode (when not using --text-file)")
    args = parser.parse_args()

    phonemes_path: Path = args.input.resolve()
    out_path: Path = args.output.resolve()

    if not phonemes_path.exists():
        sys.stderr.write(f"Missing {phonemes_path}\n")
        sys.exit(1)

    print(f"Loading audio phonemes from {phonemes_path.name}...")
    observed_records = [json.loads(line) for line in phonemes_path.read_text().splitlines()]
    observed_phs = [r["phoneme"] for r in observed_records]
    print(f"  {len(observed_phs)} audio phonemes")

    if args.text_file is not None:
        text_path = args.text_file.resolve()
        if not text_path.exists():
            sys.stderr.write(f"Missing text file: {text_path}\n")
            sys.exit(1)
        print(f"Loading tokens from text file {text_path.name} and running espeak G2P...")
        tokens = collect_file_tokens(text_path)
    else:
        print(f"Loading FW I.{args.episode} tokens (book {args.book}) and running espeak G2P...")
        tokens = collect_fw_tokens(args.book, args.episode)
    orths = [t for _, t in tokens]
    per_token_phs = g2p_tokens(orths)
    expected_phs: list[str] = []
    expected_token_idx: list[int] = []
    for ti, phs in enumerate(per_token_phs):
        for p in phs:
            expected_phs.append(p)
            expected_token_idx.append(ti)
    print(f"  {len(tokens)} tokens, {len(expected_phs)} expected phonemes")

    print("Encoding for alignment...")
    alphabet = build_alphabet(observed_phs, expected_phs)
    print(f"  phoneme alphabet size: {len(alphabet)}")
    obs_enc = encode(observed_phs, alphabet)
    exp_enc = encode(expected_phs, alphabet)

    print("Running Smith-Waterman local alignment...")
    aligner = PairwiseAligner()
    aligner.mode = "local"
    # Gentle scoring — the Wake's nonce vocabulary produces espeak G2P that
    # diverges from Joyce's actual phonology widely. Strict mismatch/gap
    # penalties cause SW to bail out of nonce-rich stretches and only anchor
    # the most-English-like sub-region. Looser scoring keeps the alignment
    # extending through divergent passages.
    aligner.match_score = 2
    aligner.mismatch_score = -0.5
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.3
    alignments = aligner.align(obs_enc, exp_enc)
    best = alignments[0]
    print(f"  best score: {best.score}")
    print(f"  observed span: {best.aligned[0][0][0]}–{best.aligned[0][-1][1]}")
    print(f"  expected span: {best.aligned[1][0][0]}–{best.aligned[1][-1][1]}")

    exp_to_obs = expected_to_observed_map(
        best.aligned[0], best.aligned[1], len(expected_phs)
    )

    print(f"Writing {out_path.name}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_aligned = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ti, (page_line, orth) in enumerate(tokens):
            exp_for_token = per_token_phs[ti]
            # Find which expected positions belong to this token
            exp_positions = [k for k, t in enumerate(expected_token_idx) if t == ti]
            obs_positions = [exp_to_obs[k] for k in exp_positions if exp_to_obs[k] is not None]

            if obs_positions:
                n_aligned += 1
                obs_for_token = [observed_phs[i] for i in obs_positions]
                start_s = observed_records[min(obs_positions)]["start_s"]
                end_s = observed_records[max(obs_positions)]["end_s"]
                coverage = len(obs_positions) / max(len(exp_for_token), 1)
                ed = edit_distance(exp_for_token, obs_for_token)
                record = {
                    "page_line": page_line,
                    "orth": orth,
                    "expected_ipa": exp_for_token,
                    "observed_ipa": obs_for_token,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "aligned": True,
                    "edit_distance": ed,
                    "coverage": round(coverage, 3),
                }
            else:
                record = {
                    "page_line": page_line,
                    "orth": orth,
                    "expected_ipa": exp_for_token,
                    "observed_ipa": [],
                    "start_s": None,
                    "end_s": None,
                    "aligned": False,
                    "edit_distance": None,
                    "coverage": 0.0,
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {n_aligned}/{len(tokens)} tokens aligned to audio")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
