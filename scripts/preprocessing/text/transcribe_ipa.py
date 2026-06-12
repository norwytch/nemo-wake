#!/usr/bin/env python3
"""
Transcribe the Finnegans Wake corpus to IPA via espeak-ng (multilingual G2P).

Reads data/raw/book{NN}_ep{NN}.jsonl, runs espeak-ng on each whitespace-split
token, writes data/ipa/{language}/book{NN}_ep{NN}.jsonl with per-token IPA.

The output is organized per-language by design — the polyglot Wake produces a
different IPA reading under each language's G2P rules, and the right artifact
is the *set* of those readings (eventually one per language that FWEET flags
for a given token). The en-us pass is the naive-English-G2P baseline.

Language caveat for the en-us pass: it treats the entire Wake as English.
This is wrong but a defensible baseline — English is the dominant matrix
language and per-token language detection (via FWEET annotations) is a
separate track. espeak phonemizes neologisms (passencore, mememormee) using
English G2P rules, producing a naive English reading. The result is most
useful in comparison to the per-language passes that will follow.

System dependency: espeak-ng must be installed.
    macOS:   brew install espeak-ng
    Linux:   apt install espeak-ng

Usage:
    python scripts/preprocessing/transcribe_ipa.py
    python scripts/preprocessing/transcribe_ipa.py --language en-us
    python scripts/preprocessing/transcribe_ipa.py --force
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.ipa import ipa_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"


def check_espeak() -> None:
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        sys.stderr.write(
            "espeak-ng not found on PATH. Install with:\n"
            "  macOS:  brew install espeak-ng\n"
            "  Linux:  apt install espeak-ng\n"
        )
        sys.exit(1)


def transcribe_chapter(raw_path: Path, out_path: Path, backend, force: bool) -> int:
    if out_path.exists() and not force:
        n = sum(1 for _ in out_path.read_text().splitlines())
        print(f"  SKIP  {out_path.name} ({n} lines; use --force to re-do)")
        return 0

    raw_lines = [json.loads(s) for s in raw_path.read_text(encoding="utf-8").splitlines()]

    # Collect all tokens across the chapter for batch phonemization.
    all_tokens: list[str] = []
    origins: list[tuple[int, str]] = []  # (line_index, field)
    for li, r in enumerate(raw_lines):
        for field in ("text", "left_margin", "right_margin"):
            val = r.get(field)
            if not val:
                continue
            for tok in val.split():
                all_tokens.append(tok)
                origins.append((li, field))

    ipa_outputs = backend.phonemize(all_tokens, strip=True)
    assert len(ipa_outputs) == len(all_tokens), (
        f"phonemizer returned {len(ipa_outputs)} IPA strings for {len(all_tokens)} tokens"
    )

    per_line: list[dict[str, list[list[str]]]] = [
        {"text": [], "left_margin": [], "right_margin": []} for _ in raw_lines
    ]
    for (li, field), orth, ipa in zip(origins, all_tokens, ipa_outputs):
        per_line[li][field].append([orth, ipa])

    with out_path.open("w", encoding="utf-8") as f:
        for r, fields in zip(raw_lines, per_line):
            record = {
                "book": r["book"],
                "episode": r["episode"],
                "page": r["page"],
                "line": r["line"],
                "page_line": r["page_line"],
                "tokens": fields["text"],
                "left_margin_tokens": fields["left_margin"] or None,
                "right_margin_tokens": fields["right_margin"] or None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  → {out_path.relative_to(REPO_ROOT)} ({len(raw_lines)} lines)")
    return len(raw_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe the Wake corpus to IPA via espeak-ng"
    )
    parser.add_argument(
        "--language",
        default="en-us",
        help="espeak-ng language code (default: en-us; try fr-fr, de, it, la for re-passes)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-transcribe even if output exists"
    )
    args = parser.parse_args()

    check_espeak()

    from phonemizer.backend import EspeakBackend  # noqa: E402

    backend = EspeakBackend(args.language, preserve_punctuation=True, with_stress=True)

    out_dir = ipa_dir(args.language)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(RAW_DIR.glob("book*.jsonl"))
    if not raw_paths:
        sys.stderr.write(f"No raw chapters in {RAW_DIR}. Run scrape_trent.py first.\n")
        sys.exit(1)

    total = 0
    for raw_path in raw_paths:
        out_path = out_dir / raw_path.name
        total += transcribe_chapter(raw_path, out_path, backend, args.force)

    print(f"\nDone. {total} new lines transcribed (language={args.language}).")


if __name__ == "__main__":
    main()
